"""
Agent Node — v2 recursive unit of the HiveMind cluster.

Architecture (v2):
  ┌─────────────────────────────────────────────────────┐
  │  ROOT (orchestrator role)                           │
  │   • Classify: simple / ambiguous / complex          │
  │   • For complex: plan output_shape + file owners    │
  │   │                                                 │
  │   ├─ FILE_OWNER agents (analyst role)               │
  │   │   • Own one output file / domain                │
  │   │   • Spawn WORKER child agents (functions/chunks)│
  │   │   • Audit + merge their workers (upward-heavy)  │
  │   │                                                 │
  │   └─ ROOT final audit (orchestrator, upward-heavy)  │
  └─────────────────────────────────────────────────────┘

Model assignment:
  Downward (planning/decompose) → lighter: ANALYST or WORKER
  Upward   (audit/merge)        → heavier: ORCHESTRATOR or ANALYST

output_shape (declared by orchestrator at plan time):
  single_file   → one fenced block, assembled verbatim
  multi_file    → multiple named fenced blocks, assembled verbatim
  document      → prose, synthesize + dedup
  analysis      → conclusions + evidence only, no padding

Shell commands: ##RUN## blocks inside ##SOLVE## answers are still supported.
File extraction: fenced blocks with filenames are written to output_dir.
"""
import asyncio
import json
import re
import time
import sys as _sys
import uuid
import pathlib
import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

from core.llm import chat
from core.model_classes import ModelClass


def _debug_log(msg: str):
    entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n"
    pathlib.Path("hivemind_debug.log").open("a").write(entry)
    print(f"\033[2m{entry.strip()}\033[0m", file=_sys.stderr)


# ── Enums ─────────────────────────────────────────────────────────────────────

class AgentStatus(Enum):
    PENDING   = "pending"
    PLANNING  = "planning"
    RUNNING   = "running"
    MERGING   = "merging"
    DONE      = "done"
    ERROR     = "error"


class AgentRole(Enum):
    ORCHESTRATOR = "orchestrator"   # root — classify, plan, final audit
    FILE_OWNER   = "file_owner"     # owns one output domain, audits workers
    WORKER       = "worker"         # leaf — writes functions / content chunks
    AUDITOR      = "auditor"        # standalone audit pass (future use)


class OutputShape(Enum):
    SINGLE_FILE  = "single_file"    # one fenced block, verbatim
    MULTI_FILE   = "multi_file"     # multiple named fenced blocks, verbatim
    DOCUMENT     = "document"       # prose, synthesize + dedup
    ANALYSIS     = "analysis"       # conclusions + evidence, no padding


# ── Registry ──────────────────────────────────────────────────────────────────

_agent_registry: dict[str, "AgentNode"] = {}
MAX_TOTAL_AGENTS = 40


def _make_task_id(parent_id: Optional[str], child_index: int) -> str:
    if parent_id is None:
        return "task"
    return f"{parent_id}.{child_index + 1}"


# ── Workspace helpers ─────────────────────────────────────────────────────────

def _workspace_path(root_task: str, base_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    slug = re.sub(r"[^a-z0-9]+", "_", root_task.lower())[:48].strip("_")
    ws = (base_dir or pathlib.Path.cwd()) / "hivemind_workspace"
    ws.mkdir(exist_ok=True)
    return ws / f"{slug}.md"


def _read_workspace(path: pathlib.Path) -> str:
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def _write_workspace(path: pathlib.Path, content: str):
    path.write_text(content)


def _append_workspace(path: pathlib.Path, section: str, content: str):
    existing = _read_workspace(path)
    entry = f"\n\n## {section}\n\n{content}"
    _write_workspace(path, existing + entry)


# ── File extraction ───────────────────────────────────────────────────────────

_FENCED_FILE = re.compile(
    r"```(?:[\w.+-]*?\s*)?(?:#\s*)?(?:file:\s*)?([^\n`\s][^\n`]*\.[a-zA-Z0-9]+)\n(.*?)```",
    re.DOTALL,
)
_HEADING_FENCED = re.compile(
    r"#{1,4}\s+([^\n`]+\.[a-zA-Z0-9]+)\s*\n```[\w]*\n(.*?)```",
    re.DOTALL,
)
_FENCED_LANG = re.compile(r"```(\w[^\n`]*)\n(.*?)```", re.DOTALL)
_LANG_EXT = {
    "python": "py", "py": "py", "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts", "bash": "sh", "sh": "sh",
    "html": "html", "css": "css", "json": "json", "yaml": "yml",
    "toml": "toml", "rust": "rs", "go": "go", "c": "c", "cpp": "cpp",
    "text": "txt", "markdown": "md", "md": "md",
}
_ALLOWED_EXTS = {
    "py", "js", "ts", "html", "css", "json", "yml", "yaml",
    "sh", "md", "txt", "rs", "go", "c", "cpp", "toml",
}

# Matches explicit paths in task text: ~/some/path/file.ext or /abs/path/file.ext
_TASK_PATH_RE = re.compile(r"(?:^|\s)(~[^\s]*\.[a-zA-Z0-9]+|/[^\s]*\.[a-zA-Z0-9]+)")


def _resolve_fname(fname: str, output_dir: pathlib.Path) -> pathlib.Path | None:
    fname = fname.strip().lstrip("#").strip()
    if "." not in fname:
        return None
    ext = fname.split(".")[-1].lower()
    if ext not in _ALLOWED_EXTS:
        return None
    p = pathlib.Path(fname).expanduser()
    if p.is_absolute():
        return p
    if " " in fname and not fname.startswith("/") and not fname.startswith("~"):
        return None
    return output_dir / fname


def _extract_and_write_files(result: str, output_dir: pathlib.Path,
                              root_task: str,
                              explicit_path: pathlib.Path | None = None) -> list[pathlib.Path]:
    written: list[pathlib.Path] = []
    seen: set[pathlib.Path] = set()

    def _write(fname: str, code: str):
        path = _resolve_fname(fname, output_dir)
        if path is None or path in seen:
            return
        seen.add(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        written.append(path)

    def _write_to(path: pathlib.Path, code: str):
        if path in seen:
            return
        seen.add(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        written.append(path)

    # Pattern 1: fenced blocks with filename on the opening line
    for m in _FENCED_FILE.finditer(result):
        _write(m.group(1), m.group(2))

    # Pattern 2: heading then fenced block
    if not written:
        for m in _HEADING_FENCED.finditer(result):
            _write(m.group(1), m.group(2))

    # Pattern 3: any fenced block by language
    if not written:
        slug = re.sub(r"[^a-z0-9]+", "_", root_task.lower())[:32].strip("_")
        for lang_tag, code in _extract_fenced_blocks(result):
            # lang_tag may be "python" or "python chess.py" — extract just the language word
            lang = lang_tag.split()[0].lower() if lang_tag else ""
            ext = _LANG_EXT.get(lang)
            if not ext:
                continue
            if explicit_path and explicit_path.suffix.lstrip(".") == ext:
                _write_to(explicit_path, code)
            else:
                fname = f"{slug}.{ext}"
                base, n = slug, 1
                while (output_dir / fname) in seen:
                    fname = f"{base}_{n}.{ext}"
                    n += 1
                _write(fname, code)

    # Pattern 4: explicit path requested but no fenced block matched
    if not written and explicit_path:
        ext = explicit_path.suffix.lstrip(".").lower()
        if ext in _ALLOWED_EXTS:
            _write_to(explicit_path, result)

    return written


# ── Shell execution ───────────────────────────────────────────────────────────

_RUN_MARKER = re.compile(r"##RUN##\s*\n(.*?)(?=\n##|\Z)", re.DOTALL)
_SUDO_RE    = re.compile(r"\bsudo\b")


async def _exec_command(
    cmd: str,
    cwd: Optional[pathlib.Path],
    sudo_callback: Optional[Callable],
) -> str:
    env_cmd = cmd.strip()
    stdin_data: Optional[bytes] = None

    if _SUDO_RE.search(env_cmd) and sudo_callback:
        password = await sudo_callback(env_cmd)
        if password is not None:
            env_cmd = env_cmd.replace("sudo ", "sudo -S ", 1)
            stdin_data = (password + "\n").encode()

    try:
        proc = await asyncio.create_subprocess_shell(
            env_cmd,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.PIPE if stdin_data else None,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(input=stdin_data), timeout=60)
        output = stdout.decode(errors="replace").strip()
        exit_code = proc.returncode
        result = f"[exit {exit_code}]\n{output}" if output else f"[exit {exit_code}]"
        _debug_log(f"RUN {env_cmd!r} → {result[:120]}")
        return result
    except asyncio.TimeoutError:
        return "[ERROR] command timed out after 60s"
    except Exception as e:
        return f"[ERROR] {e}"


# ── Orchestrator classify-and-plan ────────────────────────────────────────────

_CLASSIFY_SYSTEM = """\
You are HiveMind's Orchestrator. Classify the user's task and decide how to handle it.
Reply with valid JSON ONLY — no preamble, no explanation.

Classification rules:
  "simple"     — The work itself can be done in one short pass by a single agent.
                 Examples: Q&A, explanations, fixing a function, short essays,
                 a quick utility script (< ~80 lines), conversation, summaries.
  "ambiguous"  — The task is unclear, missing critical info, or could mean very
                 different things. Ask ONE clarifying question.
  "complex"    — The work has genuinely independent subsystems or sections that
                 benefit from parallel specialist agents — regardless of whether
                 the final output is one file or many.
                 Examples: games (board, rules, UI, game loop are independent),
                 compilers/interpreters, multi-section documents, full web apps,
                 any program whose subsystems can be built and tested in isolation.

  CRITICAL: "output is one file" does NOT mean simple. A chess engine, a web
  server, a language interpreter — these are complex even in one file because
  they have independent subsystems. Judge by the WORK, not the output count.

For "simple":
  {"route": "simple", "model_class": "worker"}

For "ambiguous":
  {"route": "ambiguous", "clarification": "One concise question."}

For "complex":
  {
    "route": "complex",
    "task_type": "<generate|debug>",
    "output_shape": "<single_file|multi_file|document|analysis>",
    "file_owners": [
      {
        "domain": "short_label",
        "task": "What this domain covers.",
        "depends_on": [],
        "workers": [
          {"task": "def fn_name(params) -> type: one sentence. Calls: [other_fn if needed].", "model_class": "worker"}
        ]
      }
    ],
    "interfaces": "KEEP UNDER 200 CHARS. No newlines. MUST specify: piece encoding, color values, coordinate types, board orientation. E.g.: 'board[rank][file] rank0=black_back rank7=white_back | piece:2ch piece[0]=type(rnbqkp) piece[1]=color | color:+/- ONLY | file:int 0-7 rank:int 0-7'. Omit for prose."
  }
  IMPORTANT: output_shape and file_owners MUST appear before interfaces in the JSON.
  The interfaces field is lowest priority — complete the plan first, add interfaces last.

task_type rules:
  "generate" — create new code from scratch (default)
  "debug"    — a file is provided and the task asks to fix, debug, or correct it.
               Detect: prompt contains a file path, or uses words like fix/debug/broken/crash/error.
               In debug mode: each worker fixes ONE specific bug in the provided file.
               Worker task field must include: the exact function name to fix, the bug description,
               and the correct signature. Workers output only the corrected function body.
               CRITICAL for debug: each function name must appear in AT MOST ONE worker across
               ALL file_owners. Never assign two workers to fix the same function.

output_shape guide:
  single_file  — all output assembles into one file (e.g. a chess game, a web page)
  multi_file   — output is multiple distinct named files
  document     — prose output (report, README, guide, essay)
  analysis     — structured analysis/conclusions (no filler prose)

Rules for "complex" code tasks:
  • Minimum 2 file_owners, maximum 6.
  • Each file_owner owns one logical subsystem and has 2-6 workers.
  • EACH WORKER WRITES EXACTLY ONE FUNCTION — the task field must contain the exact def signature.
  • Prefer MORE workers with SMALLER scope over fewer workers with larger scope.
    A worker that writes ~10-30 lines is far more reliable than one writing 80+ lines.
    Split generously: one function per worker, always.
  • Function names must be unique across ALL workers in the ENTIRE plan — no duplicates AT ALL.
    Before finalising the plan, scan every worker task and verify no function name appears twice.
    If two domains need the same operation, one calls the other's function by name.
  • All workers must use the exact types from the interfaces contract.
  • Tell workers what functions from other workers they may call, INCLUDING the full signature and return semantics:
    "Call get_piece_moves(board: list, file: int, rank: int) -> list[tuple]: returns list of valid moves."
    "Call add_task(tasks: list[dict], description: str) -> None: mutates tasks in-place, returns nothing — call as statement."
    NEVER write just a bare function name — always include the full def signature and state whether it returns a value or mutates in-place.
  • One worker per file_owner must implement the entry point. Name it EXACTLY `main()` — always, for every project. The auto-assembler looks for `main()` to add the `if __name__ == '__main__'` block. Never name it run_game, start_app, or anything else.
  • Any function that dispatches on 4+ cases (e.g. piece type switch) MUST be split:
    one worker per case, plus one thin dispatcher worker that calls them by name.
    NEVER assign a 6-case switch to a single worker — it will produce incomplete output.
  • Each file_owner must list domain names it depends on in `depends_on` (empty list if independent).
    If domain B needs types/functions from domain A, B must list A in depends_on.
    Order workers so dependencies come first (independent domains first).
  • file_owners run in parallel; workers within each file_owner run in parallel.
  • For single_file output: the assembly agent stitches all worker functions into one file.

Worker model_class assignment — choose based on function complexity:
  "worker"  — simple, well-defined functions: data init, simple lookups, parsing,
               short math, rendering. Expected output: ~10-30 lines.
               Examples: initialize_board, parse_move_input, find_king, render_board.
  "analyst" — complex functions requiring deep logic, multiple control paths, or
               coordination across many rules. Expected output: 40-100+ lines.
               Examples: get_piece_moves (all piece types), apply_move (castling+en passant+promotion),
               get_legal_moves (filter + simulate), run_game / main (full loop wiring everything).
  Rule: if the function must handle 3+ distinct cases or call 3+ other functions, use "analyst".
"""


def _safe_model_class(value: str, default: ModelClass = ModelClass.WORKER) -> ModelClass:
    try:
        return ModelClass(value.strip().lower())
    except (ValueError, AttributeError):
        _debug_log(f"[model_class] unknown value {value!r}, defaulting to {default.value}")
        return default


def _safe_output_shape(value: str) -> OutputShape:
    try:
        return OutputShape(value.strip().lower())
    except (ValueError, AttributeError):
        return OutputShape.DOCUMENT


def _repair_json_strings(text: str) -> str:
    """Escape literal newlines/tabs inside JSON string values (LLM often emits them)."""
    result = []
    in_string = False
    escaped = False
    for ch in text:
        if escaped:
            result.append(ch)
            escaped = False
        elif ch == "\\" and in_string:
            result.append(ch)
            escaped = True
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\t":
            result.append("\\t")
        elif in_string and ch == "\r":
            result.append("\\r")
        else:
            result.append(ch)
    return "".join(result)


def _parse_orchestrate_raw(raw: str) -> dict | None:
    clean = raw.strip()
    # Strip <think>...</think> blocks (Qwen, DeepSeek reasoning models)
    clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        inner = []
        for line in lines[1:]:
            if line.strip() == "```":
                break
            inner.append(line)
        clean = "\n".join(inner).strip()
    start = clean.find("{")
    end   = clean.rfind("}")
    if start != -1 and end != -1:
        clean = clean[start:end + 1]

    # First attempt: direct parse
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Second attempt: repair literal newlines/tabs inside string values
    try:
        return json.loads(_repair_json_strings(clean))
    except json.JSONDecodeError:
        pass

    # Truncated JSON recovery: the plan was cut mid-string (common when interfaces
    # field is verbose). Try to salvage any complete file_owner objects already parsed.
    try:
        start = clean.find("{")
        if start == -1:
            return None
        fragment = clean[start:]
        # Collect complete "workers" arrays by finding file_owner blocks that closed properly
        owners = []
        for m in re.finditer(r'\{\s*"domain"\s*:\s*"([^"]+)".*?"workers"\s*:\s*(\[(?:[^\[\]]|\[[^\[\]]*\])*\])', fragment, re.DOTALL):
            try:
                workers_raw = json.loads(m.group(2))
                owners.append({"domain": m.group(1), "task": m.group(1), "depends_on": [], "workers": workers_raw})
            except json.JSONDecodeError:
                pass
        # Extract route/shape/interfaces via regex
        route_m = re.search(r'"route"\s*:\s*"(\w+)"', fragment)
        shape_m = re.search(r'"output_shape"\s*:\s*"(\w+)"', fragment)
        iface_m = re.search(r'"interfaces"\s*:\s*"((?:[^"\\]|\\.)*)"', fragment)
        route = route_m.group(1) if route_m else "simple"
        shape = shape_m.group(1) if shape_m else "single_file"
        if iface_m:
            try:
                iface = iface_m.group(1).encode("utf-8").decode("unicode_escape")
            except (UnicodeDecodeError, ValueError):
                iface = iface_m.group(1)  # keep raw on decode failure
        else:
            iface = ""
        if owners and route == "complex":
            _debug_log(f"[orchestrate] truncated JSON recovered: {len(owners)} owners")
            return {"route": route, "output_shape": shape, "interfaces": iface, "file_owners": owners}
    except Exception:
        pass
    return None


async def _orchestrate(task: str) -> dict:
    result = None
    for attempt in range(2):
        raw = await chat(
            [{"role": "user", "content": f"Task: {task}"}],
            system=_CLASSIFY_SYSTEM,
            temperature=0.2,
            max_tokens=8000,
            model_class=ModelClass.ORCHESTRATOR,
        )
        result = _parse_orchestrate_raw(raw)
        if result is not None:
            break
        _debug_log(f"[orchestrate] JSON parse failed (attempt {attempt+1})\nRaw: {raw[:300]}")

    if result is None:
        result = {"route": "simple", "model_class": "worker"}

    # Sanity: if we got file_owners but interfaces is absurdly long, trim it
    # so it doesn't blow worker prompts
    if "interfaces" in result and len(result["interfaces"]) > 400:
        result["interfaces"] = result["interfaces"][:400]
        _debug_log("[orchestrate] interfaces field trimmed to 400 chars")

    # Debug mode: enforce one worker per function — drop duplicates across all file_owners
    if result.get("task_type") == "debug" and result.get("file_owners"):
        seen_fns: set[str] = set()
        for fo in result["file_owners"]:
            deduped = []
            for w in fo.get("workers", []):
                fn_m = re.search(r'def (\w+)\s*\(', w.get("task", ""))
                fn_name = fn_m.group(1) if fn_m else None
                if fn_name and fn_name in seen_fns:
                    _debug_log(f"[orchestrate] debug dedup: dropping duplicate worker for {fn_name}")
                    continue
                if fn_name:
                    seen_fns.add(fn_name)
                deduped.append(w)
            fo["workers"] = deduped

    _debug_log(
        f"[orchestrate] → {result.get('route')} | "
        f"shape={result.get('output_shape','—')} | "
        f"owners={len(result.get('file_owners', []))}"
    )
    return result


# ── Merge system prompts by output shape ──────────────────────────────────────

def _merge_system(shape: OutputShape, is_file_owner: bool = False) -> tuple[str, int]:
    """Return (system_prompt, max_tokens) tuned to the output shape."""
    if shape == OutputShape.SINGLE_FILE:
        if is_file_owner:
            prompt = (
                "You are HiveMind's domain assembler. Each worker wrote exactly one function. "
                "Your job is to stitch them into one clean code block AND validate the result.\n\n"
                "ASSEMBLY RULES:\n"
                "1. Combine ALL worker functions into one fenced code block:\n"
                "   ```python <domain_name>.py\n   <stitched code>\n   ```\n"
                "2. Deduplicate imports — one copy of each at the top.\n"
                "3. Order: imports → constants → functions (dependencies before callers).\n"
                "4. Preserve all logic verbatim — do NOT rewrite, summarize, or paraphrase.\n"
                "5. Fix ONLY: broken for/if/def statements, unclosed brackets, truncated lines.\n"
                "6. If two workers defined the same function name, keep the more complete one.\n\n"
                "VALIDATION — before outputting, check the assembled code for these bugs:\n"
                "A. Fake imports: remove any `from <name> import` where <name> is NOT a stdlib "
                "   or third-party package — these are worker artefacts where the worker invented "
                "   a module name for functions that are already defined in this same file.\n"
                "CRITICAL: Your entire response must be ONE fenced code block and nothing else.\n"
                "No reasoning, no explanation, no steps, no commentary before or after the block.\n"
                "No stub placeholders like `def fn(...) ...` — only complete, runnable function bodies.\n"
                "Start your response with ``` and end it with ```."
            )
        else:
            prompt = (
                "You are HiveMind's assembly agent. Combine the worker outputs into "
                "one complete, working file.\n"
                "RULES:\n"
                "1. Assemble ALL worker code into a single fenced block with the filename:\n"
                "   ```python chess.py\n   <full combined code>\n   ```\n"
                "2. Preserve every function and class VERBATIM — do not rewrite logic.\n"
                "3. Order sections logically (imports → data structures → functions → main).\n"
                "4. Remove duplicate imports or definitions — keep one copy.\n"
                "5. End with a brief '## How to run' section."
            )
        tokens = 6000
    elif shape == OutputShape.MULTI_FILE:
        prompt = (
            "You are HiveMind's assembly agent. Combine the worker outputs into "
            "complete, working files.\n"
            "RULES:\n"
            "1. Output EACH file in its own fenced block with filename on the opening line:\n"
            "   ```python filename.py\n   <code>\n   ```\n"
            "2. Assemble code VERBATIM — do not paraphrase or rewrite logic.\n"
            "3. Fix import conflicts or duplicate definitions, but change nothing else.\n"
            "4. End with a brief '## How to run' section."
        )
        tokens = 5000
    elif shape == OutputShape.DOCUMENT:
        if is_file_owner:
            prompt = (
                "You are HiveMind's section auditor. Review the worker outputs for "
                "your assigned domain and produce a single coherent section.\n"
                "RULES:\n"
                "1. Eliminate redundancy and duplication.\n"
                "2. Preserve all unique information.\n"
                "3. Write in a consistent voice.\n"
                "4. Do not add filler — every sentence must carry information."
            )
        else:
            prompt = (
                "You are HiveMind's document synthesis agent. Combine all sections "
                "into one well-structured final document.\n"
                "RULES:\n"
                "1. Synthesize — do not concatenate.\n"
                "2. Eliminate all redundancy across sections.\n"
                "3. Preserve every unique fact, example, and insight.\n"
                "4. Output clean markdown with consistent heading levels."
            )
        tokens = 2500
    else:  # ANALYSIS
        if is_file_owner:
            prompt = (
                "You are HiveMind's analysis auditor. Consolidate the worker findings "
                "for your domain into key conclusions with supporting evidence.\n"
                "RULES:\n"
                "1. Conclusions first, evidence second.\n"
                "2. Cut all filler — no 'in summary', 'it is worth noting', etc.\n"
                "3. Flag any contradictions between workers."
            )
        else:
            prompt = (
                "You are HiveMind's final analysis agent. Combine all domain findings "
                "into a single structured analysis.\n"
                "RULES:\n"
                "1. Top-level conclusions first.\n"
                "2. Evidence and caveats below each conclusion.\n"
                "3. Cut all filler and duplication.\n"
                "4. If workers contradict, note the disagreement explicitly."
            )
        tokens = 2000

    return prompt, tokens


# ── Manifest helpers (one file per worker, no lock needed) ───────────────────

def _manifest_path(workspace_path: pathlib.Path, task_id: str) -> pathlib.Path:
    """Each worker owns exactly one manifest file — no shared-write contention."""
    return workspace_path.parent / "bin" / f"manifest.{task_id}.json"


def _manifest_read(workspace_path: pathlib.Path) -> dict:
    """Merge all per-worker manifest files in the bin dir into one dict."""
    bin_dir = workspace_path.parent / "bin"
    result: dict = {}
    try:
        for f in bin_dir.glob("manifest.*.json"):
            try:
                result.update(json.loads(f.read_text()))
            except (json.JSONDecodeError, OSError):
                pass
    except OSError:
        pass
    return result


def _manifest_write_entry(
    workspace_path: pathlib.Path,
    task_id: str,
    fn_name: str,
    signature: str,
    domain: str,
) -> None:
    """Write this worker's manifest entry to its own file. Atomic, no contention."""
    if not fn_name or not signature:
        return
    entry = {task_id: {"fn_name": fn_name, "signature": signature,
                        "domain": domain, "written_at": time.time()}}
    path = _manifest_path(workspace_path, task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry))


def _extract_signature(code: str) -> str:
    """Extract the first top-level def line (including return annotation)."""
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            return stripped.rstrip(":").strip()
    return ""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _dedup_top_level_fns(source: str) -> str:
    """Remove earlier duplicate top-level function definitions, keeping the last one."""
    # Split into chunks: each chunk starts at a column-0 `def ` line
    chunks: list[str] = []
    current: list[str] = []
    for line in source.splitlines(keepends=True):
        if line.startswith("def ") and current:
            chunks.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("".join(current))

    # Walk chunks, track function names; on duplicate, drop the earlier chunk
    seen: dict[str, int] = {}   # fn_name → index in chunks list
    for i, chunk in enumerate(chunks):
        m = re.match(r'def (\w+)\s*\(', chunk)
        if m:
            name = m.group(1)
            if name in seen:
                chunks[seen[name]] = ""  # erase earlier definition
                _debug_log(f"[dedup] removed duplicate def {name}")
            seen[name] = i

    return "".join(c for c in chunks if c)


def _extract_fenced_blocks(text: str) -> list[tuple[str, str]]:
    """
    Extract all fenced code blocks from text as (lang_tag, body) pairs.
    Uses a line scanner so that backticks inside string literals don't
    prematurely close the fence — only a line whose *entire* content is
    three or more backticks counts as a closing fence.
    """
    blocks: list[tuple[str, str]] = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        # Opening fence: line starts with ``` followed by optional lang tag
        if line.startswith("```") and len(line) >= 3:
            fence_char_count = len(line) - len(line.lstrip("`"))
            lang_tag = line[fence_char_count:].strip()
            body_lines: list[str] = []
            i += 1
            while i < len(lines):
                inner = lines[i]
                # Closing fence: line is only backtick(s), same count or more
                stripped = inner.strip()
                if stripped and all(c == "`" for c in stripped) and len(stripped) >= fence_char_count:
                    break
                body_lines.append(inner)
                i += 1
            blocks.append((lang_tag, "\n".join(body_lines)))
        i += 1
    return blocks


def _extract_code_for_bin(text: str) -> str:
    """Extract the largest fenced python code block, falling back to raw text."""
    blocks = _extract_fenced_blocks(text)
    best = ""
    # Prefer python-tagged blocks
    for lang_tag, body in blocks:
        lang = lang_tag.split()[0].lower() if lang_tag else ""
        if lang in ("python", "py") and len(body) > len(best):
            best = body
    if best:
        return best.strip()
    # Fall back to any fenced block
    for lang_tag, body in blocks:
        if len(body) > len(best):
            best = body
    if best:
        return best.strip()
    return text.strip()


# ── Agent Node ────────────────────────────────────────────────────────────────

@dataclass
class AgentNode:
    task: str
    depth: int = 0
    parent_id: Optional[str] = None
    child_index: int = 0
    agent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:6])
    task_id: str = ""
    root_task: str = ""

    status: AgentStatus = AgentStatus.PENDING
    children: list["AgentNode"] = field(default_factory=list)
    result: str = ""
    error: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0

    # v2 fields
    role: AgentRole = AgentRole.WORKER
    model_class: ModelClass = ModelClass.WORKER
    output_shape: OutputShape = OutputShape.DOCUMENT
    domain: str = ""                     # file_owner's domain label

    max_depth: int = 4
    on_update: Optional[Callable] = None

    is_project: bool = False
    workspace_path: Optional[pathlib.Path] = None
    output_dir: Optional[pathlib.Path] = None
    sudo_callback: Optional[Callable] = None
    on_shell_run: Optional[Callable] = None
    files_written: list[pathlib.Path] = field(default_factory=list)

    # Transient: worker spec dicts passed from orchestrator to file_owner at construction.
    # Not propagated further — consumed in _file_owner_run.
    _worker_specs: list[dict] = field(default_factory=list)
    _interfaces: str = ""
    _depends_on: list[str] = field(default_factory=list)
    _task_type: str = "generate"   # "generate" | "debug"
    _source_file: str = ""         # full source content for debug tasks
    _contract: dict = field(default_factory=dict)  # structured worker contract from _brief_workers
    _sibling_sigs: dict = field(default_factory=dict)  # {fn_name: signature} from all sibling domains

    def __post_init__(self):
        _agent_registry[self.agent_id] = self
        self.task_id = _make_task_id(self.parent_id, self.child_index)
        if not self.root_task:
            self.root_task = self.task

    def _emit(self):
        if self.on_update:
            self.on_update(self)

    def elapsed(self) -> float:
        if self.started_at == 0:
            return 0.0
        end = self.ended_at if self.ended_at else time.time()
        return end - self.started_at

    def _budget(self) -> int:
        return MAX_TOTAL_AGENTS - len(_agent_registry)

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(self, semaphore: asyncio.Semaphore) -> None:
        self.started_at = time.time()
        self.status = AgentStatus.PLANNING
        self._emit()

        try:
            if self.role == AgentRole.ORCHESTRATOR:
                await self._root_run(semaphore)
            elif self.role == AgentRole.FILE_OWNER:
                await self._file_owner_run(semaphore)
            else:
                await self._worker_run(semaphore)
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.error = str(e)
            self.result = f"[ERROR] {e}"
            _debug_log(f"[{self.task_id}] ERROR: {e}")
            self._emit()
        finally:
            self.ended_at = time.time()

        # Write output files at root — only for project tasks, not Q&A/prose
        if (self.role == AgentRole.ORCHESTRATOR
                and self.is_project
                and self.result
                and not self.result.startswith("[ERROR]")):
            if not self.files_written:
                out = self.output_dir or pathlib.Path.cwd()
                _pm = _TASK_PATH_RE.search(self.root_task)
                _explicit = pathlib.Path(_pm.group(1).strip()).expanduser() if _pm else None
                self.files_written = _extract_and_write_files(
                    self.result, out, self.root_task, explicit_path=_explicit
                )
            for f in self.files_written:
                _debug_log(f"[{self.task_id}] FILE WRITTEN: {f}")

    # ── Root: classify → route ────────────────────────────────────────────────

    async def _root_run(self, semaphore: asyncio.Semaphore):
        async with semaphore:
            plan = await _orchestrate(self.task)

        route = plan.get("route", "simple")

        if route == "ambiguous":
            self.result = plan.get("clarification", "Could you clarify what you mean?")
            self.status = AgentStatus.ERROR
            self.error = "needs_clarification"
            self._emit()
            return

        if route == "simple":
            self.status = AgentStatus.RUNNING
            self._emit()
            mc = _safe_model_class(plan.get("model_class", "worker"))
            answer = await self._solve_direct(model_class=mc)
            answer = await self._execute_runs(answer, model_class=mc)
            self.result = answer
            self.status = AgentStatus.DONE
            self._emit()
            return

        # "complex" — spawn file_owner agents
        file_owners = plan.get("file_owners", [])
        if not file_owners:
            # Planned complex but gave no owners — solve directly
            self.status = AgentStatus.RUNNING
            self._emit()
            self.result = await self._solve_direct(model_class=ModelClass.ORCHESTRATOR)
            self.status = AgentStatus.DONE
            self._emit()
            return

        self.output_shape = _safe_output_shape(plan.get("output_shape", "document"))
        self._interfaces = plan.get("interfaces", "") or ""
        self._task_type = plan.get("task_type", "generate")

        # For debug tasks: load source file from path mentioned in the task
        if self._task_type == "debug":
            _pm = _TASK_PATH_RE.search(self.root_task)
            if _pm:
                _src_path = pathlib.Path(_pm.group(1).strip()).expanduser()
                try:
                    self._source_file = _src_path.read_text()
                    _debug_log(f"[{self.task_id}] debug mode: loaded source {_src_path} ({len(self._source_file)} bytes)")
                except Exception as e:
                    _debug_log(f"[{self.task_id}] debug mode: could not load source: {e}")

        self.is_project = self.output_shape in (OutputShape.SINGLE_FILE, OutputShape.MULTI_FILE)
        if not self.is_project:
            self.is_project = bool(_TASK_PATH_RE.search(self.root_task))

        self.workspace_path = _workspace_path(
            self.root_task, self.output_dir or pathlib.Path.cwd()
        )
        _write_workspace(
            self.workspace_path,
            f"# HiveMind Workspace\n\n**Task:** {self.root_task}\n"
            f"**Shape:** {self.output_shape.value}\n"
            f"**Started:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"**File owners:** {len(file_owners)}\n\n---\n"
        )

        budget = self._budget() - 1
        if len(file_owners) > budget:
            _debug_log(f"[{self.task_id}] budget trim owners {len(file_owners)}→{budget}")
            file_owners = file_owners[:budget]

        self.status = AgentStatus.RUNNING

        # Build a cross-domain signature map: {fn_name: full_def_line}
        # extracted from every worker task string across all file_owners.
        # Each file_owner gets the signatures of ALL other domains' functions
        # so the contract compiler can fill in callee signatures accurately.
        _all_sigs: dict[str, str] = {}
        for fo in file_owners:
            for ws in fo.get("workers", []):
                _sig_m = re.search(
                    r'(def \w+\s*\([^)]*\)\s*(?:->\s*[^\n:]+)?)',
                    ws.get("task", "")
                )
                if _sig_m:
                    _sig = _sig_m.group(1).strip()
                    _fn_m = re.search(r'def (\w+)\s*\(', _sig)
                    if _fn_m:
                        _all_sigs[_fn_m.group(1)] = _sig

        self.children = [
            AgentNode(
                task=fo["task"],
                depth=1,
                parent_id=self.task_id,
                child_index=i,
                root_task=self.root_task,
                role=AgentRole.FILE_OWNER,
                # file_owners plan with ANALYST (downward-light)
                model_class=ModelClass.ANALYST,
                output_shape=self.output_shape,
                domain=fo.get("domain", f"domain_{i+1}"),
                max_depth=self.max_depth,
                on_update=self.on_update,
                is_project=self.is_project,
                workspace_path=self.workspace_path,
                output_dir=self.output_dir,
                sudo_callback=self.sudo_callback,
                on_shell_run=self.on_shell_run,
                # carry worker specs so file_owner can spawn them
                _worker_specs=fo.get("workers", []),
                _interfaces=self._interfaces,
                _depends_on=fo.get("depends_on", []),
                _task_type=self._task_type,
                _source_file=self._source_file,
                _sibling_sigs=_all_sigs,
            )
            for i, fo in enumerate(file_owners)
        ]
        self._emit()
        _debug_log(
            f"[{self.task_id}] spawning {len(self.children)} file_owner agents: "
            + ", ".join(f"{c.task_id}({c.domain})" for c in self.children)
        )

        # File owners run in parallel
        await asyncio.gather(*[child.run(semaphore) for child in self.children])

        # Root final assembly — dependency-aware pure concatenation (no LLM merge)
        self.status = AgentStatus.MERGING
        self._emit()

        def _topo_sort(owners):
            result = []
            remaining = list(owners)
            resolved = set()
            max_iter = len(owners) * len(owners) + 1
            i = 0
            while remaining and i < max_iter:
                i += 1
                for owner in list(remaining):
                    if all(dep in resolved for dep in getattr(owner, "_depends_on", [])):
                        result.append(owner)
                        resolved.add(owner.domain)
                        remaining.remove(owner)
            result.extend(remaining)  # unresolvable deps go last
            return result

        ordered = _topo_sort(self.children)
        sections = []
        for owner in ordered:
            if owner.result and owner.result.strip():
                sections.append(f"# ══ {owner.domain} ══\n{owner.result.strip()}")

        if self._task_type == "debug" and self._source_file:
            # Debug mode: splice fixed functions back into the source file
            final_code = self._splice_debug_fixes(sections)
        else:
            final_code = "\n\n".join(sections)

        # Auto-append entrypoint if a main fn exists but __name__ guard is missing
        if (self._task_type != "debug"
                and "if __name__" not in final_code):
            _main_m = re.search(r'^def (main(?:_\w+)?|run_game|run_app|game_loop|start_game|start_app)\s*\(', final_code, re.MULTILINE)
            if _main_m:
                final_code = final_code.rstrip() + f"\n\nif __name__ == '__main__':\n    {_main_m.group(1)}()\n"
                _debug_log(f"[{self.task_id}] auto-appended __main__ entrypoint for {_main_m.group(1)}")

        self.result = final_code

        # Write assembled code directly to output_dir (no fenced wrapper in v3)
        if self.output_dir and final_code.strip():
            slug = re.sub(r"[^a-z0-9]+", "_", self.root_task.lower())[:48].strip("_")
            out_path = pathlib.Path(self.output_dir) / f"{slug}.py"
            out_path.write_text(final_code)
            _debug_log(f"[{self.task_id}] wrote final output: {out_path} ({len(final_code)} bytes)")

        self.status = AgentStatus.DONE
        self._emit()

    # ── File owner: spawn workers → audit → merge ─────────────────────────────

    async def _file_owner_run(self, semaphore: asyncio.Semaphore):
        self.status = AgentStatus.RUNNING
        self._emit()

        worker_specs = getattr(self, "_worker_specs", [])

        if not worker_specs:
            # No workers planned — solve directly with ANALYST (still downward-light)
            answer = await self._solve_direct(model_class=ModelClass.ANALYST)
            answer = await self._execute_runs(answer, model_class=ModelClass.ANALYST)
            if self.workspace_path:
                _append_workspace(self.workspace_path, f"{self.task_id}({self.domain})", answer)
            self.result = answer
            self.status = AgentStatus.DONE
            self._emit()
            return

        budget = self._budget() - 1
        if len(worker_specs) > budget:
            _debug_log(f"[{self.task_id}] budget trim workers {len(worker_specs)}→{budget}")
            worker_specs = worker_specs[:budget]

        # Compile structured contracts for each worker
        worker_specs = await self._brief_workers(worker_specs, semaphore)

        # Build index: fn_name → worker_spec index within this domain
        domain_fns: dict[str, int] = {}
        for i, ws in enumerate(worker_specs):
            fn = ws.get("contract", {}).get("fn_name", "")
            if fn:
                domain_fns[fn] = i

        # Partition into phase1 (no intra-domain deps) and phase2 (waits for siblings)
        phase1_specs: list[tuple[int, dict]] = []
        phase2_specs: list[tuple[int, dict]] = []
        for i, ws in enumerate(worker_specs):
            calls = ws.get("contract", {}).get("calls", {})
            if any(callee in domain_fns for callee in calls):
                phase2_specs.append((i, ws))
            else:
                phase1_specs.append((i, ws))

        _debug_log(
            f"[{self.task_id}] ({self.domain}) phase1={len(phase1_specs)} phase2={len(phase2_specs)}"
        )

        def _make_worker_node(ws: dict, child_index: int) -> "AgentNode":
            return AgentNode(
                task=ws["task"],
                depth=self.depth + 1,
                parent_id=self.task_id,
                child_index=child_index,
                root_task=self.root_task,
                role=AgentRole.WORKER,
                model_class=_safe_model_class(ws.get("model_class", "worker")),
                output_shape=self.output_shape,
                domain=self.domain,
                max_depth=self.max_depth,
                on_update=self.on_update,
                is_project=self.is_project,
                workspace_path=self.workspace_path,
                output_dir=self.output_dir,
                sudo_callback=self.sudo_callback,
                on_shell_run=self.on_shell_run,
                _interfaces=self._interfaces,
                _task_type=self._task_type,
                _source_file=self._source_file,
                _contract=ws.get("contract", {}),
                _sibling_sigs=self._sibling_sigs,
            )

        # ── Phase 1: fire independent workers immediately ──────────────────────
        phase1_children = [_make_worker_node(ws, i) for i, ws in phase1_specs]
        self.children = list(phase1_children)
        self._emit()
        _debug_log(
            f"[{self.task_id}] ({self.domain}) phase1 workers: "
            + ", ".join(f"{c.task_id}({c.model_class.value})" for c in phase1_children)
        )
        await asyncio.gather(*[child.run(semaphore) for child in phase1_children])

        # ── Phase 2: dependent workers wait for sibling bin files, then read manifest ──
        if phase2_specs:
            bin_dir = pathlib.Path(self.workspace_path).parent / "bin"

            # Map fn_name → task_id of the phase1 worker that owns it
            fn_to_task_id: dict[str, str] = {}
            for child in phase1_children:
                fn = child._contract.get("fn_name", "")
                if fn:
                    fn_to_task_id[fn] = child.task_id

            async def _spawn_phase2_worker(orig_index: int, ws: dict) -> None:
                calls = ws.get("contract", {}).get("calls", {})
                # Wait for each dependency's bin file (max 60s)
                deadline = time.time() + 60.0
                dep_task_ids = [fn_to_task_id[fn] for fn in calls if fn in fn_to_task_id]
                while dep_task_ids and time.time() < deadline:
                    if all((bin_dir / f"{tid}.py").exists() for tid in dep_task_ids):
                        break
                    await asyncio.sleep(0.5)
                else:
                    if dep_task_ids and not all((bin_dir / f"{tid}.py").exists() for tid in dep_task_ids):
                        _debug_log(f"[{self.task_id}] phase2 worker {orig_index} timed out waiting for deps")

                # Read manifest and resolve actual signatures for calls
                manifest = _manifest_read(self.workspace_path)
                real_calls: dict[str, str] = {}
                for callee_fn, planned_sig in calls.items():
                    # Find this callee in the manifest by fn_name within same domain
                    actual_sig = next(
                        (v["signature"] for v in manifest.values()
                         if v.get("fn_name") == callee_fn and v.get("domain") == self.domain),
                        planned_sig,  # fall back to planned if not yet written
                    )
                    real_calls[callee_fn] = actual_sig

                updated_ws = {**ws, "contract": {**ws.get("contract", {}), "calls": real_calls}}
                child = _make_worker_node(updated_ws, orig_index)
                self.children.append(child)
                self._emit()
                _debug_log(
                    f"[{self.task_id}] phase2 spawning {child.task_id} "
                    f"with resolved calls: {real_calls}"
                )
                await child.run(semaphore)

            await asyncio.gather(*[_spawn_phase2_worker(i, ws) for i, ws in phase2_specs])

        # Run auditor (self acts as auditor coordinator)
        self.status = AgentStatus.MERGING
        self._emit()
        await self._auditor_run(semaphore)

        # Collect approved results from bin files
        import ast as _ast_fo
        approved_code = []
        for child in self.children:
            bin_file = pathlib.Path(self.workspace_path).parent / "bin" / f"{child.task_id}.py"
            if bin_file.exists() and bin_file.read_text().strip():
                code = bin_file.read_text().strip()
                try:
                    _ast_fo.parse(code)
                    approved_code.append(code)
                except SyntaxError:
                    _debug_log(f"[{self.task_id}] dropping {child.task_id}: still invalid after audit")

        self.result = "\n\n".join(approved_code)
        if self.workspace_path:
            _append_workspace(self.workspace_path, f"{self.task_id}({self.domain})", self.result)
        self.status = AgentStatus.DONE
        self._emit()

    # ── Worker: solve leaf task ───────────────────────────────────────────────

    async def _worker_run(self, semaphore: asyncio.Semaphore):
        self.status = AgentStatus.RUNNING
        self._emit()
        answer = await self._solve_direct(model_class=self.model_class)
        answer = await self._execute_runs(answer, model_class=self.model_class)
        if self.workspace_path:
            _append_workspace(self.workspace_path, self.task_id, answer)
            # Write extracted code to bin/{task_id}.py
            bin_dir = pathlib.Path(self.workspace_path).parent / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            bin_file = bin_dir / f"{self.task_id}.py"
            code = _extract_code_for_bin(answer)
            bin_file.write_text(code)
            _debug_log(f"[{self.task_id}] wrote bin/{self.task_id}.py ({bin_file.stat().st_size} bytes)")
            # Write manifest entry — actual signature extracted from real output
            _sig = _extract_signature(code)
            _fn = self._contract.get("fn_name", "")
            if _sig and _fn:
                _manifest_write_entry(self.workspace_path, self.task_id, _fn, _sig, self.domain)
                _debug_log(f"[{self.task_id}] manifest: {_fn} → {_sig}")
        self.result = answer
        self.status = AgentStatus.DONE
        self._emit()

    # ── Auditor: syntax + LLM check, up to 3 rounds ─────────────────────────

    async def _auditor_run(self, semaphore: asyncio.Semaphore):
        """Audit all workers under this file_owner. Retries up to MAX_ROUNDS."""
        import ast as _ast_audit
        MAX_ROUNDS = 3
        ws = pathlib.Path(self.workspace_path).parent

        for round_num in range(MAX_ROUNDS):
            failed_workers: list[tuple["AgentNode", str]] = []

            for worker in self.children:
                bin_file = ws / "bin" / f"{worker.task_id}.py"
                if not bin_file.exists() or not bin_file.read_text().strip():
                    failed_workers.append((worker, "empty output"))
                    continue

                code = bin_file.read_text()

                # Step 1: cheap syntax check
                try:
                    _ast_audit.parse(code)
                except SyntaxError as e:
                    failed_workers.append((worker, f"SyntaxError: {e}"))
                    continue
                # passes syntax

            failed_ids = {id(fw) for fw, _ in failed_workers}

            if not failed_workers:
                _debug_log(f"[{self.task_id}] audit round {round_num+1}: all workers passed")
                break

            # Step 2: LLM review of all passing files together
            passing_workers = [w for w in self.children if id(w) not in failed_ids]
            if passing_workers:
                combined = ""
                for w in passing_workers:
                    code = (ws / "bin" / f"{w.task_id}.py").read_text()
                    combined += f"# --- {w.domain} ({w.task_id}) ---\n{code}\n\n"

                llm_issues = await self._audit_llm(combined, semaphore)
                for task_id, issue in llm_issues:
                    worker = next((w for w in self.children if w.task_id == task_id), None)
                    if worker and id(worker) not in failed_ids:
                        failed_workers.append((worker, f"LLM audit: {issue}"))
                        failed_ids.add(id(worker))

            if not failed_workers:
                _debug_log(f"[{self.task_id}] audit round {round_num+1}: LLM pass — all clean")
                break

            if round_num == MAX_ROUNDS - 1:
                # Out of budget — log and skip failed workers
                for worker, reason in failed_workers:
                    _debug_log(
                        f"[{self.task_id}] audit: giving up on {worker.task_id} "
                        f"after {MAX_ROUNDS} rounds: {reason}"
                    )
                break

            # Relaunch failed workers with feedback
            _debug_log(
                f"[{self.task_id}] audit round {round_num+1}: "
                f"{len(failed_workers)} workers need retry"
            )
            for worker, reason in failed_workers:
                worker.status = AgentStatus.PENDING
                bin_file = ws / "bin" / f"{worker.task_id}.py"
                existing_code = bin_file.read_text() if bin_file.exists() else ""
                worker._audit_feedback = (
                    f"Previous attempt failed: {reason}\n"
                    f"Your previous code:\n```python\n{existing_code}\n```\n"
                    "Please fix the issue and rewrite the complete function."
                )

            await asyncio.gather(*[w.run(semaphore) for w, _ in failed_workers])

    async def _brief_workers(
        self, worker_specs: list[dict], semaphore: asyncio.Semaphore
    ) -> list[dict]:
        """
        File owner compiles a structured JSON contract per worker via one LLM call.
        Each contract has: fn_name, signature, what, calls (callee→sig), forbidden.
        Returns worker_specs with a 'contract' key added to each entry.
        Falls back to minimal contracts parsed from task text if LLM call fails.
        """
        iface_note = f"\nInterface contract: {self._interfaces}" if self._interfaces else ""

        # Build a sibling signature note — all known fn signatures from other domains
        sibling_note = ""
        if self._sibling_sigs:
            sibling_lines = "\n".join(
                f"  {sig}  {'[returns None — mutates in-place]' if '-> None' in sig or sig.rstrip().endswith('None') else '[returns a value]'}"
                for sig in self._sibling_sigs.values()
            )
            sibling_note = f"\n\nAll functions available in scope (from other domains):\n{sibling_lines}"

        worker_list = "\n".join(
            f"{i+1}. {ws['task']}" for i, ws in enumerate(worker_specs)
        )
        system = (
            "You are a contract compiler for a multi-agent code system.\n"
            "Output a JSON array — one object per function — and NOTHING ELSE. No prose, no fences.\n\n"
            "Each object must have exactly these keys:\n"
            "  fn_name    : bare function name (no parens)\n"
            "  signature  : full def line, single line, exact params and return type\n"
            "  what       : 2-3 sentences describing what the function must do. "
            "If the function calls others, state exactly HOW each callee is called "
            "(e.g. 'call add_task(tasks, desc) as a statement — it mutates tasks in-place and returns None').\n"
            "  calls      : object mapping each callee fn_name to its EXACT def signature — "
            "look up the signature in the 'All functions available in scope' list above. "
            "If a callee is not in that list, infer it from the task hint. NEVER leave a signature empty.\n"
            "  forbidden  : list of strings — things this worker must NOT do\n\n"
            "CRITICAL RULES FOR forbidden — you MUST include these for every function:\n"
            "  • For every callee in 'calls' whose signature ends in '-> None': add "
            "\"assign result of <fn_name> — it returns None, call as bare statement only, e.g. fn(x) not x = fn(x)\".\n"
            "  • For every callee in 'calls' that takes a mutable container (list, dict) as first arg "
            "and returns None: add \"reassign the container after calling <fn_name>\".\n"
            "  • Always add \"reimplement any function listed in calls\".\n\n"
            "The array must have exactly as many entries as input functions, same order."
        )
        user = (
            f"Domain: {self.domain}\n"
            f"Domain task: {self.task}{iface_note}{sibling_note}\n\n"
            f"Functions ({len(worker_specs)}):\n{worker_list}"
        )

        def _minimal_contract(ws: dict, index: int) -> dict:
            """Fallback: build a minimal contract from the task text."""
            task = ws["task"]
            sig_m = re.search(r'(def \w+\((?:[^()]*|\[[^\[\]]*\])*\)\s*(?:->\s*[^\n:]+)?)', task)
            sig = sig_m.group(1).strip() if sig_m else ""
            fn_m = re.search(r'def (\w+)\s*\(', sig)
            fn_name = fn_m.group(1) if fn_m else f"fn_{index+1}"
            calls_m = re.search(r'[Cc]alls?:\s*\[([^\]]+)\]', task)
            calls_list = [c.strip() for c in calls_m.group(1).split(",")] if calls_m else []
            return {
                "fn_name": fn_name,
                "signature": sig,
                "what": task,
                "calls": {fn: "" for fn in calls_list},
                "forbidden": [],
            }

        contracts: list[dict] = []
        try:
            async with semaphore:
                raw = await chat(
                    [{"role": "user", "content": user}],
                    system=system,
                    temperature=0.2,
                    max_tokens=3000,
                    model_class=ModelClass.ANALYST,
                )
            # Strip think blocks and fences
            clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(l for l in lines[1:] if l.strip() != "```")
            parsed = json.loads(clean)
            if isinstance(parsed, list) and len(parsed) == len(worker_specs):
                contracts = parsed
                _debug_log(f"[{self.task_id}] compiled {len(contracts)} contracts for '{self.domain}'")
            else:
                raise ValueError(f"expected {len(worker_specs)} contracts, got {len(parsed) if isinstance(parsed, list) else type(parsed)}")
        except Exception as e:
            _debug_log(f"[{self.task_id}] _brief_workers contract parse failed ({e}), using minimal contracts")
            contracts = [_minimal_contract(ws, i) for i, ws in enumerate(worker_specs)]

        return [{**ws, "contract": contracts[i]} for i, ws in enumerate(worker_specs)]

    async def _audit_llm(
        self, combined_code: str, semaphore: asyncio.Semaphore
    ) -> list[tuple[str, str]]:
        """Ask LLM to review combined code. Returns list of (task_id, issue)."""
        iface_note = f"\nInterface contract: {self._interfaces}" if self._interfaces else ""
        system = (
            "You are a code auditor. Review these functions for: "
            "(1) signature mismatches with the interface contract, "
            "(2) obvious logic bugs, "
            "(3) stub placeholders (... or pass-only bodies). "
            "For each problem found, output ONLY lines in format: FAIL task_id: reason. "
            "If all functions are correct output: PASS"
            + iface_note
        )
        async with semaphore:
            raw = await chat(
                [{"role": "user", "content": combined_code}],
                system=system,
                temperature=0.2,
                max_tokens=1024,
                model_class=ModelClass.ANALYST,
            )
        issues: list[tuple[str, str]] = []
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("FAIL "):
                rest = line[5:]  # strip "FAIL "
                if ": " in rest:
                    task_id, reason = rest.split(": ", 1)
                    issues.append((task_id.strip(), reason.strip()))
        _debug_log(f"[{self.task_id}] _audit_llm: {len(issues)} issues found")
        return issues

    # ── Solve helpers ─────────────────────────────────────────────────────────

    async def _solve_direct(self, model_class: ModelClass) -> str:
        ws_context = ""
        if self.workspace_path:
            existing = _read_workspace(self.workspace_path)
            if existing:
                ws_context = f"\n\nWorkspace context:\n{existing[-1500:]}"

        _explicit = None
        _path_m = _TASK_PATH_RE.search(self.task)
        if _path_m:
            _explicit = pathlib.Path(_path_m.group(1).strip()).expanduser()

        if self.is_project and self.output_shape in (OutputShape.SINGLE_FILE, OutputShape.MULTI_FILE):
            if _explicit:
                file_hint = (
                    f"\n\nWrite your complete output as a single fenced block. "
                    f"Put the filename `{_explicit.name}` on the opening line, e.g.:\n"
                    f"```python {_explicit.name}\n<code>\n```\n"
                    f"Do not add commentary outside the fenced block."
                )
            elif self.domain:
                file_hint = (
                    f"\n\nYou are writing content for: {self.domain}\n"
                    "Wrap your output in a fenced block with the filename on the opening line:\n"
                    f"```python {self.domain}\n<code>\n```"
                )
            else:
                file_hint = (
                    "\n\nWrap each file in a fenced block with its filename on the opening line:\n"
                    "```python snake.py\n<code>\n```"
                )
        else:
            file_hint = ""

        domain_context = f"\nYou are working on domain: {self.domain}." if self.domain else ""

        # Workers on single-file code tasks write one focused piece;
        # the file_owner analyst stitches all workers' output into the final file.
        if (self.role == AgentRole.WORKER
                and self.output_shape in (OutputShape.SINGLE_FILE, OutputShape.MULTI_FILE)
                and self.is_project):

            import re as _re

            if self._task_type == "debug" and self._source_file:
                # ── DEBUG WORKER: fix one specific bug in the source file ─────
                sig_m = _re.search(r'(def \w+\([^)]*\)\s*(?:->\s*[^\n:]+)?)', self.task)
                sig_line = sig_m.group(1).strip() if sig_m else ""

                system = "\n".join(filter(None, [
                    "You are a single-function bug-fix worker. These constraints are ABSOLUTE:",
                    f"SIGNATURE: {sig_line}" if sig_line else "",
                    "• Output ONLY the corrected function — exact same name and signature as above.",
                    "• Do NOT output the entire file. One function only.",
                    "• Wrap output in ONE fenced block: ```python\n<fixed function>\n```",
                ]))

                worker_scope = (
                    "YOU ARE A BUG-FIX WORKER. You will receive the full source file for context "
                    "and a specific bug to fix. Output ONLY the corrected function — nothing else.\n\n"
                    "RULES:\n"
                    "• Fix the exact bug described — do not change anything else\n"
                    "• Keep the exact same function name and signature\n"
                    "• Write ONLY that one function — no other functions, no main block\n"
                    "• Wrap output in ONE fenced block: ```python\n<fixed function>\n```"
                )

            else:
                # ── GENERATE WORKER ───────────────────────────────────────────
                if self._contract:
                    # Contract JSON from _brief_workers — the primary path
                    # calls dict may have been resolved with real manifest signatures by phase2
                    contract_json = json.dumps(self._contract, indent=2)
                    iface_line = f"\nINTERFACE CONTRACT:\n{self._interfaces}" if self._interfaces else ""

                    # Build a callee cheat-sheet: every function in calls, annotated with call semantics
                    calls = self._contract.get("calls", {})
                    callee_lines = []
                    for callee_fn, callee_sig in calls.items():
                        if not callee_sig:
                            # fall back to sibling_sigs if contract left it empty
                            callee_sig = self._sibling_sigs.get(callee_fn, "")
                        if callee_sig:
                            returns_none = callee_sig.rstrip().endswith("None") or "-> None" in callee_sig
                            note = "CALL AS STATEMENT — returns None, mutates in-place" if returns_none else "CALL AND USE RETURN VALUE"
                            callee_lines.append(f"  {callee_sig}  # {note}")
                    callee_ref = ("\n\nCALLEE REFERENCE — these functions are already in scope:\n"
                                  + "\n".join(callee_lines)) if callee_lines else ""

                    system = (
                        "You are a single-function code worker. "
                        "Your contract is below as JSON. Every field is ABSOLUTE — no deviations.\n\n"
                        f"{contract_json}"
                        f"{callee_ref}"
                        f"{iface_line}\n\n"
                        "RULES:\n"
                        "• Implement exactly fn_name with the exact signature shown.\n"
                        "• Every function in 'calls' already exists in scope — call it by that exact name "
                        "with the exact parameters shown. DO NOT reimplement it.\n"
                        "• CRITICAL: If a callee is annotated '# CALL AS STATEMENT', call it as a bare "
                        "statement — NEVER assign its result. E.g. `add_task(tasks, desc)` not `tasks = add_task(tasks, desc)`.\n"
                        "• Do not violate any item in 'forbidden'.\n"
                        "• CRITICAL: If your function needs to match triple backticks (e.g. markdown fence detection), "
                        "NEVER write raw ``` inside a string literal — use '`' * 3 or chr(96) * 3 instead. "
                        "Raw triple backticks truncate the output fence and corrupt your response.\n"
                        "• Output ONE fenced python block: ```python\n<your function>\n```"
                    )
                else:
                    # Fallback: no contract (simple tasks, non-project workers)
                    sig_m = _re.search(r'(def \w+\([^)]*\)\s*(?:->\s*[^\n:]+)?)', self.task)
                    sig_line = sig_m.group(1).strip() if sig_m else ""
                    contract_lines = [
                        "You are a single-function code worker. These constraints are ABSOLUTE:",
                        f"SIGNATURE: {sig_line}" if sig_line else "",
                        "• Function name, parameter names, parameter count, return type — EXACT match to SIGNATURE.",
                    ]
                    if self._interfaces:
                        contract_lines += ["CONTRACT:", self._interfaces]
                    system = "\n".join(l for l in contract_lines if l)

                worker_scope = (
                    "YOU ARE A SINGLE-FUNCTION WORKER. Implement exactly the one function in your contract.\n\n"
                    "RULES:\n"
                    "• Write ONLY that function — def line, body, stdlib imports only\n"
                    "• Complete implementation — no pass, no ..., no placeholders\n"
                    "• Do NOT write a second function, class, main block, or example usage\n"
                    "• Do NOT reimplement functions listed in 'calls' — call them by exact name\n"
                    "• Nested helpers go INSIDE your function as nested defs\n"
                    "• Do NOT import project functions — they are already in scope\n"
                    "• CRITICAL: If a callee's signature ends in '-> None', it mutates in-place and "
                    "returns nothing — call it as a statement, NEVER assign its result (e.g. 'fn(x)' not 'x = fn(x)')\n"
                    "• CRITICAL: If your function needs to check for triple backticks (e.g. markdown fences), "
                    "NEVER write raw ``` in a string literal — use '`' * 3 or chr(96) * 3 instead. "
                    "Raw triple backticks inside your code will break the output fence and truncate your response.\n"
                    "• Wrap output in ONE fenced block: ```python\n<your function>\n```"
                )
        else:
            system = (
                "You are a HiveMind agent. Solve the task completely and directly. "
                "Do not ask clarifying questions — do your best with the information given."
                + domain_context
                + file_hint
            )
            worker_scope = ""

        # Scale token budget by role + model class
        if self.role == AgentRole.WORKER:
            _max_tok = 8000 if self.model_class == ModelClass.ANALYST else 6000
        else:
            _max_tok = 2048

        if self.role == AgentRole.WORKER and self.is_project:
            if self._task_type == "debug" and self._source_file:
                # Extract just the target function from source to stay within token limits.
                # The function name comes from the first `def fn_name` in the task description.
                import re as _re2
                _fn_m = _re2.search(r'def (\w+)\s*\(', self.task)
                _fn_name = _fn_m.group(1) if _fn_m else None
                if _fn_name:
                    _fn_pattern = re.compile(
                        rf'^(def {re.escape(_fn_name)}\s*\(.*?)(?=\n^(?:def |class )|\Z)',
                        re.DOTALL | re.MULTILINE,
                    )
                    _fn_m2 = _fn_pattern.search(self._source_file)
                    _fn_context = f"```python\n{_fn_m2.group(0).strip()}\n```" if _fn_m2 else "(function not found in source)"
                else:
                    _fn_context = "(could not identify target function)"
                user_prompt = (
                    f"{worker_scope}\n\n"
                    f"Current (broken) function:\n{_fn_context}\n\n"
                    f"Bug to fix: {self.task}{ws_context}"
                )
            else:
                user_prompt = f"{worker_scope}\n\nTask: {self.task}{ws_context}"
        else:
            user_prompt = f"Task: {self.task}{ws_context}"

        audit_fb = getattr(self, "_audit_feedback", "")
        if audit_fb:
            user_prompt += f"\n\n{audit_fb}"

        return await chat(
            [{"role": "user", "content": user_prompt}],
            system=system,
            temperature=0.4,
            max_tokens=_max_tok,
            model_class=model_class,
        )

    # ── Shell execution ───────────────────────────────────────────────────────

    async def _execute_runs(self, answer: str,
                             model_class: Optional[ModelClass] = None) -> str:
        mc = model_class or self.model_class
        messages: list[dict] = []
        current = answer
        MAX_ITERS = 12

        for _ in range(MAX_ITERS):
            runs = _RUN_MARKER.findall(current)
            if not runs:
                break

            run_results: list[str] = []
            for cmd in runs:
                cmd = cmd.strip()
                if not cmd:
                    continue
                _debug_log(f"[{self.task_id}] SHELL: {cmd}")
                output = await _exec_command(cmd, self.output_dir, self.sudo_callback)
                run_results.append(f"$ {cmd}\n{output}")
                if self.on_shell_run:
                    await self.on_shell_run(cmd, output)

            if not run_results:
                break

            clean = _RUN_MARKER.sub("", current).strip()
            run_block = "\n\n".join(run_results)

            messages.append({"role": "user", "content": (
                f"Shell output:\n{run_block}\n\n"
                f"Your answer so far:\n{clean}\n\n"
                "Continue. You may run more ##RUN## commands or write your final answer."
            )})

            current = await chat(
                messages,
                system=(
                    "You are a HiveMind agent with shell access via ##RUN## blocks. "
                    "When you have enough information, write your final answer without ##RUN##."
                ),
                temperature=0.3,
                max_tokens=2048,
                model_class=mc,
            )
            messages.append({"role": "assistant", "content": current})

        return current

    # ── Merge (shape-aware, role-aware) ───────────────────────────────────────

    async def _merge(self, model_class: ModelClass, is_file_owner: bool = False) -> str:
        # single_file root merge: extract code from each file_owner result and
        # concatenate in Python — no LLM call for assembly avoids 6K token / connection limits.
        if (self.output_shape == OutputShape.SINGLE_FILE
                and not is_file_owner
                and self.depth == 0):
            return self._concatenate_single_file()

        # Root orchestrator merges hit Groq compound (6K body limit) — keep tight.
        # File_owner ANALYST merges go to llama-4-scout/nemotron (131K+ context) — give full content.
        if is_file_owner:
            MAX_CHILD_CHARS = 8000
        elif self.output_shape in (OutputShape.SINGLE_FILE, OutputShape.MULTI_FILE):
            MAX_CHILD_CHARS = 1500
        else:
            MAX_CHILD_CHARS = 800
        child_results = "\n\n".join(
            f"### [{c.task_id}] {c.task}\n{c.result[:MAX_CHILD_CHARS]}"
            + ("…[truncated]" if len(c.result) > MAX_CHILD_CHARS else "")
            for c in self.children
        )

        ws_context = ""
        if self.workspace_path:
            existing = _read_workspace(self.workspace_path)
            if existing:
                ws_context = f"\n\nWorkspace:\n{existing[-1500:]}"

        system, max_tokens = _merge_system(self.output_shape, is_file_owner=is_file_owner)
        domain_note = f"\nYou are merging output for domain: {self.domain}." if self.domain else ""
        iface_note = f"\nShared interface contract: {self._interfaces}" if self._interfaces and is_file_owner else ""

        return await chat(
            [{"role": "user", "content": (
                f"Original task: {self.task}\n"
                + domain_note
                + iface_note
                + f"\n\nSub-agent outputs:\n{child_results}{ws_context}\n\n"
                "Produce a unified, complete output."
            )}],
            system=system,
            temperature=0.3,
            max_tokens=max_tokens,
            model_class=model_class,
        )

    def _concatenate_single_file(self) -> str:
        """
        Pure-Python assembly for single_file tasks.
        Extracts code from each file_owner's result (strips fences, deduplicates
        imports) and concatenates into one fenced block. No LLM call needed.
        """
        import_lines: list[str] = []
        body_sections: list[str] = []

        # Shell-like langs to skip when extracting code bodies
        _SKIP_LANGS = {"bash", "sh", "shell", "zsh", "console", "text", ""}

        for child in self.children:
            raw = child.result or ""
            # If result contains a fenced block, discard everything outside the fences.
            # This strips analyst prose preambles and reasoning chains reliably.
            _first_fence = raw.find("```")
            _last_fence  = raw.rfind("```")
            if _first_fence != -1 and _last_fence != _first_fence:
                raw = raw[_first_fence:_last_fence + 3]
            # Extract fenced code blocks — skip shell/prose blocks, take largest code block
            fenced_blocks = _extract_fenced_blocks(raw)
            code_blocks = [
                body.strip()
                for lang_tag, body in fenced_blocks
                if (lang_tag.split()[0].lower() if lang_tag else "") not in _SKIP_LANGS and body.strip()
            ]
            if code_blocks:
                # Join all code blocks; if one block is ≥ 80% of total, use it alone
                # to avoid duplicating import-only helper blocks from the "How to run" section
                total = sum(len(b) for b in code_blocks)
                dominant = [b for b in code_blocks if len(b) >= 0.8 * total]
                code = dominant[0] if dominant else "\n\n".join(code_blocks)
            else:
                # No fenced code block at all — analyst output pure prose (reasoning essay,
                # no code). Try to extract def blocks directly from raw text as last resort.
                def_blocks = re.findall(r'^(def \w+.*?)(?=\ndef |\Z)', raw, re.DOTALL | re.MULTILINE)
                if def_blocks:
                    code = "\n\n".join(b.strip() for b in def_blocks)
                else:
                    # Pure prose, no recoverable code — skip this domain entirely
                    _debug_log(f"[{self.task_id}] skipping {child.domain}: no code block found in analyst output")
                    continue

            if not code:
                continue

            # Strip stub placeholder lines like `def fn(...) ...` or `def fn(...):\n    ...`
            # These are analyst summaries, not real code
            code = re.sub(r'^def [^\n]+\.\.\.[^\n]*\n?', '', code, flags=re.MULTILINE)
            # Two-line stubs: `def fn(...):\n    ...\n`
            code = re.sub(r'^(def [^\n]+:\n)\s*\.\.\.\s*\n?', '', code, flags=re.MULTILINE)

            # Strip trailing lines that look truncated (open bracket, trailing comma/operator)
            code_lines = code.rstrip().splitlines()
            # Strip trailing lines that look truncated: open bracket, unclosed string,
            # trailing comma/operator, or imbalanced quotes.
            while code_lines:
                last = code_lines[-1].rstrip()
                # Open bracket / operator at end of line
                if last.endswith(("(", "[", "{", ",", "\\", "+", "-", "*", "/", "%",
                                   "=", "==", "!=", "<=", ">=", "<", ">",
                                   "and", "or", "not", "in", "is", ":", "->")):
                    code_lines.pop()
                    continue
                # Unclosed string literal: odd number of unescaped quotes
                sq = last.count("'") - last.count("\\'")
                dq = last.count('"') - last.count('\\"')
                if sq % 2 != 0 or dq % 2 != 0:
                    code_lines.pop()
                    continue
                break
            code = "\n".join(code_lines)

            # Split imports from body
            lines = code.splitlines()
            imp, body = [], []
            in_body = False
            for line in lines:
                stripped = line.strip()
                if not in_body and (
                    stripped.startswith("import ")
                    or stripped.startswith("from ")
                    or stripped == ""
                ):
                    imp.append(line)
                else:
                    in_body = True
                    body.append(line)

            # Validate: skip sections with unclosed triple-quoted strings
            try:
                import ast as _ast
                _ast.parse(code)
                _code_valid = True
            except SyntaxError:
                # Try to parse just enough to detect unclosed docstrings
                _tq_dq = code.count('"""')
                _tq_sq = code.count("'''")
                _code_valid = (_tq_dq % 2 == 0 and _tq_sq % 2 == 0)
                if not _code_valid:
                    _debug_log(f"[{self.task_id}] skipping {child.domain}: unclosed triple-quote string")

            if _code_valid:
                import_lines.extend(imp)
            if body and _code_valid:
                body_sections.append(f"# ── {child.domain} ──\n" + "\n".join(body))

        # Strip fake internal imports: `from <unknown_module> import` where the module
        # is not a known stdlib/third-party package — these are worker artefacts where
        # the worker invented a module name for functions that live in the same file.
        _KNOWN_PKGS = {
            "os", "sys", "re", "io", "math", "json", "copy", "time", "random",
            "string", "typing", "collections", "itertools", "functools", "pathlib",
            "dataclasses", "enum", "abc", "ast", "textwrap", "struct", "hashlib",
            "datetime", "threading", "asyncio", "subprocess", "shutil", "tempfile",
            "unittest", "logging", "warnings", "contextlib", "inspect", "types",
            "heapq", "bisect", "queue", "array", "weakref", "gc", "platform",
            "numpy", "np", "pandas", "pd", "pygame", "flask", "django", "requests",
            "PIL", "cv2", "sklearn", "torch", "tensorflow", "scipy", "matplotlib",
        }
        _fake_import_re = re.compile(r'^from\s+(\w+)\s+import', re.MULTILINE)

        def _is_real_import(line: str) -> bool:
            m = _fake_import_re.match(line.strip())
            if not m:
                return True  # plain `import x` or blank — keep
            mod = m.group(1).split(".")[0]
            return mod in _KNOWN_PKGS

        # Deduplicate imports while preserving order
        seen_imports: set[str] = set()
        deduped_imports: list[str] = []
        for line in import_lines:
            key = line.strip()
            if key and key not in seen_imports and _is_real_import(line):
                seen_imports.add(key)
                deduped_imports.append(line)

        # Infer filename and language from task text and dominant lang in blocks
        slug = re.sub(r"[^a-z0-9]+", "_", self.root_task.lower())[:24].strip("_")
        task_lower = self.root_task.lower()
        if any(w in task_lower for w in ("webpage", "html", "website", "web page")):
            ext, lang = "html", "html"
        elif any(w in task_lower for w in ("javascript", " js ")):
            ext, lang = "js", "javascript"
        else:
            ext, lang = "py", "python"
        fname = f"{slug}.{ext}"

        if ext == "py":
            assembled = "\n".join(deduped_imports) + "\n\n" + "\n\n".join(body_sections)
        else:
            # For non-Python, concatenate all sections without import dedup
            assembled = "\n\n".join(body_sections)

        _debug_log(f"[{self.task_id}] concatenation ({lang}): {len(assembled)} chars → {fname}")
        return f"```{lang} {fname}\n{assembled}\n```"

    # ── Debug splice ──────────────────────────────────────────────────────────

    def _splice_debug_fixes(self, sections: list[str]) -> str:
        """
        Replace functions in the source file with fixed versions from workers.
        Each section is a worker's output containing one corrected function.
        Functions are matched by name and replaced in-place.
        """
        import ast as _ast

        result = self._source_file

        for section in sections:
            # Extract the fixed function code from the section
            code = _extract_code_for_bin(section)
            if not code.strip():
                continue

            # Find the function name in the fixed code
            fn_m = re.search(r'^def (\w+)\s*\(', code, re.MULTILINE)
            if not fn_m:
                _debug_log(f"[{self.task_id}] splice: no function found in section, skipping")
                continue
            fn_name = fn_m.group(1)

            # Find the function in the source file and replace it
            # Match: def fn_name(...): ... up to the next top-level def/class or EOF
            pattern = re.compile(
                rf'^(def {re.escape(fn_name)}\s*\(.*?)(?=\n^(?:def |class )|\Z)',
                re.DOTALL | re.MULTILINE,
            )
            if pattern.search(result):
                result = pattern.sub(code.rstrip(), result, count=1)
                _debug_log(f"[{self.task_id}] splice: replaced {fn_name}")
            else:
                # Function not found in source — append it
                result = result.rstrip() + "\n\n" + code.strip() + "\n"
                _debug_log(f"[{self.task_id}] splice: {fn_name} not found in source, appended")

        # Deduplicate top-level functions: if the same name appears twice (old + new),
        # keep the last definition. Split on `\ndef ` boundaries at column 0.
        result = _dedup_top_level_fns(result)
        return result

    # ── Tree helpers ──────────────────────────────────────────────────────────

    def all_nodes(self) -> list["AgentNode"]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.all_nodes())
        return nodes
