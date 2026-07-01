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
        for lang_tag, code in _FENCED_LANG.findall(result):
            # lang_tag may be "python" or "python chess.py" — extract just the language word
            lang = lang_tag.split()[0].lower()
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
    "output_shape": "<single_file|multi_file|document|analysis>",
    "file_owners": [
      {
        "domain": "short_label",
        "task": "What this domain covers.",
        "workers": [
          {"task": "def fn_name(params) -> type: one sentence. Calls: [other_fn if needed].", "model_class": "worker"}
        ]
      }
    ],
    "interfaces": "KEEP UNDER 200 CHARS. Key shared types only, no newlines. E.g.: 'board:list[list[str|None]] 8x8 board[rank][file] rank0=top, piece=2char type+color e.g.k+/p- piece[0]=type piece[1]=color'. Omit for prose."
  }
  IMPORTANT: output_shape and file_owners MUST appear before interfaces in the JSON.
  The interfaces field is lowest priority — complete the plan first, add interfaces last.

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
  • Function names must be unique across ALL workers in the entire plan — no duplicates.
  • All workers must use the exact types from the interfaces contract.
  • Tell workers what functions from other workers they may call, e.g.:
    "Call get_piece_moves(board, file, rank, en_passant) — do NOT reimplement it."
  • One worker per file_owner must implement the entry point (main/run/game_loop) that calls the others by their exact function names.
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
                owners.append({"domain": m.group(1), "task": m.group(1), "workers": workers_raw})
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

        # Write output files at root
        if self.role == AgentRole.ORCHESTRATOR and self.result and not self.result.startswith("[ERROR]"):
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

        # Root final audit — upward-heavy: ORCHESTRATOR
        self.status = AgentStatus.MERGING
        self._emit()
        self.result = await self._merge(
            model_class=ModelClass.ORCHESTRATOR,
            is_file_owner=False,
        )
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

        self.children = [
            AgentNode(
                task=ws["task"],
                depth=self.depth + 1,
                parent_id=self.task_id,
                child_index=i,
                root_task=self.root_task,
                role=AgentRole.WORKER,
                # workers use WORKER class (downward-light)
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
            )
            for i, ws in enumerate(worker_specs)
        ]
        self._emit()
        _debug_log(
            f"[{self.task_id}] ({self.domain}) spawning {len(self.children)} workers: "
            + ", ".join(f"{c.task_id}({c.model_class.value})" for c in self.children)
        )

        # Workers run in parallel
        await asyncio.gather(*[child.run(semaphore) for child in self.children])

        # File-owner audit — upward-heavy: ANALYST audits its own workers
        self.status = AgentStatus.MERGING
        self._emit()
        self.result = await self._merge(
            model_class=ModelClass.ANALYST,
            is_file_owner=True,
        )
        # Sanity-check analyst result quality; fall back to raw worker concat if broken.
        _unk_count = self.result.count("<unk>")
        _unk_ratio = _unk_count / max(len(self.result), 1)
        _tq_dq = self.result.count('"""')
        _tq_sq = self.result.count("'''")
        _analyst_bad = (_unk_ratio > 0.01 or _unk_count > 5
                        or _tq_dq % 2 != 0 or _tq_sq % 2 != 0)
        if _analyst_bad:
            _debug_log(f"[{self.task_id}] analyst result invalid (unk={_unk_ratio:.3f}, tq={_tq_count}), using raw worker concat")
            self.result = "\n\n".join(c.result for c in self.children if c.result)
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
        self.result = answer
        self.status = AgentStatus.DONE
        self._emit()

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
            if self._interfaces:
                iface_note = (
                    "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "ENFORCED CONTRACT — your code MUST satisfy these:\n"
                    f"{self._interfaces}\n"
                    "Any variable, type, or value that contradicts the above is a bug.\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )
            else:
                iface_note = ""
            worker_scope = (
                "\n\nYOU ARE A SINGLE-FUNCTION WORKER. Your only job is to implement the ONE "
                "function whose exact signature is stated in the task. Nothing else.\n\n"
                "STRICT RULES:\n"
                "• Write ONLY that one function — its def line, body, and any stdlib imports it needs\n"
                "• Use the EXACT function name, parameter names, and return type from the task signature\n"
                "• The implementation must be complete — no `pass`, no `...`, no placeholders\n"
                "• Do NOT write a second function, a class, a main block, or example usage\n"
                "• Do NOT reimplement functions other workers own — call them by their exact name\n"
                "• If your function needs a small helper, define it as a nested def INSIDE your function\n"
                "• Do NOT add `from <module> import` for things defined in this same project — "
                "all functions are in one file and are already in scope\n"
                "• Wrap your output in ONE fenced block: ```python\\n<your function>\\n```"
                + iface_note
            )
        else:
            worker_scope = ""

        system = (
            "You are a HiveMind agent. Solve the task completely and directly. "
            "Do not ask clarifying questions — do your best with the information given."
            + domain_context
            + worker_scope
            + file_hint
        )
        # Scale token budget by role + model class
        # analyst-class workers handle complex functions and need more room
        if self.role == AgentRole.WORKER:
            _max_tok = 8000 if self.model_class == ModelClass.ANALYST else 6000
        else:
            _max_tok = 2048
        return await chat(
            [{"role": "user", "content": f"Task: {self.task}{ws_context}"}],
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
            fenced_blocks = _FENCED_LANG.findall(raw)
            code_blocks = [
                body.strip()
                for lang_tag, body in fenced_blocks
                if lang_tag.split()[0].lower() not in _SKIP_LANGS and body.strip()
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

    # ── Tree helpers ──────────────────────────────────────────────────────────

    def all_nodes(self) -> list["AgentNode"]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.all_nodes())
        return nodes
