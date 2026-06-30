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
    r"```(?:[\w.+-]*\s+)?(?:#\s*)?(?:file:\s*)?([^\n`]+\.[a-zA-Z0-9]+)\n(.*?)```",
    re.DOTALL,
)
_HEADING_FENCED = re.compile(
    r"#{1,4}\s+([^\n`]+\.[a-zA-Z0-9]+)\s*\n```[\w]*\n(.*?)```",
    re.DOTALL,
)
_FENCED_LANG = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)
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
        for lang, code in _FENCED_LANG.findall(result):
            ext = _LANG_EXT.get(lang.lower())
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
  "simple"     — Can be answered in one pass by a single agent.
                 Examples: Q&A, explanations, short writing, single-file scripts,
                 conversation, guides, summaries.
  "ambiguous"  — The task is unclear, missing critical info, or could mean very
                 different things. Ask ONE clarifying question.
  "complex"    — Requires multiple distinct output files or independent domain
                 areas that genuinely benefit from parallel specialist agents.
                 Examples: multi-file codebases, large multi-section documents,
                 research tasks spanning multiple independent topics.

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
        "domain": "Short label for this owner's area (e.g. 'game_logic.py')",
        "task": "What this file-owner agent is responsible for producing.",
        "workers": [
          {
            "task": "Specific function/section this worker writes.",
            "model_class": "worker"
          }
        ]
      }
    ]
  }

output_shape guide:
  single_file  — all output goes into one file
  multi_file   — output is multiple distinct named files
  document     — prose output (report, README, guide, essay)
  analysis     — structured analysis/conclusions (no filler prose)

Rules for "complex":
  • Minimum 2 file_owners, maximum 6.
  • Each file_owner has 1-4 workers. Workers write specific functions/sections.
  • Keep workers focused and independent — no cross-dependencies within a file_owner.
  • file_owners are run in parallel; workers within each file_owner run in parallel.
  • Only use "complex" when parallel work genuinely speeds up a non-trivial task.
    A single Python script, a short essay, or a step-by-step guide is SIMPLE.
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


async def _orchestrate(task: str) -> dict:
    raw = await chat(
        [{"role": "user", "content": f"Task: {task}"}],
        system=_CLASSIFY_SYSTEM,
        temperature=0.2,
        max_tokens=1500,
        model_class=ModelClass.ORCHESTRATOR,
    )

    clean = raw.strip()
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

    try:
        result = json.loads(clean)
    except json.JSONDecodeError as e:
        _debug_log(f"[orchestrate] JSON parse failed: {e}\nRaw: {raw[:300]}")
        result = {"route": "simple", "model_class": "worker"}

    _debug_log(
        f"[orchestrate] → {result.get('route')} | "
        f"shape={result.get('output_shape','—')} | "
        f"owners={len(result.get('file_owners', []))}"
    )
    return result


# ── Merge system prompts by output shape ──────────────────────────────────────

def _merge_system(shape: OutputShape, is_file_owner: bool = False) -> tuple[str, int]:
    """Return (system_prompt, max_tokens) tuned to the output shape."""
    if shape in (OutputShape.SINGLE_FILE, OutputShape.MULTI_FILE):
        prompt = (
            "You are HiveMind's assembly agent. Combine the worker outputs into "
            "a complete, working output.\n"
            "RULES:\n"
            "1. Assemble code VERBATIM — do not paraphrase or rewrite logic.\n"
            "2. Do NOT merge multiple files into one.\n"
            "3. Output EVERY file in its own fenced block with filename on the opening line:\n"
            "   ```python filename.py\n   <code>\n   ```\n"
            "4. Fix import conflicts or duplicate definitions, but change nothing else.\n"
            "5. End with a brief '## How to run' section."
        )
        tokens = 4000
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
            )
            for i, ws in enumerate(worker_specs)
        ]
        self._emit()
        _debug_log(
            f"[{self.task_id}] ({self.domain}) spawning {len(self.children)} workers"
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

        system = (
            "You are a HiveMind agent. Solve the task completely and directly. "
            "Do not ask clarifying questions — do your best with the information given."
            + domain_context
            + file_hint
        )
        return await chat(
            [{"role": "user", "content": f"Task: {self.task}{ws_context}"}],
            system=system,
            temperature=0.4,
            max_tokens=2048,
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
        # Cap each child result to avoid 413s on large trees
        MAX_CHILD_CHARS = 3000
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

        return await chat(
            [{"role": "user", "content": (
                f"Original task: {self.task}\n"
                + domain_note
                + f"\n\nSub-agent outputs:\n{child_results}{ws_context}\n\n"
                "Produce a unified, complete output."
            )}],
            system=system,
            temperature=0.3,
            max_tokens=max_tokens,
            model_class=model_class,
        )

    # ── Tree helpers ──────────────────────────────────────────────────────────

    def all_nodes(self) -> list["AgentNode"]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.all_nodes())
        return nodes
