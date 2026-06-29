"""
Agent Node — recursive unit of the HiveMind cluster.
Markers: ##SPLIT##, ##SOLVE##, ##CLARIFY##, ##RUN##
Project tasks get a shared workspace .md file all agents read/write.
##RUN## lets agents execute shell commands; sudo is routed to the user.

Inter-agent context passing:
  Parent nodes pass structured JSON context to children embedded in the
  task string:  {"task": "...", "context": {...}}
  Children parse this on receipt so they inherit project_root, known_files,
  parent findings, and shell rules — eliminating path guessing.
"""
import asyncio
import json
import re
import subprocess
import time
import uuid
import pathlib
import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
from core.llm import chat  # prefer_merge=True for synthesis passes; pool models for leaf solves

try:
    import ray
    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False

RAY_THRESHOLD = 4


def _debug_log(msg: str):
    entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n"
    pathlib.Path("hivemind_debug.log").open("a").write(entry)
    import sys
    print(f"\033[2m{entry.strip()}\033[0m", file=sys.stderr)


class AgentStatus(Enum):
    PENDING  = "pending"
    PLANNING = "planning"
    RUNNING  = "running"
    MERGING  = "merging"
    DONE     = "done"
    ERROR    = "error"


_agent_registry: dict[str, "AgentNode"] = {}
MAX_TOTAL_AGENTS = 30


def _make_task_id(parent_id: Optional[str], child_index: int) -> str:
    """Build a dotted hierarchical task ID like 'task', 'task.1', 'task.1.2'."""
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
# Heading immediately before a fenced block: ### filename.ext\n```lang\n...\n```
_HEADING_FENCED = re.compile(
    r"#{1,4}\s+([^\n`]+\.[a-zA-Z0-9]+)\s*\n```[\w]*\n(.*?)```",
    re.DOTALL,
)
_FENCED_LANG = re.compile(
    r"```(\w+)\n(.*?)```",
    re.DOTALL,
)
_LANG_EXT = {
    "python": "py", "py": "py", "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts", "bash": "sh", "sh": "sh",
    "html": "html", "css": "css", "json": "json", "yaml": "yml",
    "toml": "toml", "rust": "rs", "go": "go", "c": "c", "cpp": "cpp",
    "text": "txt",
}

def _extract_and_write_files(result: str, output_dir: pathlib.Path, root_task: str) -> list[pathlib.Path]:
    """
    Scan result text for fenced code blocks and write them as files.
    Handles three patterns:
      1. ```python filename.py  (filename on opening fence line)
      2. ### filename.py\\n```python  (markdown heading before fence)
      3. ```python  (unnamed — slug fallback, only if nothing else matched)
    Returns list of paths written.
    """
    written: list[pathlib.Path] = []
    seen_names: set[str] = set()

    def _write(fname: str, code: str):
        fname = fname.strip().lstrip("#").strip()
        if "." not in fname or " " in fname:
            return
        # Reject bare language names mistaken for filenames
        if fname.split(".")[-1] not in _LANG_EXT.values() and \
           fname.split(".")[-1] not in {"py","js","ts","html","css","json","yml","yaml","sh","md","txt","rs","go","c","cpp","toml"}:
            return
        if fname in seen_names:
            return
        seen_names.add(fname)
        path = output_dir / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        written.append(path)

    # Pass 1: inline filename on fence opening line  (```python job.py)
    for m in _FENCED_FILE.finditer(result):
        _write(m.group(1), m.group(2))

    # Pass 2: markdown heading before fence  (### job.py\n```python\n...)
    if not written:
        for m in _HEADING_FENCED.finditer(result):
            _write(m.group(1), m.group(2))

    # Pass 3: unnamed language blocks — slug fallback
    if not written:
        slug = re.sub(r"[^a-z0-9]+", "_", root_task.lower())[:32].strip("_")
        for lang, code in _FENCED_LANG.findall(result):
            ext = _LANG_EXT.get(lang.lower())
            if not ext:
                continue
            fname = f"{slug}.{ext}"
            base, n = slug, 1
            while fname in seen_names:
                fname = f"{base}_{n}.{ext}"
                n += 1
            _write(fname, code)

    return written


# ── Shell command execution ───────────────────────────────────────────────────

_RUN_MARKER = re.compile(r"##RUN##\s*\n(.*?)(?=\n##|\Z)", re.DOTALL)
_SUDO_RE    = re.compile(r"\bsudo\b")

async def _exec_command(
    cmd: str,
    cwd: Optional[pathlib.Path],
    sudo_callback: Optional[Callable],  # async fn(cmd) -> password str | None
) -> str:
    """
    Run a shell command. If it contains sudo and sudo_callback is set,
    call it to get the password from the user (never from the LLM).
    Returns combined stdout+stderr as a string.
    """
    env_cmd = cmd.strip()
    stdin_data: Optional[bytes] = None

    if _SUDO_RE.search(env_cmd) and sudo_callback:
        password = await sudo_callback(env_cmd)
        if password is not None:
            # Feed password via stdin; -S reads from stdin
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
        stdout, _ = await asyncio.wait_for(
            proc.communicate(input=stdin_data),
            timeout=60,
        )
        output = stdout.decode(errors="replace").strip()
        exit_code = proc.returncode
        result = f"[exit {exit_code}]\n{output}" if output else f"[exit {exit_code}]"
        _debug_log(f"RUN {env_cmd!r} → {result[:120]}")
        return result
    except asyncio.TimeoutError:
        return "[ERROR] command timed out after 60s"
    except Exception as e:
        return f"[ERROR] {e}"


# ── Task type detection ───────────────────────────────────────────────────────

# Must have an explicit software/code deliverable signal
_PROJECT_HINTS = re.compile(
    r"\b(build|implement|develop|refactor|migrate|setup|scaffold|"
    r"write|create|make|generate|"
    r"code|program|script|module|function|class|api|cli|server|app|"
    r"website|webpage|frontend|backend|database|pipeline)\b",
    re.IGNORECASE,
)

# Conversational/knowledge tasks — never treat as project even if other hints match
_CONVERSATIONAL = re.compile(
    r"\b(recipe|cook|food|eat|drink|meal|ingredient|marinade|how do|"
    r"what is|what are|explain|tell me|give me|can i|should i|why|"
    r"help me understand|difference between|compare|recommend|suggest|"
    r"idea|opinion|advice|tips?|trick|hack|joke|story|poem|haiku|"
    r"translate|summarize|summarise|list|pros|cons)\b",
    re.IGNORECASE,
)

def _is_project_task(task: str) -> bool:
    """True only if the task explicitly requests a software/code deliverable."""
    if _CONVERSATIONAL.search(task):
        return False
    return bool(_PROJECT_HINTS.search(task)) and len(task.split()) > 6


# ── Structured context helpers ────────────────────────────────────────────────

def _parse_task_json(raw: str) -> tuple[str, dict]:
    """
    If raw is a JSON object with a 'task' key, extract task string + context dict.
    Otherwise return raw as-is with empty context.
    """
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "task" in obj:
                return obj["task"], obj.get("context", {})
        except json.JSONDecodeError:
            pass
    return raw, {}


def _build_task_json(task: str, context: dict) -> str:
    """Wrap a subtask string with its inherited context as a JSON payload."""
    if not context:
        return task
    return json.dumps({"task": task, "context": context}, ensure_ascii=False)


async def _discover_context(output_dir: Optional[pathlib.Path], task: str = "") -> dict:
    """
    Run lightweight shell discovery to build an initial context dict for the
    root node. Finds source files and returns structured metadata children
    can use instead of guessing paths.
    """
    # Try to extract an explicit absolute path from the task description first
    import re as _re
    path_match = _re.search(r"(/(?:home|usr|var|opt|tmp|root|mnt|srv|projects?|workspace)[^\s,;\"']+)", task)
    if path_match:
        candidate = pathlib.Path(path_match.group(1))
        if candidate.is_dir():
            cwd = str(candidate)
        else:
            cwd = str(output_dir or pathlib.Path.cwd())
    else:
        cwd = str(output_dir or pathlib.Path.cwd())
    context: dict = {
        "project_root": cwd,
        "cwd": cwd,
        "known_files": [],
        "findings": "",
        "constraints": [
            "Always use absolute paths based on project_root. The known_files list contains real paths — use them directly with cat/grep/find.",
            "To read a file: ##RUN## cat <project_root>/<relative_path>. Do NOT write scripts or install packages to read files.",
            "If a ##RUN## command returns 'No such file or directory' or non-zero exit, immediately run: find <project_root> -name '<filename>' to locate the correct path. Do NOT write your answer until the file has been successfully read.",
            "Never fabricate file contents. If you cannot read a file, say so explicitly.",
            "Do NOT install packages (pip, npm, etc.) to accomplish exploration tasks. Use shell builtins: cat, grep, find, head, wc.",
        ],
    }
    try:
        proc = await asyncio.create_subprocess_shell(
            f"find {cwd} -type f -not -path '*/.git/*' -not -path '*/node_modules/*' "
            f"-not -path '*/build/*' -not -path '*/__pycache__/*' "
            f"| grep -E '\\.(js|ts|jsx|tsx|py|json|sql|md|yaml|yml|sh|kt)$' "
            f"| sort | head -80",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        files = [l for l in stdout.decode(errors="replace").splitlines() if l.strip()]
        context["known_files"] = files
        _debug_log(f"[context] discovered {len(files)} files in {cwd}")
    except Exception as e:
        _debug_log(f"[context] discovery failed: {e}")
    return context


# ── Agent Node ────────────────────────────────────────────────────────────────

@dataclass
class AgentNode:
    task: str
    depth: int = 0
    parent_id: Optional[str] = None   # parent's task_id
    child_index: int = 0               # position among siblings (0-based)
    agent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:6])
    task_id: str = ""                  # dotted hierarchical ID, set in __post_init__
    root_task: str = ""

    status: AgentStatus = AgentStatus.PENDING
    children: list["AgentNode"] = field(default_factory=list)
    result: str = ""
    error: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0

    max_depth: int = 6
    min_complexity: int = 3
    on_update: Optional[Callable] = None

    # Set by root node, propagated to children
    is_project: bool = False
    workspace_path: Optional[pathlib.Path] = None
    output_dir: Optional[pathlib.Path] = None       # where to write output files
    sudo_callback: Optional[Callable] = None         # async (cmd) -> password | None
    on_shell_run: Optional[Callable] = None          # async (cmd, output) -> None, for logging

    # Structured context passed from parent → child as JSON in the task string
    # Shape: {project_root, known_files, cwd, findings, constraints}
    task_context: dict = field(default_factory=dict)

    def __post_init__(self):
        _agent_registry[self.agent_id] = self
        self.task_id = _make_task_id(self.parent_id, self.child_index)
        # Parse JSON task payload — child agents receive {"task":..., "context":...}
        raw_task, inherited_ctx = _parse_task_json(self.task)
        self.task = raw_task
        if inherited_ctx and not self.task_context:
            self.task_context = inherited_ctx
        if not self.root_task:
            self.root_task = self.task
        if self.depth == 0:
            self.is_project = _is_project_task(self.task)
            # workspace_path is set lazily in _plan() once output_dir is known

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

    async def run(self, semaphore: asyncio.Semaphore) -> str | None:
        self.started_at = time.time()
        self.status = AgentStatus.PLANNING
        self._emit()

        try:
            async with semaphore:
                response = await self._plan()

            subtasks = self._parse_split(response)

            if subtasks and self.depth < self.max_depth:
                await self._split_and_merge(subtasks, semaphore)
            else:
                if self.depth == 0 and "##CLARIFY##" in response:
                    clarification = response.split("##CLARIFY##", 1)[1].strip()
                    self.status = AgentStatus.ERROR
                    self.error = "needs_clarification"
                    self.result = clarification
                    self._emit()
                    self.ended_at = time.time()
                    return clarification

                if "##SOLVE##" in response:
                    answer = response.split("##SOLVE##", 1)[1].strip()
                elif "##SPLIT##" in response:
                    answer = await self._force_solve()
                else:
                    answer = response.strip()
                    for label in ["##SPLIT##", "##CLARIFY##", "##SOLVE##"]:
                        if label in answer:
                            answer = answer.split(label, 1)[1].strip()
                            break

                # Root node only: write files before running ##RUN## commands
                # so agents can chmod/verify files they just created
                if self.depth == 0:
                    out = self.output_dir or pathlib.Path.cwd()
                    _pre_files = _extract_and_write_files(answer, out, self.root_task)
                    if _pre_files:
                        self.files_written = _pre_files

                # Execute any ##RUN## commands and fold output back in
                answer = await self._execute_runs(answer)

                # Write result to workspace if project task
                if self.workspace_path and answer:
                    _append_workspace(self.workspace_path, self.task_id, answer)

                _debug_log(f"[{self.task_id}] SOLVED")
                self.result = answer
                self.status = AgentStatus.DONE
                self._emit()

        except Exception as e:
            self.status = AgentStatus.ERROR
            self.error = str(e)
            self.result = f"[ERROR] {e}"
            _debug_log(f"[{self.task_id}] ERROR: {e}")
            self._emit()
        finally:
            self.ended_at = time.time()

        # Root node: extract and write code files from the final result
        # (may already be populated by the pre-RUN early write above)
        if self.depth == 0 and self.result and not self.result.startswith("[ERROR]"):
            if not getattr(self, "files_written", []):
                out = self.output_dir or pathlib.Path.cwd()
                self.files_written = _extract_and_write_files(self.result, out, self.root_task)
            for f in self.files_written:
                _debug_log(f"[{self.task_id}] FILE WRITTEN: {f}")
        else:
            if not hasattr(self, "files_written"):
                self.files_written: list[pathlib.Path] = []

        return None

    async def _execute_runs(self, answer: str) -> str:
        """
        Execute ##RUN## blocks iteratively. After each command the LLM sees the
        output and may emit more ##RUN## blocks — this enables multi-step shell
        exploration (read a file, decide what to read next, etc.).
        Caps at 12 iterations to prevent runaway loops.
        """
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

            # Feed output back and let LLM continue — may emit more ##RUN## blocks
            messages.append({"role": "user", "content": (
                f"Shell output:\n{run_block}\n\n"
                f"Your answer so far:\n{clean}\n\n"
                "Continue. You may run more ##RUN## commands to explore further, "
                "or write your final answer without ##RUN## to finish."
            )})

            current = await chat(
                messages,
                system=(
                    "You are a HiveMind agent. You have shell access via ##RUN## blocks. "
                    "Use it iteratively to explore, read files, and gather information. "
                    "When you have enough information, write your complete final answer without any ##RUN## blocks."
                ),
                temperature=0.3,
                max_tokens=2048,
                depth=self.depth,
                max_depth=self.max_depth,
                prefer_root=True,
            )
            messages.append({"role": "assistant", "content": current})

        return current

    async def _design_pass(self) -> tuple[str, int]:
        """
        For coding/project tasks: run a dedicated architecture & requirements
        pass before splitting. Writes a design doc to the workspace so all
        child agents share the same blueprint.
        Returns (design_doc_text, file_count).
        """
        # Detect if this is an exploration/documentation task vs a construction task
        _EXPLORE_HINTS = re.compile(
            r"\b(explore|read|analyse|analyze|audit|review|document|summarize|summarise|"
            r"explain|describe|investigate|examine|inspect|catalog|catalogue)\b",
            re.IGNORECASE,
        )
        _BUILD_HINTS = re.compile(
            r"\b(build|create|implement|develop|code|script|refactor|migrate|setup|make)\b",
            re.IGNORECASE,
        )
        is_exploration = bool(_EXPLORE_HINTS.search(self.root_task)) and not bool(_BUILD_HINTS.search(self.root_task))

        if is_exploration:
            # For exploration tasks, known_files tells us what source files exist
            known = ""
            if self.task_context and self.task_context.get("known_files"):
                files = self.task_context["known_files"][:50]
                known = "\n\nKnown project files (use these exact paths with cat/grep):\n" + "\n".join(f"  {f}" for f in files)
            system = (
                "You are HiveMind's ARCHITECT agent planning an EXPLORATION task — not code generation.\n\n"
                "Your job: produce a research plan that lists which source files to read and what to extract from each.\n\n"
                "Your plan must:\n"
                "1. List the key source files to read (use cat with full absolute paths — never guess paths)\n"
                "2. For each file, state exactly what information to extract\n"
                "3. Describe the output document structure\n"
                "4. RULE: Agents must use ##RUN## cat <path> to read files. No pip installs. No writing scripts. Just cat, grep, find.\n\n"
                "End your plan with a line formatted exactly as: FILES: N (where N is the number of source files to read, each becoming one subtask)."
                + known
            )
            user_content = f"Plan the exploration for:\n\n{self.root_task}"
        else:
            system = (
                "You are HiveMind's ARCHITECT agent. Before any code is written, "
                "produce a thorough design document for the task.\n\n"
                "Your design doc must cover:\n"
                "1. COMPLETE list of files to create, each with its exact filename and responsibility\n"
                "2. Key data structures / interfaces between files\n"
                "3. Edge cases and error handling requirements\n"
                "4. Visual/UX ambition — for web projects: layout, color scheme, animations, responsiveness\n"
                "5. Any shell commands needed (chmod, mkdir, pip install, etc.)\n"
                "6. How to verify the result works\n\n"
                "Be specific, concrete, and AMBITIOUS. More files = more specialisation = better result. "
                "For a webpage, split HTML structure, CSS styling, JS logic, data files, and assets into separate files. "
                "End your doc with a line formatted exactly as: FILES: N (where N is the total number of files to create). "
                "This document will be given to every agent — they will implement EXACTLY what you specify. "
                "Do NOT write any code yet. Only produce the design."
            )
            user_content = f"Design a complete, ambitious implementation plan for:\n\n{self.root_task}"

        design = await chat(
            [{"role": "user", "content": user_content}],
            system=system,
            temperature=0.2,
            max_tokens=1500,
            depth=0,
            max_depth=self.max_depth,
            prefer_root=True,  # ARCHITECT is always tier 0
        )

        # Parse file count from design doc
        file_count = 3  # default minimum
        m = re.search(r"FILES:\s*(\d+)", design)
        if m:
            file_count = max(3, int(m.group(1)))

        if self.workspace_path:
            _append_workspace(self.workspace_path, "DESIGN (task.0)", design)
        _debug_log(f"[{self.task_id}] DESIGN PASS complete — {file_count} files planned")
        return design, file_count

    async def _force_solve(self) -> str:
        ws_context = ""
        if self.workspace_path:
            existing = _read_workspace(self.workspace_path)
            if existing:
                ws_context = f"\n\nWorkspace so far:\n{existing[-2000:]}"
        return await chat(
            [{"role": "user", "content": f"Task: {self.task}\n\nSolve this completely.{ws_context}"}],
            system="You are a HiveMind agent. Answer the task directly and completely.",
            temperature=0.5, max_tokens=2048,
            depth=self.depth, max_depth=self.max_depth,
            prefer_root=True,  # force solve = leaf node = tier 0 (bottom of V)
        )

    async def _plan(self) -> str:
        if self.depth >= self.max_depth:
            return await self._force_solve()

        budget = self._budget()

        # Lazily initialise workspace path now that output_dir is known
        if self.depth == 0 and self.is_project and self.workspace_path is None:
            self.workspace_path = _workspace_path(self.root_task, self.output_dir)

        # Root node: initialise workspace, discover context, run design pass
        if self.depth == 0 and self.is_project and self.workspace_path:
            _write_workspace(
                self.workspace_path,
                f"# HiveMind Workspace\n\n**Task:** {self.root_task}\n"
                f"**Root ID:** {self.task_id}\n"
                f"**Started:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                f"---\n\n*Agent results will appear below, labeled by task ID (task, task.1, task.1.1, …)*\n"
            )
            # Discover project structure so children inherit real paths
            if not self.task_context:
                self.task_context = await _discover_context(self.output_dir, self.task)
            # For exploration tasks, skip ARCHITECT and let root split directly using known_files
            _EXPLORE_RE = re.compile(
                r"\b(explore|read|analyse|analyze|audit|review|document|summarize|summarise|"
                r"explain|describe|investigate|examine|inspect|catalog|catalogue)\b",
                re.IGNORECASE,
            )
            _BUILD_RE = re.compile(
                r"\b(build|create|implement|develop|code|script|refactor|migrate|setup|make)\b",
                re.IGNORECASE,
            )
            self._is_exploration = bool(_EXPLORE_RE.search(self.task)) and not bool(_BUILD_RE.search(self.task))
            if not self._is_exploration:
                _, self._design_file_count = await self._design_pass()

        # Read workspace context if available
        ws_context = ""
        if self.workspace_path:
            existing = _read_workspace(self.workspace_path)
            if existing:
                ws_context = f"\n\nWorkspace:\n{existing[-1500:]}"

        budget_note = (
            f" BUDGET:{budget} agents left—prefer ##SOLVE##." if budget < 8 else ""
        )
        root_ctx = (
            "" if self.depth == 0 or self.root_task == self.task
            else f"ROOT GOAL: {self.root_task}\n"
        )

        # Give every agent its hierarchical identity
        id_ctx = f"YOUR TASK ID: {self.task_id}\n"
        if self.parent_id:
            id_ctx += f"PARENT: {self.parent_id}\n"

        # Inject structured context from parent (or root discovery)
        ctx_block = ""
        if self.task_context:
            ctx = self.task_context
            lines = ["CONTEXT (from parent node — use this, do not guess):"]
            if ctx.get("project_root"):
                lines.append(f"  project_root: {ctx['project_root']}")
            if ctx.get("cwd"):
                lines.append(f"  cwd: {ctx['cwd']}")
            if ctx.get("known_files"):
                files_preview = ctx["known_files"][:40]
                lines.append(f"  known_files ({len(ctx['known_files'])} total, first {len(files_preview)} shown):")
                for f in files_preview:
                    lines.append(f"    {f}")
            if ctx.get("findings"):
                lines.append(f"  parent_findings: {ctx['findings']}")
            if ctx.get("constraints"):
                lines.append("  RULES (hard constraints — follow exactly):")
                for c in ctx["constraints"]:
                    lines.append(f"    • {c}")
            ctx_block = "\n" + "\n".join(lines) + "\n"

        file_hint = (
            "\n\nWhen your answer includes code, wrap each file in a fenced block with its filename on the opening line, e.g.:\n"
            "```python snake.py\n<code>\n```\n"
            "After all code, add a brief plain-English section titled '## How to run' explaining how to execute the result."
        ) if self.is_project else ""

        if self.depth == 0:
            file_count = getattr(self, "_design_file_count", 3)
            is_exploration = getattr(self, "_is_exploration", False)
            if is_exploration and self.task_context and self.task_context.get("known_files"):
                # Exploration mode: split into cat/grep subtasks over real files
                key_files = [
                    f for f in self.task_context["known_files"]
                    if any(f.endswith(ext) for ext in (".js", ".ts", ".py", ".json", ".sql", ".md", ".yaml", ".yml"))
                    and "node_modules" not in f and "__pycache__" not in f
                ][:12]
                project_root = self.task_context.get("project_root", "")
                files_list = "\n".join(f"  - {f}" for f in key_files)
                project_note = (
                    f"This is an EXPLORATION/DOCUMENTATION task. DO NOT write code or install packages.\n"
                    f"Split into subtasks where each subtask reads one or more real source files using ##RUN## cat <path>.\n"
                    f"The following files exist in the project (use these EXACT paths):\n{files_list}\n"
                    f"project_root: {project_root}\n"
                    f"Each subtask should: read the file(s), extract the relevant information, and return a structured section for the final document.\n"
                    f"You MUST use ##SPLIT## — do NOT ##SOLVE## this yourself. Split into {min(len(key_files), 6)} subtasks covering different parts of the codebase.\n"
                )
            elif self.is_project:
                project_note = (
                    f"This is a CODING/PROJECT task. A design document has been written to the workspace — READ IT in full before splitting.\n"
                    f"The design specifies {file_count} files. You MUST produce exactly {file_count} subtasks, one per file.\n"
                    f"Each subtask must name the specific file it will create and implement it completely — do not collapse multiple files into one subtask.\n"
                    f"Do NOT use ##SOLVE## — the split is mandatory. More agents = higher quality output.\n"
                    f"Be AMBITIOUS: each agent should produce polished, complete, production-quality work for its file.\n"
                )
            else:
                project_note = "For simple/conversational tasks, solve directly.\n"
            system = (
                "You are HiveMind — a recursive multi-agent AI cluster.\n"
                + id_ctx
                + "You are the ROOT orchestrator.\n"
                "When splitting, children will be assigned IDs like task.1, task.2, etc.\n"
                + project_note
                + budget_note
                + file_hint
            )
        else:
            # At leaf depth (max_depth-1) always solve; otherwise always split into 2
            is_leaf_depth = self.depth >= self.max_depth - 1
            if is_leaf_depth:
                split_instruction = (
                    "You are at leaf depth — always use ##SOLVE##. Implement your subtask completely.\n"
                )
            else:
                splits_remaining = self.max_depth - self.depth - 1
                split_instruction = (
                    f"You MUST use ##SPLIT## — split your subtask into exactly 2 independent sub-parts.\n"
                    f"Do NOT use ##SOLVE## at this depth ({self.depth}/{self.max_depth}). "
                    f"Each of your children will split again ({splits_remaining} more level(s) before leaf).\n"
                    "Name each sub-part precisely so it is fully self-contained and independently implementable.\n"
                    "Children will be assigned IDs extending yours (e.g. if you are task.2, children are task.2.1, task.2.2).\n"
                )
            system = (
                f"You are a HiveMind agent (depth {self.depth}/{self.max_depth}).\n"
                + id_ctx
                + root_ctx
                + ctx_block
                + "The workspace contains a DESIGN DOCUMENT — read it carefully before acting.\n"
                + split_instruction
                + budget_note
                + file_hint
            )

        system += (
            "\n\nRespond with exactly one:\n"
            "##SPLIT##\n- [subtask 1]\n- [subtask 2]\n\n"
            "##SOLVE##\n[answer]\n\n"
            + ("##CLARIFY##\n[one sentence: what is missing]\n" if self.depth == 0 else "")
            + "\n\nWithin a ##SOLVE## answer you MAY run shell commands by embedding:\n"
            "##RUN##\n<single shell command>\n"
            "The command will be executed and its output fed back to you. "
            "Use ##RUN## freely to explore, read, and understand — treat it like a terminal:\n"
            "  - Explore structure: find . -type f -name '*.js' | head -40\n"
            "  - Read files: cat path/to/file.js\n"
            "  - Search code: grep -r 'functionName' src/\n"
            "  - Understand deps: cat package.json\n"
            "  - Run/verify: python3 script.py, chmod, pip install, etc.\n"
            "You may chain multiple ##RUN## blocks in sequence — each runs and feeds back before you continue. "
            "Never use ##RUN## for interactive commands. "
            "If the command needs sudo, write it normally — the user will be prompted securely.\n"
            "\nNothing else."
        )

        user_msg = f"Task: {self.task}{ws_context}"

        # Coding tasks need more tokens and the strongest model available
        plan_tokens = 2000 if self.is_project else 600

        # V-shape: depth 0 and leaf nodes → tier 0; middle depths → lighter tiers
        is_leaf_solve = "##SOLVE##" in system or self.depth >= self.max_depth - 1
        response = await chat(
            [{"role": "user", "content": user_msg}],
            system=system,
            temperature=0.3,
            max_tokens=plan_tokens,
            depth=self.depth,
            max_depth=self.max_depth,
            prefer_root=(self.depth == 0 or is_leaf_solve),
        )

        _debug_log(f"[{self.task_id}] depth={self.depth} budget={budget} | {response[:120].strip()}")
        return response

    def _parse_split(self, response: str) -> list[str]:
        if "##SPLIT##" not in response:
            return []
        after = response.split("##SPLIT##", 1)[1]
        subtasks = []
        for line in after.strip().splitlines():
            line = line.strip()
            if line.startswith("- "):
                t = line[2:].strip()
                if t:
                    subtasks.append(t)
        return subtasks if len(subtasks) >= 2 else []

    def _subtasks_have_dependencies(self, subtasks: list[str]) -> bool:
        dep_phrases = re.compile(
            r"\b(result of|output of|based on|using the|after|once|from step|from part)\b",
            re.IGNORECASE,
        )
        for task in subtasks:
            if dep_phrases.search(task):
                return True
        return False

    async def _split_and_merge(self, subtasks: list[str], semaphore: asyncio.Semaphore):
        self.status = AgentStatus.RUNNING
        self._emit()

        budget = self._budget()
        if budget <= 0:
            _debug_log(f"[{self.task_id}] BUDGET EXHAUSTED — solving directly")
            self.result = await self._force_solve()
            self.status = AgentStatus.DONE
            self._emit()
            return

        if len(subtasks) > budget:
            _debug_log(f"[{self.task_id}] BUDGET TRIM {len(subtasks)}→{budget}")
            subtasks = subtasks[:budget]

        # Non-root agents always split into exactly 2 for a balanced binary tree
        if self.depth > 0 and len(subtasks) != 2:
            if len(subtasks) > 2:
                _debug_log(f"[{self.task_id}] BINARY TRIM {len(subtasks)}→2")
                subtasks = subtasks[:2]
            elif len(subtasks) == 1:
                # Can't split 1 item — solve directly
                self.result = await self._force_solve()
                self.status = AgentStatus.DONE
                self._emit()
                return

        self.children = [
            AgentNode(
                task=_build_task_json(st, self.task_context),
                task_context=self.task_context,
                depth=self.depth + 1,
                parent_id=self.task_id,
                child_index=i,
                root_task=self.root_task,
                max_depth=self.max_depth,
                min_complexity=self.min_complexity,
                on_update=self.on_update,
                is_project=self.is_project,
                workspace_path=self.workspace_path,
                output_dir=self.output_dir,
                sudo_callback=self.sudo_callback,
                on_shell_run=self.on_shell_run,
            )
            for i, st in enumerate(subtasks)
        ]
        _debug_log(f"[{self.task_id}] SPLIT into {len(self.children)}: {[c.task_id for c in self.children]} | registry={len(_agent_registry)}/{MAX_TOTAL_AGENTS}")
        self._emit()

        sequential = self._subtasks_have_dependencies(subtasks)
        if sequential:
            _debug_log(f"[{self.task_id}] SEQUENTIAL mode")
            for child in self.children:
                await child.run(semaphore)
        elif _RAY_AVAILABLE and len(subtasks) >= RAY_THRESHOLD:
            ray.init(ignore_reinit_error=True, log_to_driver=False)
            await self._run_children_ray(semaphore)
        else:
            await asyncio.gather(*[child.run(semaphore) for child in self.children])

        self.status = AgentStatus.MERGING
        self._emit()
        self.result = await self._merge()
        self.status = AgentStatus.DONE
        self._emit()

    async def _run_children_ray(self, semaphore: asyncio.Semaphore):
        budget_per_child = max(1, self._budget() // len(self.children))
        refs = [
            _ray_run_subtask.remote(
                child.task, child.depth, child.root_task,
                child.max_depth, child.min_complexity, budget_per_child,
            )
            for child in self.children
        ]
        _debug_log(f"[{self.task_id}] RAY dispatched {len(refs)} remote tasks")

        loop = asyncio.get_event_loop()
        payloads = await loop.run_in_executor(None, ray.get, refs)

        from core.providers import get_pool
        pool = get_pool()
        provider_map = {p.name: p for p in pool.providers}
        for result, worker_stats in payloads:
            for name, (calls, errors) in worker_stats.items():
                if name in provider_map:
                    for key in provider_map[name].api_keys:
                        key.calls += calls // max(1, len(provider_map[name].api_keys))
                    provider_map[name].errors += errors

        for child, (result, _) in zip(self.children, payloads):
            child.result = result or ""
            child.status = AgentStatus.DONE
            child._emit()

    async def _merge(self) -> str:
        child_results = "\n\n".join(
            f"### [{c.task_id}] {c.task}\n{c.result}" for c in self.children
        )

        ws_context = ""
        if self.workspace_path:
            existing = _read_workspace(self.workspace_path)
            if existing:
                ws_context = f"\n\nWorkspace:\n{existing[-2000:]}"

        system = (
            "You are HiveMind's integration agent. Your job is to combine sub-agent outputs into one unified response.\n"
            "CRITICAL RULES:\n"
            "1. Do NOT rename, restructure, or reorganise files — use the exact filenames each agent produced.\n"
            "2. Do NOT collapse multiple files into one — every file gets its own fenced block.\n"
            "3. Output EVERY file, even if it seems simple — missing files break the project.\n"
            "4. Resolve conflicts by picking the more complete/correct version.\n"
            "5. Each file must be a complete, standalone fenced block with its filename on the opening line:\n"
            "   ```python filename.py\\n<full code>\\n```\n"
            "6. End with a concise '## How to run' section.\n"
            "Do not summarise or truncate — output the full content of every file."
        )
        # V-shape going back up: depth 0 merge → tier 0, deeper merges → lighter
        result = await chat(
            [{"role": "user", "content": (
                f"Original task: {self.task}\n\n"
                f"Sub-agent outputs:\n{child_results}{ws_context}\n\n"
                "Produce a unified, complete answer."
            )}],
            system=system,
            depth=self.depth,
            max_depth=self.max_depth,
            temperature=0.3,
            max_tokens=4000,
            prefer_merge=True,
        )

        # Root node finalises workspace with tree summary
        if self.depth == 0 and self.workspace_path:
            tree_lines = []
            def _tree(node: "AgentNode", indent: int = 0):
                status_icon = {"done": "✓", "error": "✗", "merging": "⟳"}.get(node.status.value, "…")
                tree_lines.append("  " * indent + f"{status_icon} [{node.task_id}] {node.task[:80]}")
                for child in node.children:
                    _tree(child, indent + 1)
            _tree(self)
            tree_md = "\n".join(tree_lines)
            _append_workspace(self.workspace_path, "Agent Tree", f"```\n{tree_md}\n```")
            _append_workspace(self.workspace_path, "Final Answer", result)

        return result

    def all_nodes(self) -> list["AgentNode"]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.all_nodes())
        return nodes


def _run_subtask_in_worker(
    task: str, depth: int, root_task: str, max_depth: int,
    min_complexity: int, budget: int,
) -> tuple[str, dict[str, tuple[int, int]]]:
    import asyncio as _asyncio

    async def _run():
        node = AgentNode(
            task=task, depth=depth, root_task=root_task,
            max_depth=max_depth, min_complexity=min_complexity, on_update=None,
        )
        sem = _asyncio.Semaphore(8)
        await node.run(sem)

        from core.providers import get_pool
        stats = {p.name: (p.calls, p.errors) for p in get_pool().providers}
        return node.result, stats

    return _asyncio.run(_run())


if _RAY_AVAILABLE:
    _ray_run_subtask = ray.remote(_run_subtask_in_worker)
