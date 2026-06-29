"""
Agent Node — recursive unit of the HiveMind cluster.
Markers: ##SPLIT##, ##SOLVE##, ##CLARIFY##, ##RUN##
Project tasks get a shared workspace .md file all agents read/write.
##RUN## lets agents execute shell commands; sudo is routed to the user.
"""
import asyncio
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
_FENCED_LANG = re.compile(
    r"```(\w+)\n(.*?)```",
    re.DOTALL,
)
_LANG_EXT = {
    "python": "py", "py": "py", "javascript": "js", "js": "js",
    "typescript": "ts", "ts": "ts", "bash": "sh", "sh": "sh",
    "html": "html", "css": "css", "json": "json", "yaml": "yml",
    "toml": "toml", "rust": "rs", "go": "go", "c": "c", "cpp": "cpp",
}

def _extract_and_write_files(result: str, output_dir: pathlib.Path, root_task: str) -> list[pathlib.Path]:
    """
    Scan result text for fenced code blocks and write them as files.
    Returns list of paths written.
    """
    written: list[pathlib.Path] = []
    seen_names: set[str] = set()

    # First pass: blocks with explicit filenames (```python snake.py or ```# snake.py)
    for m in _FENCED_FILE.finditer(result):
        fname = m.group(1).strip().lstrip("#").strip()
        code = m.group(2)
        # Accept plain filenames and relative subpaths (src/css/style.css)
        # Reject bare language tags (no dot) and anything with spaces
        if "." in fname and " " not in fname:
            path = output_dir / fname
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(code)
            written.append(path)
            seen_names.add(fname)

    # Second pass: language-tagged blocks — derive filename from task if only one block
    if not written:
        blocks = _FENCED_LANG.findall(result)
        if blocks:
            slug = re.sub(r"[^a-z0-9]+", "_", root_task.lower())[:32].strip("_")
            for lang, code in blocks:
                ext = _LANG_EXT.get(lang.lower())
                if not ext:
                    continue
                fname = f"{slug}.{ext}"
                # avoid collisions if multiple blocks of same type
                base, n = fname, 1
                while fname in seen_names:
                    fname = f"{base[:-len(ext)-1]}_{n}.{ext}"
                    n += 1
                seen_names.add(fname)
                path = output_dir / fname
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(code)
                written.append(path)

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

_PROJECT_HINTS = re.compile(
    r"\b(build|create|write|implement|design|generate|make|produce|develop|draft|"
    r"plan|outline|code|script|report|document|file|project|deliverable|setup|"
    r"refactor|migrate|analyse|analyze|research)\b",
    re.IGNORECASE,
)

def _is_project_task(task: str) -> bool:
    """True if the task expects a concrete deliverable rather than a conversational answer."""
    return bool(_PROJECT_HINTS.search(task)) and len(task.split()) > 6


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

    def __post_init__(self):
        _agent_registry[self.agent_id] = self
        self.task_id = _make_task_id(self.parent_id, self.child_index)
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
        Find ##RUN## blocks in the answer, execute each command, replace the
        marker with the output, then ask the LLM to continue from the results.
        Returns the (possibly enriched) answer.
        """
        runs = _RUN_MARKER.findall(answer)
        if not runs:
            return answer

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
            return answer

        # Strip the ##RUN## blocks from the answer
        clean = _RUN_MARKER.sub("", answer).strip()
        run_block = "\n\n".join(run_results)

        # Ask the LLM to interpret the output and complete its response
        followup = await chat(
            [{"role": "user", "content": (
                f"You ran these shell commands as part of your task:\n\n"
                f"{run_block}\n\n"
                f"Your partial answer so far:\n{clean}\n\n"
                f"Continue your answer incorporating the command output. "
                f"If the commands succeeded, confirm it. If they failed, explain and suggest a fix."
            )}],
            system="You are a HiveMind agent completing a task after running shell commands. Be concise.",
            temperature=0.3,
            max_tokens=1024,
            depth=self.depth,
        )
        return f"{clean}\n\n{followup}".strip()

    async def _design_pass(self) -> tuple[str, int]:
        """
        For coding/project tasks: run a dedicated architecture & requirements
        pass before splitting. Writes a design doc to the workspace so all
        child agents share the same blueprint.
        Returns (design_doc_text, file_count).
        """
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
        design = await chat(
            [{"role": "user", "content": f"Design a complete, ambitious implementation plan for:\n\n{self.root_task}"}],
            system=system,
            temperature=0.2,
            max_tokens=1500,
            depth=0,
            prefer_merge=True,  # architecture pass uses the heaviest model
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
            temperature=0.5, max_tokens=2048, depth=self.depth,
            prefer_root=self.is_project,
        )

    async def _plan(self) -> str:
        if self.depth >= self.max_depth:
            return await self._force_solve()

        budget = self._budget()

        # Lazily initialise workspace path now that output_dir is known
        if self.depth == 0 and self.is_project and self.workspace_path is None:
            self.workspace_path = _workspace_path(self.root_task, self.output_dir)

        # Root node: initialise workspace, run design pass, then plan
        if self.depth == 0 and self.is_project and self.workspace_path:
            _write_workspace(
                self.workspace_path,
                f"# HiveMind Workspace\n\n**Task:** {self.root_task}\n"
                f"**Root ID:** {self.task_id}\n"
                f"**Started:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                f"---\n\n*Agent results will appear below, labeled by task ID (task, task.1, task.1.1, …)*\n"
            )
            # Design pass — write architecture doc before any splits/solves
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

        file_hint = (
            "\n\nWhen your answer includes code, wrap each file in a fenced block with its filename on the opening line, e.g.:\n"
            "```python snake.py\n<code>\n```\n"
            "After all code, add a brief plain-English section titled '## How to run' explaining how to execute the result."
        ) if self.is_project else ""

        if self.depth == 0:
            file_count = getattr(self, "_design_file_count", 3)
            project_note = (
                f"This is a CODING/PROJECT task. A design document has been written to the workspace — READ IT in full before splitting.\n"
                f"The design specifies {file_count} files. You MUST produce exactly {file_count} subtasks, one per file.\n"
                f"Each subtask must name the specific file it will create and implement it completely — do not collapse multiple files into one subtask.\n"
                f"Do NOT use ##SOLVE## — the split is mandatory. More agents = higher quality output.\n"
                f"Be AMBITIOUS: each agent should produce polished, complete, production-quality work for its file.\n"
            ) if self.is_project else "For simple/conversational tasks, solve directly.\n"
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
            system = (
                f"You are a HiveMind agent (depth {self.depth}/{self.max_depth}).\n"
                + id_ctx
                + root_ctx
                + "The workspace contains a DESIGN DOCUMENT — read it carefully and implement exactly what it specifies for your subtask.\n"
                "Solve atomically if possible. Split only if your subtask has truly independent parts.\n"
                "If you split, your children will be assigned IDs extending your own (e.g. if you are task.2, children are task.2.1, task.2.2).\n"
                + ("depth>=3: always ##SOLVE##.\n" if self.depth >= 3 else "")
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
            "Use this for: chmod, mkdir, pip install, moving files, or verifying output. "
            "Never use ##RUN## for commands requiring interactive input. "
            "If the command needs sudo, write it normally — the user will be prompted for their password securely and it will never be shown to you.\n"
            "\nNothing else."
        )

        user_msg = f"Task: {self.task}{ws_context}"

        # Coding tasks need more tokens and the strongest model available
        plan_tokens = 2000 if self.is_project else 600

        # Root orchestration uses root model; leaf solves use pool (light models going down)
        response = await chat(
            [{"role": "user", "content": user_msg}],
            system=system,
            temperature=0.3,
            max_tokens=plan_tokens,
            depth=self.depth,
            prefer_root=(self.depth == 0),  # only root orchestrator gets the heavy model going down
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

        self.children = [
            AgentNode(
                task=st,
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
        result = await chat(
            [{"role": "user", "content": (
                f"Original task: {self.task}\n\n"
                f"Sub-agent outputs:\n{child_results}{ws_context}\n\n"
                "Produce a unified, complete answer."
            )}],
            system=system,
            depth=self.depth,
            temperature=0.3,
            max_tokens=4000,
            prefer_merge=True,  # merge/synthesis passes always get the heaviest model
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
