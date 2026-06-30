"""
Agent Node — recursive unit of the HiveMind cluster.

Orchestration model:
  Root node ALWAYS runs a classify-then-route pass using ORCHESTRATOR class:
    • Simple / conversational  → solve directly (1 agent, WORKER)
    • Ambiguous                → return clarification question to user
    • Complex                  → decompose into N subtasks, each tagged with
                                 a task_type and model_class; dispatch agents

  Subtask agents receive their assigned model_class and solve or split
  at most once more (depth cap). They do NOT force-split — they decide
  based on whether their subtask is still composite.

  Merge:
    • Simple tasks (1 agent)   → no merge needed
    • Complex (multi-agent)    → ORCHESTRATOR merges at root; lighter merges
                                 mid-tree use ANALYST

Shell commands: ##RUN## blocks are still supported inside ##SOLVE## answers.
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
from core.model_classes import ModelClass, class_for_task_type, TASK_TYPE_TO_CLASS

try:
    import ray
    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False

RAY_THRESHOLD = 4


def _debug_log(msg: str):
    entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n"
    pathlib.Path("hivemind_debug.log").open("a").write(entry)
    print(f"\033[2m{entry.strip()}\033[0m", file=_sys.stderr)


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
    "text": "txt",
}


def _extract_and_write_files(result: str, output_dir: pathlib.Path,
                              root_task: str) -> list[pathlib.Path]:
    written: list[pathlib.Path] = []
    seen_names: set[str] = set()

    def _write(fname: str, code: str):
        fname = fname.strip().lstrip("#").strip()
        if "." not in fname or " " in fname:
            return
        if fname.split(".")[-1] not in _LANG_EXT.values() and \
           fname.split(".")[-1] not in {
               "py", "js", "ts", "html", "css", "json", "yml",
               "yaml", "sh", "md", "txt", "rs", "go", "c", "cpp", "toml"
           }:
            return
        if fname in seen_names:
            return
        seen_names.add(fname)
        path = output_dir / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)
        written.append(path)

    for m in _FENCED_FILE.finditer(result):
        _write(m.group(1), m.group(2))

    if not written:
        for m in _HEADING_FENCED.finditer(result):
            _write(m.group(1), m.group(2))

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
You are HiveMind's Orchestrator. Your ONLY job is to classify the user's task
and decide how to handle it. You must reply with valid JSON and nothing else.

Classification rules:
  "simple"     — Can be answered in one pass without sub-agents.
                 Examples: factual Q&A, explanations, short writing (guides,
                 emails, poems), simple data lookup, conversation.
  "ambiguous"  — The task is unclear, missing critical info, or could mean
                 multiple very different things. Ask ONE clarifying question.
  "complex"    — Requires multiple distinct pieces of work that benefit from
                 parallel specialised agents. Examples: multi-file codebases,
                 large research tasks with many independent subtopics.

For "simple" and "ambiguous" reply exactly:
  {"route": "simple", "model_class": "worker"}
  {"route": "ambiguous", "clarification": "One sentence question."}

For "complex" reply:
  {
    "route": "complex",
    "subtasks": [
      {
        "task": "Specific self-contained description of what this agent does.",
        "task_type": "<one of: plan, analyze, architect, research, code, write, implement, extract, generate, test, format, merge_simple, summarize_short>",
        "model_class": "<one of: orchestrator, analyst, worker, fast>"
      },
      ...
    ]
  }

Subtask rules for "complex":
  • Minimum 2, maximum 8 subtasks. Only create as many as genuinely needed.
  • Each subtask must be independently executable (no dependency on another's output
    unless you sequence them — mark sequential ones with "sequential": true).
  • Assign model_class based on what the subtask actually needs:
      orchestrator — complex reasoning, architecture decisions, final synthesis
      analyst      — analysis, research, structured thinking
      worker       — writing, coding, extraction, generation
      fast         — formatting, simple merging, short summaries
  • Do NOT create subtasks just to have more agents. 3 focused agents beat 10 vague ones.
"""


def _safe_model_class(value: str, default: ModelClass = ModelClass.WORKER) -> ModelClass:
    """Parse a model class string, falling back to default on invalid input."""
    try:
        return ModelClass(value.strip().lower())
    except (ValueError, AttributeError):
        _debug_log(f"[model_class] unknown value {value!r}, defaulting to {default.value}")
        return default


async def _orchestrate(task: str) -> dict:
    """
    Call the ORCHESTRATOR model to classify and plan the task.
    Returns the parsed JSON dict.
    """
    raw = await chat(
        [{"role": "user", "content": f"Task: {task}"}],
        system=_CLASSIFY_SYSTEM,
        temperature=0.2,
        max_tokens=1200,
        model_class=ModelClass.ORCHESTRATOR,
    )

    # Strip markdown fences if model wrapped JSON
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        inner = []
        for line in lines[1:]:
            if line.strip() == "```":
                break
            inner.append(line)
        clean = "\n".join(inner).strip()

    # Find outermost JSON object
    start = clean.find("{")
    end   = clean.rfind("}")
    if start != -1 and end != -1:
        clean = clean[start:end + 1]

    try:
        result = json.loads(clean)
    except json.JSONDecodeError as e:
        _debug_log(f"[orchestrate] JSON parse failed: {e}\nRaw: {raw[:300]}")
        # Fallback: treat as simple
        result = {"route": "simple", "model_class": "worker"}

    _debug_log(f"[orchestrate] → {result.get('route')} | subtasks={len(result.get('subtasks', []))}")
    return result


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

    # Model class assigned by the orchestrator for this node's work
    model_class: ModelClass = ModelClass.WORKER

    max_depth: int = 4       # lowered default; orchestrator sets per-task
    min_complexity: int = 3  # kept for compat, not used for routing
    on_update: Optional[Callable] = None

    is_project: bool = False
    workspace_path: Optional[pathlib.Path] = None
    output_dir: Optional[pathlib.Path] = None
    sudo_callback: Optional[Callable] = None
    on_shell_run: Optional[Callable] = None
    task_context: dict = field(default_factory=dict)

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

    async def run(self, semaphore: asyncio.Semaphore) -> str | None:
        self.started_at = time.time()
        self.status = AgentStatus.PLANNING
        self._emit()

        try:
            if self.depth == 0:
                await self._root_run(semaphore)
            else:
                await self._subtask_run(semaphore)
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.error = str(e)
            self.result = f"[ERROR] {e}"
            _debug_log(f"[{self.task_id}] ERROR: {e}")
            self._emit()
        finally:
            self.ended_at = time.time()

        # Write output files at root
        if self.depth == 0 and self.result and not self.result.startswith("[ERROR]"):
            if not getattr(self, "files_written", []):
                out = self.output_dir or pathlib.Path.cwd()
                self.files_written = _extract_and_write_files(
                    self.result, out, self.root_task
                )
            for f in getattr(self, "files_written", []):
                _debug_log(f"[{self.task_id}] FILE WRITTEN: {f}")
        else:
            if not hasattr(self, "files_written"):
                self.files_written: list[pathlib.Path] = []

        return None

    # ── Root: classify → route ────────────────────────────────────────────────

    async def _root_run(self, semaphore: asyncio.Semaphore):
        async with semaphore:
            plan = await _orchestrate(self.task)

        route = plan.get("route", "simple")

        if route == "ambiguous":
            clarification = plan.get("clarification", "Could you clarify what you mean?")
            self.result = clarification
            self.status = AgentStatus.ERROR
            self.error = "needs_clarification"
            self._emit()
            return

        if route == "simple":
            self.status = AgentStatus.RUNNING
            self._emit()
            answer = await self._solve_direct(
                model_class=_safe_model_class(plan.get("model_class", "worker")),
            )
            answer = await self._execute_runs(answer)
            self.result = answer
            self.status = AgentStatus.DONE
            self._emit()
            return

        # "complex" — spawn subtask agents
        subtasks = plan.get("subtasks", [])
        if not subtasks:
            # Orchestrator said complex but gave no subtasks — solve directly
            self.status = AgentStatus.RUNNING
            self._emit()
            self.result = await self._solve_direct(model_class=ModelClass.ORCHESTRATOR)
            self.status = AgentStatus.DONE
            self._emit()
            return

        # Cap to budget
        budget = self._budget() - 1  # -1 for root
        if len(subtasks) > budget:
            _debug_log(f"[{self.task_id}] budget trim {len(subtasks)}→{budget}")
            subtasks = subtasks[:budget]

        # Mark as project if any subtask involves code/file generation
        _CODE_TYPES = {"code", "implement", "generate", "architect"}
        self.is_project = any(
            st.get("task_type", "") in _CODE_TYPES for st in subtasks
        )

        # Initialise workspace
        self.workspace_path = _workspace_path(
            self.root_task, self.output_dir or pathlib.Path.cwd()
        )
        _write_workspace(
            self.workspace_path,
            f"# HiveMind Workspace\n\n**Task:** {self.root_task}\n"
            f"**Started:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"**Subtasks:** {len(subtasks)}\n\n---\n"
        )

        self.status = AgentStatus.RUNNING
        self.children = [
            AgentNode(
                task=st["task"],
                depth=1,
                parent_id=self.task_id,
                child_index=i,
                root_task=self.root_task,
                model_class=_safe_model_class(st.get("model_class", "worker")),
                max_depth=self.max_depth,
                on_update=self.on_update,
                is_project=self.is_project,
                workspace_path=self.workspace_path,
                output_dir=self.output_dir,
                sudo_callback=self.sudo_callback,
                on_shell_run=self.on_shell_run,
                task_context=self.task_context,
            )
            for i, st in enumerate(subtasks)
        ]
        self._emit()
        _debug_log(
            f"[{self.task_id}] spawning {len(self.children)} subtask agents: "
            + ", ".join(f"{c.task_id}({c.model_class.value})" for c in self.children)
        )

        # Check for sequential dependencies
        sequential_indices = {
            i for i, st in enumerate(subtasks) if st.get("sequential")
        }
        if sequential_indices:
            _debug_log(f"[{self.task_id}] sequential subtasks: {sequential_indices}")
            for child in self.children:
                await child.run(semaphore)
        else:
            await asyncio.gather(*[child.run(semaphore) for child in self.children])

        self.status = AgentStatus.MERGING
        self._emit()
        self.result = await self._merge(model_class=ModelClass.ORCHESTRATOR)
        self.status = AgentStatus.DONE
        self._emit()

    # ── Subtask: solve or split once more if needed ───────────────────────────

    async def _subtask_run(self, semaphore: asyncio.Semaphore):
        self.status = AgentStatus.RUNNING
        self._emit()

        # Subtasks at depth >= max_depth always solve directly
        if self.depth >= self.max_depth:
            answer = await self._solve_direct(model_class=self.model_class)
            answer = await self._execute_runs(answer)
            self.result = answer
            self.status = AgentStatus.DONE
            self._emit()
            return

        # Ask the assigned model whether to split further or solve
        decision = await self._subtask_plan()

        subtasks = self._parse_split(decision)
        if subtasks and self._budget() >= len(subtasks) + 1:
            # Further decomposition warranted
            self.children = [
                AgentNode(
                    task=st,
                    depth=self.depth + 1,
                    parent_id=self.task_id,
                    child_index=i,
                    root_task=self.root_task,
                    model_class=self.model_class,  # inherit class unless overridden
                    max_depth=self.max_depth,
                    on_update=self.on_update,
                    is_project=self.is_project,
                    workspace_path=self.workspace_path,
                    output_dir=self.output_dir,
                    sudo_callback=self.sudo_callback,
                    on_shell_run=self.on_shell_run,
                    task_context=self.task_context,
                )
                for i, st in enumerate(subtasks)
            ]
            self._emit()
            await asyncio.gather(*[child.run(semaphore) for child in self.children])
            self.status = AgentStatus.MERGING
            self._emit()
            self.result = await self._merge(model_class=ModelClass.ANALYST)
        else:
            # Solve directly — extract answer from whatever the model returned
            if "##SOLVE##" in decision:
                answer = decision.split("##SOLVE##", 1)[1].strip()
            elif "##SPLIT##" in decision:
                # Model wanted to split but budget is exhausted — re-ask to solve
                answer = await self._solve_direct(model_class=self.model_class)
            else:
                # Model wrote its answer inline without a marker
                answer = decision.strip()
            answer = await self._execute_runs(answer)
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

        file_hint = (
            "\n\nWhen your answer includes code, wrap each file in a fenced block "
            "with its filename on the opening line, e.g.:\n"
            "```python snake.py\n<code>\n```"
        ) if self.is_project else ""

        system = (
            "You are a HiveMind agent. Solve the task completely and directly. "
            "Do not ask clarifying questions — do your best with the information given."
            + file_hint
        )
        return await chat(
            [{"role": "user", "content": f"Task: {self.task}{ws_context}"}],
            system=system,
            temperature=0.4,
            max_tokens=2048,
            model_class=model_class,
        )

    async def _subtask_plan(self) -> str:
        """
        Ask the subtask agent: solve directly or split into parts?
        Returns the raw response (may contain ##SOLVE## or ##SPLIT##).
        """
        ws_context = ""
        if self.workspace_path:
            existing = _read_workspace(self.workspace_path)
            if existing:
                ws_context = f"\n\nWorkspace:\n{existing[-1000:]}"

        budget = self._budget()
        budget_note = " You MUST use ##SOLVE## — agent budget is low." if budget < 4 else ""

        system = (
            f"You are a HiveMind agent (depth {self.depth}/{self.max_depth}), "
            f"assigned model class: {self.model_class.value}.\n"
            f"Root goal: {self.root_task}\n\n"
            "Decide: can you solve your subtask directly, or does it contain "
            "genuinely independent parts that benefit from parallel agents?\n\n"
            "If you can solve it in one pass → respond with:\n"
            "##SOLVE##\n[your complete answer]\n\n"
            "If it has 2-3 truly independent sub-parts → respond with:\n"
            "##SPLIT##\n- [sub-part 1]\n- [sub-part 2]\n\n"
            "Only split if the sub-parts are independent and complex enough to warrant it. "
            "Simple subtasks should ALWAYS use ##SOLVE##." + budget_note
        )

        return await chat(
            [{"role": "user", "content": f"Task: {self.task}{ws_context}"}],
            system=system,
            temperature=0.3,
            max_tokens=1500,
            model_class=self.model_class,
        )

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

    # ── Shell execution ───────────────────────────────────────────────────────

    async def _execute_runs(self, answer: str) -> str:
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
                model_class=self.model_class,
            )
            messages.append({"role": "assistant", "content": current})

        return current

    # ── Merge ─────────────────────────────────────────────────────────────────

    async def _merge(self, model_class: ModelClass) -> str:
        child_results = "\n\n".join(
            f"### [{c.task_id}] {c.task}\n{c.result}"
            for c in self.children
        )

        ws_context = ""
        if self.workspace_path:
            existing = _read_workspace(self.workspace_path)
            if existing:
                ws_context = f"\n\nWorkspace:\n{existing[-1500:]}"

        is_code = self.is_project or any(
            "```" in c.result for c in self.children if c.result
        )

        if is_code:
            system = (
                "You are HiveMind's integration agent. Combine sub-agent outputs "
                "into one unified response.\n"
                "RULES:\n"
                "1. Do NOT rename or restructure files.\n"
                "2. Do NOT collapse multiple files into one.\n"
                "3. Output EVERY file in its own fenced block with filename on opening line.\n"
                "4. End with a concise '## How to run' section."
            )
        else:
            system = (
                "You are HiveMind's synthesis agent. Combine the sub-agent outputs "
                "into one coherent, well-structured final answer. "
                "Eliminate redundancy. Preserve all important information."
            )

        # Token budget: code merges need more room; prose merges can be tight
        merge_tokens = 3000 if is_code else 1500

        return await chat(
            [{"role": "user", "content": (
                f"Original task: {self.task}\n\n"
                f"Sub-agent outputs:\n{child_results}{ws_context}\n\n"
                "Produce a unified, complete answer."
            )}],
            system=system,
            temperature=0.3,
            max_tokens=merge_tokens,
            model_class=model_class,
        )

    # ── Tree helpers ──────────────────────────────────────────────────────────

    def all_nodes(self) -> list["AgentNode"]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.all_nodes())
        return nodes


# ── Backwards compat: _is_project_task and _CONVERSATIONAL imported by widget_server.py ──

_CONVERSATIONAL = re.compile(
    r"\b(recipe|cook|food|eat|drink|meal|ingredient|marinade|how do|"
    r"what is|what are|explain|tell me|give me|can i|should i|why|"
    r"help me understand|difference between|compare|recommend|suggest|"
    r"idea|opinion|advice|tips?|trick|hack|joke|story|poem|haiku|"
    r"translate|summarize|summarise|list|pros|cons|guide|tutorial|"
    r"newbie|beginner|intro|overview|basics|cheat.?sheet)\b",
    re.IGNORECASE,
)

_PROJECT_HINTS = re.compile(
    r"\b(build|implement|develop|refactor|migrate|setup|scaffold|"
    r"write|create|make|generate|"
    r"code|program|script|module|function|class|api|cli|server|app|"
    r"website|webpage|frontend|backend|database|pipeline)\b",
    re.IGNORECASE,
)


def _is_project_task(task: str) -> bool:
    if _CONVERSATIONAL.search(task):
        return False
    return bool(_PROJECT_HINTS.search(task)) and len(task.split()) > 6
