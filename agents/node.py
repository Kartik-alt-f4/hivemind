"""
Agent Node — recursive unit of the HiveMind cluster.
Markers: ##SPLIT##, ##SOLVE##, ##CLARIFY##
Project tasks get a shared workspace .md file all agents read/write.
"""
import asyncio
import re
import time
import uuid
import pathlib
import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
from core.llm import chat

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


# ── Workspace helpers ─────────────────────────────────────────────────────────

def _workspace_path(root_task: str) -> pathlib.Path:
    slug = re.sub(r"[^a-z0-9]+", "_", root_task.lower())[:48].strip("_")
    ws = pathlib.Path("hivemind_workspace")
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
    parent_id: Optional[str] = None
    agent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:6])
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

    def __post_init__(self):
        _agent_registry[self.agent_id] = self
        if not self.root_task:
            self.root_task = self.task
        if self.depth == 0:
            self.is_project = _is_project_task(self.task)
            if self.is_project:
                self.workspace_path = _workspace_path(self.root_task)

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

                # Write result to workspace if project task
                if self.workspace_path and answer:
                    label = f"Agent {self.agent_id} (depth {self.depth})"
                    _append_workspace(self.workspace_path, label, answer)

                _debug_log(f"[{self.agent_id}] SOLVED depth={self.depth}")
                self.result = answer
                self.status = AgentStatus.DONE
                self._emit()

        except Exception as e:
            self.status = AgentStatus.ERROR
            self.error = str(e)
            self.result = f"[ERROR] {e}"
            _debug_log(f"[{self.agent_id}] ERROR: {e}")
            self._emit()
        finally:
            self.ended_at = time.time()

        return None

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
        )

    async def _plan(self) -> str:
        if self.depth >= self.max_depth:
            return await self._force_solve()

        budget = self._budget()

        # Root node: initialise workspace and detect task type
        if self.depth == 0 and self.is_project and self.workspace_path:
            _write_workspace(
                self.workspace_path,
                f"# HiveMind Workspace\n\n**Task:** {self.root_task}\n"
                f"**Started:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            )

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

        if self.depth == 0:
            system = (
                "You are HiveMind — a recursive multi-agent AI cluster.\n"
                "You are the ROOT orchestrator. Split complex tasks into independent workstreams.\n"
                "For simple/conversational tasks, solve directly.\n"
                + (f"This is a PROJECT task. A workspace file has been created for agents to share work.\n" if self.is_project else "")
                + budget_note
            )
        else:
            system = (
                f"You are a HiveMind agent (depth {self.depth}/{self.max_depth}).\n"
                + root_ctx
                + "Solve atomically if possible. Split only into 2-3 truly independent parts.\n"
                + ("depth>=3: always ##SOLVE##.\n" if self.depth >= 3 else "")
                + budget_note
            )

        system += (
            "\n\nRespond with exactly one:\n"
            "##SPLIT##\n- [subtask 1]\n- [subtask 2]\n\n"
            "##SOLVE##\n[answer]\n\n"
            + ("##CLARIFY##\n[one sentence: what is missing]\n" if self.depth == 0 else "")
            + "Nothing else."
        )

        user_msg = f"Task: {self.task}{ws_context}"

        response = await chat(
            [{"role": "user", "content": user_msg}],
            system=system,
            temperature=0.3,
            max_tokens=600,
            depth=self.depth,
        )

        _debug_log(f"[{self.agent_id}] depth={self.depth} budget={budget} | {response[:120].strip()}")
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
            _debug_log(f"[{self.agent_id}] BUDGET EXHAUSTED — solving directly")
            self.result = await self._force_solve()
            self.status = AgentStatus.DONE
            self._emit()
            return

        if len(subtasks) > budget:
            _debug_log(f"[{self.agent_id}] BUDGET TRIM {len(subtasks)}→{budget}")
            subtasks = subtasks[:budget]

        self.children = [
            AgentNode(
                task=st,
                depth=self.depth + 1,
                parent_id=self.agent_id,
                root_task=self.root_task,
                max_depth=self.max_depth,
                min_complexity=self.min_complexity,
                on_update=self.on_update,
                is_project=self.is_project,
                workspace_path=self.workspace_path,
            )
            for st in subtasks
        ]
        _debug_log(f"[{self.agent_id}] SPLIT into {len(self.children)} | registry={len(_agent_registry)}/{MAX_TOTAL_AGENTS}")
        self._emit()

        sequential = self._subtasks_have_dependencies(subtasks)
        if sequential:
            _debug_log(f"[{self.agent_id}] SEQUENTIAL mode")
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
        _debug_log(f"[{self.agent_id}] RAY dispatched {len(refs)} remote tasks")

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
            f"### {c.task}\n{c.result}" for c in self.children
        )

        ws_context = ""
        if self.workspace_path:
            existing = _read_workspace(self.workspace_path)
            if existing:
                ws_context = f"\n\nWorkspace:\n{existing[-2000:]}"

        system = (
            "You are HiveMind's integration agent. Synthesise parallel sub-agent outputs "
            "into one coherent, complete response. Remove redundancy. Resolve contradictions."
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
            max_tokens=3000,
        )

        # Root node finalises workspace
        if self.depth == 0 and self.workspace_path:
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
