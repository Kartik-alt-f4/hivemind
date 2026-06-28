"""
Agent Node - the recursive unit of the HiveMind tree.
Marker-based planning: ##SPLIT##, ##SOLVE##, ##CLARIFY##
Budget enforced BEFORE child instantiation to prevent explosions.
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


def _debug_log(msg: str):
    entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n"
    pathlib.Path("hivemind_debug.log").open("a").write(entry)
    import sys
    print(f"\033[2m{entry.strip()}\033[0m", file=sys.stderr)


class AgentStatus(Enum):
    PENDING   = "pending"
    PLANNING  = "planning"
    RUNNING   = "running"
    MERGING   = "merging"
    DONE      = "done"
    ERROR     = "error"


# Global registry — budget enforced here
_agent_registry: dict[str, "AgentNode"] = {}
MAX_TOTAL_AGENTS = 30


@dataclass
class AgentNode:
    task: str
    depth: int = 0
    parent_id: Optional[str] = None
    agent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:6])

    status: AgentStatus = AgentStatus.PENDING
    children: list["AgentNode"] = field(default_factory=list)
    result: str = ""
    error: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0

    max_depth: int = 6
    min_complexity: int = 3
    on_update: Optional[Callable] = None

    def __post_init__(self):
        _agent_registry[self.agent_id] = self

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
                # Clarify gate (root only)
                if self.depth == 0 and "##CLARIFY##" in response:
                    clarification = response.split("##CLARIFY##", 1)[1].strip()
                    self.status = AgentStatus.ERROR
                    self.error = "needs_clarification"
                    self.result = clarification
                    self._emit()
                    self.ended_at = time.time()
                    return clarification

                # Extract answer
                if "##SOLVE##" in response:
                    answer = response.split("##SOLVE##", 1)[1].strip()
                elif "##SPLIT##" in response:
                    # Wanted to split but no valid subtasks parsed — solve instead
                    answer = await self._force_solve()
                else:
                    answer = response.strip()
                    for label in ["##SPLIT##", "##CLARIFY##", "##SOLVE##"]:
                        if label in answer:
                            answer = answer.split(label, 1)[1].strip()
                            break

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
        system = "You are a focused AI agent. Answer the task directly and completely."
        return await chat(
            [{"role": "user", "content": f"Task: {self.task}\n\nSolve this completely."}],
            system=system, temperature=0.5, max_tokens=2048,
        )

    async def _plan(self) -> str:
        if self.depth >= self.max_depth:
            return await self._force_solve()

        budget = self._budget()
        budget_note = (
            f"\nBUDGET WARNING: Only {budget} agents left globally. Prefer ##SOLVE##."
            if budget < 8 else ""
        )
        clarify_option = (
            "\n##CLARIFY##\n[one sentence: what exactly is missing]\n"
            "(Only use if task is a single word/pronoun with zero context)\n"
            if self.depth == 0 else ""
        )

        system = (
            f"You are agent node depth={self.depth}/{self.max_depth} in a recursive multi-agent cluster.\n\n"
            + (
                "You are the ROOT NODE. Your job is to ORCHESTRATE, not to answer.\n"
                "Split this task into independent workstreams for sub-agents UNLESS it is\n"
                "genuinely a single atomic question (one fact, one calculation).\n"
                "When in doubt at depth=0: SPLIT.\n\n"
                if self.depth == 0 else
                "RULE: Split into 2-3 parts only if they are TRULY independent workstreams.\n"
                "RULE: depth >= 3 → always ##SOLVE## (you're deep enough).\n"
                "RULE: narrow/specific tasks → always ##SOLVE##.\n"
            )
            + f"{budget_note}\n\n"
            "OUTPUT — pick exactly one:\n\n"
            "##SPLIT##\n"
            "- [self-contained subtask 1 with full context]\n"
            "- [self-contained subtask 2 with full context]\n"
            "- [subtask 3 only if truly needed]\n\n"
            "##SOLVE##\n"
            "[your complete answer here]\n"
            + clarify_option +
            "\nOutput ONLY the marker and content. Nothing else."
        )

        response = await chat(
            [{"role": "user", "content": f"Task: {self.task}"}],
            system=system,
            temperature=0.3,
            max_tokens=800,
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

    async def _split_and_merge(self, subtasks: list[str], semaphore: asyncio.Semaphore):
        self.status = AgentStatus.RUNNING
        self._emit()

        # ── Budget check BEFORE instantiation ──────────────────────────────
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

        # ── Create children (counted here) ──────────────────────────────────
        self.children = [
            AgentNode(
                task=st,
                depth=self.depth + 1,
                parent_id=self.agent_id,
                max_depth=self.max_depth,
                min_complexity=self.min_complexity,
                on_update=self.on_update,
            )
            for st in subtasks
        ]
        _debug_log(f"[{self.agent_id}] SPLIT into {len(self.children)} | registry={len(_agent_registry)}/{MAX_TOTAL_AGENTS}")
        self._emit()

        await asyncio.gather(*[child.run(semaphore) for child in self.children])

        self.status = AgentStatus.MERGING
        self._emit()
        self.result = await self._merge()
        self.status = AgentStatus.DONE
        self._emit()

    async def _merge(self) -> str:
        child_results = "\n\n".join(
            f"--- Subtask: {c.task} ---\n{c.result}"
            for c in self.children
        )
        system = (
            "You are an integration agent. Combine results from parallel sub-agents "
            "into one coherent, well-structured response. Remove redundancy. "
            "Resolve contradictions. The whole must be better than the sum of parts."
        )
        return await chat(
            [{"role": "user", "content": (
                f"Original task: {self.task}\n\n"
                f"Sub-agent outputs:\n{child_results}\n\n"
                "Produce a unified, complete answer."
            )}],
            system=system,
            temperature=0.3,
            max_tokens=3000,
        )

    def all_nodes(self) -> list["AgentNode"]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.all_nodes())
        return nodes