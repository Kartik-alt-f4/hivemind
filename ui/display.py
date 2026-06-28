"""
HiveMind Terminal UI
A live-updating terminal display showing the agent tree as it executes.
"""
import asyncio
import time
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns
from rich.align import Align
from rich import box
from agents.node import AgentNode, AgentStatus
from core.providers import get_pool

console = Console()

STATUS_STYLE = {
    AgentStatus.PENDING:  ("○", "dim white"),
    AgentStatus.PLANNING: ("◈", "yellow"),
    AgentStatus.RUNNING:  ("◉", "cyan"),
    AgentStatus.MERGING:  ("⟳", "magenta"),
    AgentStatus.DONE:     ("●", "green"),
    AgentStatus.ERROR:    ("✗", "red"),
}

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def _spinner(t: float) -> str:
    return SPINNER_FRAMES[int(t * 8) % len(SPINNER_FRAMES)]


def _build_tree(node: AgentNode, tree: Tree, tick: float):
    icon, style = STATUS_STYLE[node.status]

    # Spinner for active states
    if node.status in (AgentStatus.PLANNING, AgentStatus.RUNNING, AgentStatus.MERGING):
        spin = _spinner(tick)
        label_parts = Text()
        label_parts.append(f"{spin} ", style="bold yellow")
        label_parts.append(f"[{node.agent_id}] ", style="dim")
        label_parts.append(node.status.value.upper() + " ", style=style)
        label_parts.append(node.task[:60] + ("…" if len(node.task) > 60 else ""), style="white")
        label_parts.append(f"  {node.elapsed():.1f}s", style="dim")
    else:
        label_parts = Text()
        label_parts.append(f"{icon} ", style=f"bold {style}")
        label_parts.append(f"[{node.agent_id}] ", style="dim")
        task_display = node.task[:60] + ("…" if len(node.task) > 60 else "")
        label_parts.append(task_display, style=style if node.status == AgentStatus.DONE else "white")
        if node.ended_at:
            label_parts.append(f"  {node.elapsed():.1f}s", style="dim green")

    branch = tree.add(label_parts)
    for child in node.children:
        _build_tree(child, branch, tick)


def _build_stats_table(root: AgentNode) -> Table:
    all_nodes = root.all_nodes()
    counts = {s: 0 for s in AgentStatus}
    for n in all_nodes:
        counts[n.status] += 1

    pool = get_pool()
    provider_stats = pool.stats()

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column(style="dim")
    table.add_column()

    table.add_row("agents", str(len(all_nodes)))
    table.add_row("done", Text(str(counts[AgentStatus.DONE]), style="green"))
    table.add_row("active", Text(str(
        counts[AgentStatus.RUNNING] + counts[AgentStatus.PLANNING] + counts[AgentStatus.MERGING]
    ), style="cyan"))
    table.add_row("errors", Text(str(counts[AgentStatus.ERROR]), style="red" if counts[AgentStatus.ERROR] else "dim"))
    table.add_row("", "")
    for ps in provider_stats:
        table.add_row(
            ps["name"].lower(),
            f"{ps['calls']} calls / {ps['errors']} err"
        )
    return table


def build_layout(root: AgentNode, task: str, tick: float, done: bool = False) -> Panel:
    # Header
    elapsed = root.elapsed()
    status_text = "✓ Complete" if done else f"{_spinner(tick)} Running"
    status_style = "bold green" if done else "bold yellow"

    header = Text()
    header.append("⬡ HIVEMIND ", style="bold cyan")
    header.append("agent cluster  ", style="dim")
    header.append(status_text, style=status_style)
    header.append(f"  {elapsed:.1f}s", style="dim")

    # Agent tree
    tree = Tree(Text(f'"{task}"', style="bold white"))
    for child in root.children:
        _build_tree(child, tree, tick)
    # If root has no children yet (planning phase), show root itself
    if not root.children:
        _build_tree(root, tree, tick)

    # Stats sidebar
    stats = _build_stats_table(root)

    content = Columns([tree, stats], expand=True)

    return Panel(
        content,
        title=header,
        border_style="cyan" if not done else "green",
        padding=(1, 2),
    )


async def run_with_ui(root: AgentNode, task: str, semaphore: asyncio.Semaphore) -> str | None:
    """Run the agent tree and display a live UI. Returns clarification string or None."""
    tick = 0.0
    done = False

    def on_update(agent):
        pass  # Live display handles re-render on its own tick

    root.on_update = on_update

    with Live(
        build_layout(root, task, tick),
        console=console,
        refresh_per_second=8,
        vertical_overflow="visible",
    ) as live:
        agent_task = asyncio.create_task(root.run(semaphore))

        while not agent_task.done():
            tick += 0.125
            live.update(build_layout(root, task, tick))
            await asyncio.sleep(0.125)

        done = True
        live.update(build_layout(root, task, tick, done=True))

    result = await agent_task
    return result  # None on success, clarification string if vague
