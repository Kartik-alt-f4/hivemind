#!/usr/bin/env python3
"""
HiveMind — Recursive Multi-Agent CLI (v2)

Standalone entry point that runs the agent cluster directly (no server).
For the widget/server interface use: widget_server.py + hm.py
"""
import asyncio
import os
import sys
import argparse
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.text import Text

console = Console()


def print_banner():
    console.print(Panel(
        Text.assemble(
            ("⬡ HIVEMIND v2\n", "bold cyan"),
            ("Recursive Agent Cluster\n", "dim white"),
            ("─────────────────────────────\n", "dim"),
            ("orchestrator → file_owners → workers → audit → merge\n", "dim white"),
            ("Downward-light · Upward-heavy · Output-shape-aware", "dim white"),
        ),
        border_style="cyan",
        padding=(1, 4),
    ))


def ensure_env():
    env_path = Path(".env")
    example_path = Path(".env.example")
    if not env_path.exists():
        if example_path.exists():
            console.print(
                "[yellow]⚠  No .env found. Copy .env.example to .env and add your API keys.[/yellow]"
            )
            console.print("   [dim]cp .env.example .env && nano .env[/dim]\n")
        else:
            console.print("[red]No .env or .env.example found. Run from the hivemind/ directory.[/red]")
        sys.exit(1)


async def run_task(task: str, max_agents: int):
    from agents.node import AgentNode, AgentRole
    import agents.node as _node_module
    from core.providers import get_pool

    _node_module._agent_registry.clear()

    pool = get_pool()
    console.print(
        f"[dim]Providers: {', '.join(f'{p.name}({p.model_class.value})' for p in pool.tiers)}[/dim]\n"
    )

    semaphore = asyncio.Semaphore(max_agents)
    root = AgentNode(task=task, depth=0, role=AgentRole.ORCHESTRATOR)

    await root.run(semaphore)

    # Stats
    pool = get_pool()
    console.print()
    for ps in pool.stats():
        if ps["calls"] or ps["errors"]:
            console.print(
                f"[dim]  {ps['name'].lower():<18} {ps['calls']} calls  {ps['errors']} errors[/dim]"
            )

    all_nodes = root.all_nodes()
    console.print(
        f"[dim]  Total agents: {len(all_nodes)}  |  Time: {root.elapsed():.1f}s[/dim]\n"
    )

    # Result
    if root.error == "needs_clarification":
        console.print(Panel(
            root.result,
            title="[bold yellow]⚠ Needs Clarification[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        ))
    elif root.result.startswith("[ERROR]"):
        console.print(Panel(
            root.result,
            title="[bold red]✗ Error[/bold red]",
            border_style="red",
            padding=(1, 2),
        ))
    else:
        console.print(Panel(
            root.result,
            title="[bold green]✓ Result[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))

    if root.files_written:
        console.print("[dim]Files written:[/dim]")
        for f in root.files_written:
            console.print(f"  [green]✓[/green] {f}")


def main():
    print_banner()
    ensure_env()

    parser = argparse.ArgumentParser(
        prog="hivemind",
        description="HiveMind v2 — recursive multi-agent task solver",
    )
    parser.add_argument("task", nargs="?", help="Task to solve (omit for interactive)")
    parser.add_argument(
        "--max-agents", type=int,
        default=int(os.getenv("MAX_PARALLEL_AGENTS", "8")),
        help="Max concurrent agents (default: 8)",
    )
    parser.add_argument("--interactive", "-i", action="store_true")
    args = parser.parse_args()

    async def loop():
        first = True
        while True:
            if args.task and first:
                task = args.task
            else:
                console.print()
                task = Prompt.ask("[bold cyan]⬡[/bold cyan] Task")
                if not task.strip():
                    break
            first = False
            await run_task(task=task.strip(), max_agents=args.max_agents)
            if not args.interactive:
                break

    asyncio.run(loop())


if __name__ == "__main__":
    main()
