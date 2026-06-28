#!/usr/bin/env python3
"""
HiveMind — Recursive Multi-Agent CLI
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
from rich.syntax import Syntax

console = Console()


def print_banner():
    console.print(Panel(
        Text.assemble(
            ("⬡ HIVEMIND\n", "bold cyan"),
            ("Recursive Agent Cluster\n", "dim white"),
            ("─────────────────────────────\n", "dim"),
            ("Any OpenAI-compatible backend · Dynamic task splitting\n", "dim white"),
            ("Key rotation · Parallel execution · Live tree view", "dim white"),
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
            console.print(f"   [dim]cp .env.example .env && nano .env[/dim]\n")
        else:
            console.print("[red]No .env or .env.example found. Make sure you're in the hivemind/ directory.[/red]")
        sys.exit(1)


async def run_task(task: str, max_depth: int, max_agents: int, min_complexity: int):
    from agents.node import AgentNode
    import agents.node as _node_module
    # Reset global agent state between runs
    _node_module._agent_registry = {}
    _node_module._agent_count = 0

    from ui.display import run_with_ui
    from core.providers import get_pool

    # Validate providers loaded
    pool = get_pool()
    console.print(
        f"[dim]Loaded {len(pool.providers)} provider(s): "
        f"{', '.join(p.name for p in pool.providers)}[/dim]\n"
    )

    semaphore = asyncio.Semaphore(max_agents)

    root = AgentNode(
        task=task,
        depth=0,
        max_depth=max_depth,
        min_complexity=min_complexity,
    )

    clarification = await run_with_ui(root, task, semaphore)

    # Show usage summary
    pool = get_pool()
    console.print()
    for ps in pool.stats():
        total = ps['calls'] + ps['errors']
        console.print(
            f"[dim]  {ps['name'].lower():<10} {ps['calls']} calls  {ps['errors']} errors  "
            f"({total} requests this session)[/dim]"
        )
    console.print(
        f"[dim]  Note: Gemini Flash Lite free tier ~1500 req/day · Groq free tier ~14400 req/day[/dim]\n"
    )

    # Print final result
    console.print()
    if clarification:
        console.print(Panel(
            clarification,
            title="[bold yellow]⚠ Needs Clarification[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        ))
    else:
        console.print(Panel(
            root.result,
            title="[bold green]✓ Final Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))

    # Show tree stats
    all_nodes = root.all_nodes()
    console.print(
        f"\n[dim]Total agents: {len(all_nodes)}  |  "
        f"Time: {root.elapsed():.1f}s  |  "
        f"API calls: {sum(p.calls for p in pool.providers)}[/dim]\n"
    )


def main():
    print_banner()
    ensure_env()

    parser = argparse.ArgumentParser(
        prog="hivemind",
        description="Recursive multi-agent task solver",
        add_help=True,
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Task to solve (if omitted, you'll be prompted)",
    )
    parser.add_argument(
        "--max-depth", type=int,
        default=int(os.getenv("MAX_DEPTH", "6")),
        help="Max recursion depth (default: 6)",
    )
    parser.add_argument(
        "--max-agents", type=int,
        default=int(os.getenv("MAX_PARALLEL_AGENTS", "8")),
        help="Max concurrent agents (default: 8)",
    )
    parser.add_argument(
        "--min-complexity", type=int,
        default=int(os.getenv("MIN_COMPLEXITY_TO_SPLIT", "3")),
        help="Complexity threshold to split (1-10, default: 3)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Keep asking for tasks after each run",
    )

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
            await run_task(
                task=task.strip(),
                max_depth=args.max_depth,
                max_agents=args.max_agents,
                min_complexity=args.min_complexity,
            )

            if not args.interactive:
                break

    asyncio.run(loop())


if __name__ == "__main__":
    main()