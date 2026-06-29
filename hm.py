#!/usr/bin/env python3
"""
hm — HiveMind terminal CLI
Usage:
  hm "your task"          one-shot, streams result
  hm                      interactive REPL (Ctrl+C or 'exit' to quit)
  hm --status             show server health + key usage
"""
import sys
import os
import json
import time
import signal
import subprocess
import threading
import argparse
import uuid
import getpass
from pathlib import Path

# ── Optional rich ─────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich import print as rprint
    _RICH = True
except ImportError:
    _RICH = False

# ── Live agent tree visualizer ────────────────────────────────────────────────
if _RICH:
    class AgentTree:
        """
        Tracks node_update events and renders a growing tree of agents.
        States cycle through colours; completion 'flows' back up to the root.
        """
        # State → (icon, Rich color)
        _STYLE = {
            "pending":  ("○", "dim white"),
            "planning": ("◎", "bold cyan"),
            "running":  ("●", "bold yellow"),
            "merging":  ("◈", "bold magenta"),
            "done":     ("✓", "bold green"),
            "error":    ("✗", "bold red"),
        }
        # Depth → indent colour gradient (dark → bright)
        _DEPTH_COLOR = ["white", "cyan", "green", "yellow", "magenta", "blue"]

        def __init__(self):
            self._nodes: dict[str, dict] = {}   # task_id → node dict
            self._lock = threading.Lock()
            self._tick = 0  # animation frame counter

        def update(self, msg: dict):
            with self._lock:
                tid = msg["task_id"]
                if tid not in self._nodes:
                    self._nodes[tid] = msg.copy()
                else:
                    self._nodes[tid].update(msg)

        def _sorted_ids(self) -> list[str]:
            """Return task IDs in tree order (depth-first by dotted ID)."""
            def sort_key(tid: str):
                parts = tid.split(".")
                # "task" sorts first, then numerically by each segment
                if parts[0] == "task" and len(parts) == 1:
                    return []
                return [int(p) if p.isdigit() else 0 for p in parts[1:]]
            return sorted(self._nodes.keys(), key=sort_key)

        def render(self) -> "Text":
            self._tick += 1
            spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            frame = spinner_frames[self._tick % len(spinner_frames)]

            with self._lock:
                if not self._nodes:
                    return Text("  waiting for agents…", style="dim")

                lines = Text()
                ids = self._sorted_ids()
                for tid in ids:
                    node = self._nodes[tid]
                    depth = node.get("depth", 0)
                    state = node.get("state", "pending")
                    task  = node.get("task", "")
                    elapsed = node.get("elapsed", 0)

                    icon, style = self._STYLE.get(state, ("?", "white"))
                    depth_color = self._DEPTH_COLOR[min(depth, len(self._DEPTH_COLOR) - 1)]

                    # Tree indent: connector lines
                    if depth == 0:
                        prefix = ""
                    else:
                        prefix = "  " * (depth - 1) + "└─ "

                    # Animate spinner on active nodes
                    if state in ("running", "planning", "merging"):
                        icon = frame if state == "running" else icon

                    # Truncate long task names
                    max_task = max(20, 60 - len(prefix))
                    short_task = task[:max_task] + ("…" if len(task) > max_task else "")

                    elapsed_str = f" [{elapsed}s]" if elapsed > 0 else ""

                    line = Text()
                    line.append(prefix, style=f"dim {depth_color}")
                    line.append(f"{icon} ", style=style)
                    line.append(f"[{tid}]", style=f"bold {depth_color}")
                    line.append(f" {short_task}", style="white" if state != "done" else "dim white")
                    if elapsed_str:
                        line.append(elapsed_str, style="dim")
                    line.append("\n")
                    lines.append_text(line)

                return lines

        def panel(self) -> "Panel":
            total = len(self._nodes)
            done  = sum(1 for n in self._nodes.values() if n.get("state") == "done")
            title = f"[bold cyan]HiveMind Cluster[/bold cyan] [dim]({done}/{total} agents)[/dim]"
            return Panel(self.render(), title=title, border_style="cyan", padding=(0, 1))


SERVER_URL  = os.environ.get("HIVEMIND_SERVER", "http://localhost:7779")
SERVER_BIN  = Path(__file__).parent / "widget_server.py"
VENV_PYTHON = Path(__file__).parent / ".venv" / "bin" / "python3"
HISTORY_FILE = Path.home() / ".local" / "share" / "hivemind" / "history.json"

console = Console() if _RICH else None


# ── Server management ─────────────────────────────────────────────────────────

def _server_running() -> bool:
    try:
        import httpx
        r = httpx.get(f"{SERVER_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def ensure_server() -> bool:
    if _server_running():
        return True
    python = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    _print_dim(f"Starting server ({python} {SERVER_BIN})…")
    subprocess.Popen(
        [python, str(SERVER_BIN)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(20):
        time.sleep(0.5)
        if _server_running():
            return True
    return False


# ── History ───────────────────────────────────────────────────────────────────

def _load_history() -> list[dict]:
    try:
        if HISTORY_FILE.exists():
            return json.loads(HISTORY_FILE.read_text())
    except Exception:
        pass
    return []


def _save_history(history: list[dict]):
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Keep last 500 entries
        HISTORY_FILE.write_text(json.dumps(history[-500:], indent=2))
    except Exception:
        pass


# ── Output helpers ────────────────────────────────────────────────────────────

def _print_dim(msg: str):
    if _RICH:
        console.print(f"[dim]{msg}[/dim]")
    else:
        print(f"\033[2m{msg}\033[0m", file=sys.stderr)


def _print_error(msg: str):
    if _RICH:
        from rich.markup import escape
        console.print(f"[bold red]✗[/bold red] {escape(msg)}")
    else:
        print(f"✗ {msg}", file=sys.stderr)


def _print_result(text: str):
    if _RICH:
        console.print(Markdown(text))
    else:
        print(text)


def _print_stats(agents: int, per_key: dict, tokens: int, key_health: dict):
    # Show per-key usage: gemini1:2  gemini2:1  groq1:3
    # Rate-limited keys get a ! suffix
    limited: set[str] = set()
    for keys in key_health.values():
        for k in keys:
            if k.get("rate_limited") and k.get("label"):
                limited.add(k["label"])

    parts = [
        f"{label}:{calls}{'!' if label in limited else ''}"
        for label, calls in sorted(per_key.items())
    ]
    summary = "  ".join(parts) if parts else "—"
    msg = f"⬡ {agents} agent{'s' if agents != 1 else ''}  {summary}  ~{tokens} tok"
    if _RICH:
        console.print(f"[dim]{msg}[/dim]")
    else:
        print(f"\033[2m{msg}\033[0m")


# ── Core streaming task ───────────────────────────────────────────────────────

def run_task(task: str, history: list[dict]) -> str | None:
    """
    Stream a task to the server. Returns the result text, or None on error.
    Shows a live growing tree of agents while work is in progress.
    """
    import httpx
    from rich.markup import escape

    req_id = uuid.uuid4().hex
    payload = {"task": task, "cwd": os.getcwd(), "request_id": req_id}

    result_text:   list[str]  = []
    error_text:    list[str]  = []
    stats_data:    list[dict] = []
    files_written: list[str]  = []
    # Shell/sudo events buffered while Live is active (printed after it exits)
    deferred_shell: list[dict] = []

    tree = AgentTree() if _RICH else None

    # stream_done signals the render loop to stop; stream_exc carries any error
    stream_done  = threading.Event()
    stream_exc:  list[Exception] = []
    sudo_pending: list[dict]     = []   # sudo events that need interactive handling

    def _stream_thread():
        """Runs in background thread — reads SSE stream, updates tree state."""
        try:
            with httpx.Client(timeout=None) as client:
                with client.stream("POST", f"{SERVER_URL}/chat", json=payload) as resp:
                    resp.raise_for_status()
                    for raw_line in resp.iter_lines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        try:
                            msg = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        mtype = msg.get("type")
                        if mtype == "node_update":
                            if tree:
                                tree.update(msg)
                        elif mtype == "result":
                            result_text.append(msg.get("text", ""))
                        elif mtype == "error":
                            error_text.append(msg.get("text", ""))
                        elif mtype == "usage":
                            stats_data.append(msg)
                        elif mtype == "files":
                            files_written.extend(msg.get("paths", []))
                        elif mtype == "shell_run":
                            deferred_shell.append(msg)
                        elif mtype == "sudo_prompt":
                            sudo_pending.append(msg)
        except Exception as e:
            stream_exc.append(e)
        finally:
            stream_done.set()

    t = threading.Thread(target=_stream_thread, daemon=True)
    t.start()

    try:
        if _RICH:
            with Live(tree.panel(), console=console, refresh_per_second=10,
                      vertical_overflow="visible", transient=True) as live:
                while not stream_done.is_set():
                    if sudo_pending:
                        msg = sudo_pending.pop(0)
                        live.stop()
                        cmd_str = msg.get("cmd", "")
                        sudo_req_id = msg.get("request_id", req_id)
                        console.print(f"\n[yellow]⚠ sudo:[/yellow] [dim]{escape(cmd_str)}[/dim]")
                        try:
                            pw = getpass.getpass("  Password: ")
                        except (KeyboardInterrupt, EOFError):
                            pw = ""
                        try:
                            httpx.post(f"{SERVER_URL}/sudo-input",
                                       json={"request_id": sudo_req_id, "password": pw},
                                       timeout=5)
                        except Exception:
                            pass
                        live.start()
                    live.update(tree.panel())
                    time.sleep(0.10)
                # Hold final state for 0.5s so user can see the completed tree
                live.update(tree.panel())
                time.sleep(0.5)
        else:
            stream_done.wait()
    except KeyboardInterrupt:
        stream_done.set()
        print()
        return None
    except Exception as e:
        stream_done.set()
        _print_error(str(e))
        return None

    t.join(timeout=5)

    if stream_exc:
        _print_error(str(stream_exc[0]))
        return None

    # Always print the final tree state after run completes
    if _RICH and tree and tree._nodes:
        console.print(tree.panel())

    # Print shell output after the live panel has closed
    if deferred_shell and _RICH:
        for ev in deferred_shell:
            console.print(f"\n[dim]$ {escape(ev.get('cmd', ''))}[/dim]")
            console.print(f"[dim]{escape(ev.get('output', ''))}[/dim]")

    if error_text:
        _print_error(error_text[0])
        return None

    result = result_text[0] if result_text else ""
    _print_result(result)

    if files_written:
        if _RICH:
            console.print("\n[bold green]Files written:[/bold green]")
            for f in files_written:
                console.print(f"  [green]✓[/green] {f}")
        else:
            print("\nFiles written:")
            for f in files_written:
                print(f"  ✓ {f}")

    if stats_data:
        s = stats_data[0]
        _print_stats(
            agents=s.get("agents", 0),
            per_key=s.get("per_key_calls", {}),
            tokens=s.get("tokens_this_msg", 0),
            key_health=s.get("key_health", {}),
        )

    return result or None


# ── Status command ────────────────────────────────────────────────────────────

def show_status():
    import httpx
    try:
        r = httpx.get(f"{SERVER_URL}/health", timeout=3)
        data = r.json()
    except Exception as e:
        _print_error(f"Server not reachable: {e}")
        return

    if _RICH:
        status_icon = "[green]●[/green]" if data.get("ok") else "[red]○[/red]"
        console.print(f"\n{status_icon} HiveMind server  {SERVER_URL}\n")

        tokens = data.get("tokens", {})
        console.print(
            f"  [dim]Tokens today:[/dim]  {tokens.get('day_total', 0):,}  "
            f"[dim]session:[/dim] {tokens.get('session_total', 0):,}"
        )

        by_calls = tokens.get("by_provider_calls", {})
        if by_calls:
            console.print(f"  [dim]API calls today:[/dim]  " +
                          "  ".join(f"{k}:{v}" for k, v in by_calls.items()))

        providers = data.get("providers", [])
        if providers:
            t = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
            t.add_column("Provider")
            t.add_column("Model", style="dim")
            t.add_column("Calls", justify="right")
            t.add_column("Keys", justify="right")
            for p in providers:
                t.add_row(p, "", "", "")
            console.print()
            console.print(t)
        print()
    else:
        ok = "UP" if data.get("ok") else "DOWN"
        print(f"Server: {ok}  {SERVER_URL}")
        tokens = data.get("tokens", {})
        print(f"Tokens today: {tokens.get('day_total', 0):,}")


# ── REPL ──────────────────────────────────────────────────────────────────────

PROMPT_NORMAL = "\033[1;35m❯\033[0m "
PROMPT_BUSY   = "\033[2m…\033[0m "

def repl():
    history = _load_history()
    if _RICH:
        console.print("[bold]HiveMind[/bold] [dim]— Ctrl+C or 'exit' to quit[/dim]\n")
    else:
        print("HiveMind — Ctrl+C or 'exit' to quit\n")

    while True:
        try:
            task = input(PROMPT_NORMAL).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not task:
            continue
        if task.lower() in ("exit", "quit", "q"):
            break
        if task.lower() in ("status", "/status"):
            show_status()
            continue

        history.append({"role": "user", "text": task, "ts": time.time()})
        result = run_task(task, history)
        if result:
            history.append({"role": "assistant", "text": result, "ts": time.time()})
        _save_history(history)
        print()

    _save_history(history)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="hm",
        description="HiveMind CLI — multi-agent AI in your terminal",
        add_help=True,
    )
    parser.add_argument("task", nargs="*", help="Task to run (omit for REPL)")
    parser.add_argument("--status", action="store_true", help="Show server health")
    parser.add_argument("--no-server", action="store_true",
                        help="Don't auto-launch server if not running")
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if not args.no_server:
        if not ensure_server():
            _print_error("Could not start server. Run widget_server.py manually.")
            sys.exit(1)

    if args.task:
        task = " ".join(args.task)
        history = _load_history()
        result = run_task(task, history)
        if result:
            history.append({"role": "user",      "text": task,   "ts": time.time()})
            history.append({"role": "assistant",  "text": result, "ts": time.time()})
            _save_history(history)
        sys.exit(0 if result else 1)
    else:
        repl()


if __name__ == "__main__":
    main()
