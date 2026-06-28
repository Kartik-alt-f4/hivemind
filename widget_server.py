#!/usr/bin/env python3
"""
HiveMind Widget Server
FastAPI + WebSocket bridge between the AGS widget and the agent cluster.

Endpoints:
  GET  /health          → {"ok": true, "providers": [...]}
  POST /chat            → streams JSON lines over HTTP (SSE-style)
  WS   /ws/chat         → WebSocket streaming

Config: loads from ~/.config/hivemind/.env first, falls back to repo .env
Token tracking: in-memory, resets at EOD 23:59:59 via scheduled task
"""

import asyncio
import json
import os
import sys
import time
import datetime
from pathlib import Path
from typing import AsyncIterator

# ── Config path resolution ────────────────────────────────────────────────────
CONFIG_PATHS = [
    Path.home() / ".config/hivemind/.env",   # preferred — safe, user-owned
    Path(__file__).parent / ".env",           # fallback — repo location
]

_env_loaded = False
for _p in CONFIG_PATHS:
    if _p.exists():
        from dotenv import load_dotenv
        load_dotenv(_p)
        print(f"[server] loaded config from {_p}", file=sys.stderr)
        _env_loaded = True
        break

if not _env_loaded:
    print("[server] WARNING: no .env found in config paths", file=sys.stderr)

# Add repo root to path so we can import the cluster modules
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Token meter ───────────────────────────────────────────────────────────────
# Tracks estimated tokens per session and accumulated per-day total.
# Resets at EOD 23:59:59 local time. In-memory only — no persistence.
# Token counts are estimates: we count words*1.3 as a rough proxy since
# the OpenAI-compat APIs used here don't return usage in all responses.

def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))

class TokenMeter:
    def __init__(self):
        self._day: str = ""
        self.day_total: int = 0
        self.session_total: int = 0
        self._by_provider_tokens: dict[str, int] = {}   # token estimates per provider
        self._by_provider_calls: dict[str, int] = {}    # API call counts per provider (EOD)
        self._reset_day()

    def _today(self) -> str:
        return datetime.date.today().isoformat()

    def _reset_day(self):
        self._day = self._today()
        self.day_total = 0
        self._by_provider_tokens = {}
        self._by_provider_calls = {}

    def _check_rollover(self):
        if self._today() != self._day:
            self._reset_day()

    def add(self, text: str, provider: str = "unknown", calls: int = 0) -> int:
        self._check_rollover()
        n = _estimate_tokens(text)
        self.day_total += n
        self.session_total += n
        self._by_provider_tokens[provider] = self._by_provider_tokens.get(provider, 0) + n
        if calls:
            self._by_provider_calls[provider] = self._by_provider_calls.get(provider, 0) + calls
        return n

    def snapshot(self) -> dict:
        self._check_rollover()
        return {
            "day_total": self.day_total,
            "session_total": self.session_total,
            "by_provider": dict(self._by_provider_tokens),
            "by_provider_calls": dict(self._by_provider_calls),
            "reset_at": "23:59:59",
        }

METER = TokenMeter()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="HiveMind Widget Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health endpoint ───────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    try:
        from core.providers import get_pool
        pool = get_pool()
        providers = [p.name for p in pool.providers]
        if pool.root_provider:
            providers = ["ROOT"] + providers
        return {"ok": True, "providers": providers, "tokens": METER.snapshot()}
    except Exception as e:
        return {"ok": False, "error": str(e), "providers": [], "tokens": METER.snapshot()}

# ── Core task runner ──────────────────────────────────────────────────────────
async def _run_task(task: str) -> AsyncIterator[dict]:
    """Run a HiveMind task and yield status/result dicts as it progresses."""
    from agents.node import AgentNode, _agent_registry
    import agents.node as _node_module
    from core.providers import get_pool

    # Reset global agent state between runs
    _node_module._agent_registry.clear()

    pool = get_pool()

    yield {"type": "status", "text": "planning…"}

    # Snapshot call counts before the run so we can compute per-request deltas
    _pre = pool.stats()
    calls_before: dict[str, int] = {s["name"]: s["calls"] for s in _pre}
    # per-key: keyed by label (e.g. "gemini1")
    key_calls_before: dict[str, int] = {
        k["label"]: k["calls"]
        for s in _pre if "key_status" in s
        for k in s["key_status"]
    }

    root = AgentNode(task=task, depth=0, max_depth=5)
    semaphore = asyncio.Semaphore(8)

    # Run the agent tree; send keep-alive pings every 5s so the stream
    # doesn't stall in curl/GLib while waiting for the blocking work.
    run_task = asyncio.create_task(root.run(semaphore))
    ping_interval = 5.0
    while not run_task.done():
        done, _ = await asyncio.wait({run_task}, timeout=ping_interval)
        if not done:
            yield {"type": "status", "text": "working…"}

    try:
        run_task.result()
    except Exception as e:
        from core.providers import AllProvidersExhausted
        if isinstance(e, AllProvidersExhausted):
            yield {"type": "error", "text": str(e)}
        else:
            yield {"type": "error", "text": str(e)}
        return

    result_text = root.result or ""
    all_nodes = root.all_nodes()
    agent_count = len(all_nodes)
    full_text = task + " " + " ".join(n.result for n in all_nodes if n.result)

    # Per-request deltas (calls this run only, not cumulative)
    stats = pool.stats()
    per_provider_calls: dict[str, int] = {}
    for s in stats:
        delta = s["calls"] - calls_before.get(s["name"], 0)
        if delta > 0:
            per_provider_calls[s["name"]] = delta

    # Token estimate attributed to top provider by delta calls
    top = max(per_provider_calls.items(), key=lambda x: x[1], default=("unknown", 0))
    tokens_used = METER.add(full_text, top[0], calls=top[1])
    for name, delta in per_provider_calls.items():
        if name != top[0]:
            METER._by_provider_calls[name] = METER._by_provider_calls.get(name, 0) + delta

    # Per-key call deltas for this request (e.g. {"gemini1": 2, "groq1": 1})
    per_key_calls: dict[str, int] = {}
    for s in stats:
        for k in s.get("key_status", []):
            delta = k["calls"] - key_calls_before.get(k["label"], 0)
            if delta > 0:
                per_key_calls[k["label"]] = delta

    # Key health snapshot (full status for rate-limit display)
    key_health: dict[str, list[dict]] = {
        s["name"]: s["key_status"]
        for s in stats
        if "key_status" in s
    }

    yield {"type": "status", "text": f"done — {agent_count} agent(s)"}
    # Send usage BEFORE result so the widget has stats when the result bubble is added
    yield {
        "type": "usage",
        "agents": agent_count,
        "per_provider_calls": per_provider_calls,
        "per_key_calls": per_key_calls,
        "tokens_this_msg": tokens_used,
        "tokens": METER.snapshot(),
        "key_health": key_health,
    }
    yield {"type": "result", "text": result_text}

# ── POST /chat — HTTP streaming ───────────────────────────────────────────────
@app.post("/chat")
async def chat_http(body: dict):
    task = (body.get("task") or "").strip()
    if not task:
        return {"error": "task is required"}

    async def stream():
        async for msg in _run_task(task):
            yield json.dumps(msg) + "\n"

    return StreamingResponse(
        stream(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )

# ── WS /ws/chat — WebSocket streaming ────────────────────────────────────────
@app.websocket("/ws/chat")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            task = (data.get("task") or "").strip()
            if not task:
                await ws.send_json({"type": "error", "text": "task is required"})
                continue

            async for msg in _run_task(task):
                await ws.send_json(msg)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "error", "text": str(e)})
        except Exception:
            pass

# ── EOD token reset scheduler ─────────────────────────────────────────────────
async def _schedule_eod_reset():
    """Sleep until 23:59:59 local time, reset meter, repeat."""
    while True:
        now = datetime.datetime.now()
        tomorrow = now.replace(hour=23, minute=59, second=59, microsecond=0)
        if now >= tomorrow:
            tomorrow += datetime.timedelta(days=1)
        wait = (tomorrow - now).total_seconds()
        await asyncio.sleep(wait)
        METER._reset_day()
        print(f"[server] token meter reset at EOD", file=sys.stderr)

@app.on_event("startup")
async def startup():
    asyncio.create_task(_schedule_eod_reset())
    print("[server] HiveMind widget server ready on :7779", file=sys.stderr)

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7779, log_level="warning", http="h11")
