#!/usr/bin/env python3
"""
Standalone Ollama test — run this before using Ollama in the cluster.
Tests: localhost connectivity, model availability, auth (none needed), JSON compliance, speed.

Ollama uses /api/tags (not /models) for model listing.
No API key required — use any placeholder value.

Usage:
    python tests/test_ollama.py
"""
import asyncio
import httpx
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("PROVIDER_OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL    = os.getenv("PROVIDER_OLLAMA_MODEL", "llama3")
RAW_KEYS = os.getenv("PROVIDER_OLLAMA_KEYS", "ollama")
KEYS     = [k.strip() for k in RAW_KEYS.split(",") if k.strip()] or ["ollama"]

# Derive the Ollama root URL (strip /v1 suffix for native API endpoints)
OLLAMA_ROOT = BASE_URL.rstrip("/").removesuffix("/v1")

# ANSI colors
GRN = "\033[32m"; RED = "\033[31m"; YLW = "\033[33m"; DIM = "\033[2m"; RST = "\033[0m"
def ok(msg):   print(f"  {GRN}✓{RST} {msg}")
def err(msg):  print(f"  {RED}✗{RST} {msg}")
def info(msg): print(f"  {DIM}{msg}{RST}")
def hdr(msg):  print(f"\n{YLW}── {msg}{RST}")

# ── Helpers ───────────────────────────────────────────────────────────────────
async def call(key: str, messages: list, system: str = "", max_tokens: int = 200) -> dict:
    payload = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": ([{"role": "system", "content": system}] + messages) if system else messages,
    }
    t0 = time.time()
    async with httpx.AsyncClient(timeout=60.0) as client:  # longer timeout — local inference is slower
        resp = await client.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
        )
    elapsed = time.time() - t0
    return {"status": resp.status_code, "body": resp.text, "elapsed": elapsed, "resp": resp}

# ── Tests ─────────────────────────────────────────────────────────────────────
async def test_config():
    hdr("1. Config check")
    info(f"BASE_URL    : {BASE_URL}")
    info(f"OLLAMA_ROOT : {OLLAMA_ROOT}")
    info(f"MODEL       : {MODEL}")
    info(f"Auth        : not required (Ollama has no API keys)")
    ok("Config loaded")
    return True

async def test_connectivity():
    hdr("2. Localhost connectivity")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{OLLAMA_ROOT}/")
        ok(f"Ollama reachable at {OLLAMA_ROOT} (HTTP {r.status_code})")
        return True
    except httpx.ConnectError:
        err(f"Cannot connect to {OLLAMA_ROOT}")
        err("Is Ollama running? Start it with: ollama serve")
        info("Install Ollama: https://ollama.com/download")
        return False
    except Exception as e:
        err(f"Connection error: {e}")
        return False

async def test_auth(key: str, idx: int):
    hdr(f"3. Inference test — model: {MODEL}")
    result = await call(key, [{"role": "user", "content": "Say the word OK and nothing else."}])
    info(f"HTTP status : {result['status']}")
    info(f"Latency     : {result['elapsed']:.2f}s")

    if result["status"] == 404:
        err(f"404 — model '{MODEL}' not found locally")
        info(f"Pull it first: ollama pull {MODEL}")
        return False
    if result["status"] == 400:
        err(f"400 Bad Request")
        info(f"Response: {result['body'][:400]}")
        return False
    if result["status"] != 200:
        err(f"Unexpected status {result['status']}")
        info(f"Response: {result['body'][:300]}")
        return False

    try:
        data = json.loads(result["body"])
        content = data["choices"][0]["message"]["content"]
        ok(f"Inference OK — model replied: {repr(content.strip())}")
        return True
    except Exception as e:
        err(f"Parsed response but couldn't extract content: {e}")
        info(f"Raw: {result['body'][:300]}")
        return False

async def test_model_list(key: str):
    hdr("4. Locally available models (via /api/tags)")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(f"{OLLAMA_ROOT}/api/tags")
        if r.status_code != 200:
            err(f"/api/tags returned {r.status_code}")
            return
        data = r.json()
        models = sorted([m["name"] for m in data.get("models", [])])
        ok(f"{len(models)} model(s) pulled locally:")
        for m in models:
            marker = " ◄ CURRENT" if m.split(":")[0] == MODEL.split(":")[0] else ""
            info(f"    {m}{marker}")
        if not models:
            info("No models pulled yet — run: ollama pull llama3")
        elif MODEL not in models and MODEL + ":latest" not in models:
            err(f"'{MODEL}' not pulled — run: ollama pull {MODEL}")
    except Exception as e:
        err(f"Could not reach /api/tags: {e}")

async def test_json_compliance(key: str):
    hdr("5. JSON compliance test")
    prompt = 'Respond with ONLY this JSON, nothing else: {"status": "ok", "value": 42}'
    result = await call(
        key,
        [{"role": "user", "content": prompt}],
        system="You are a JSON API. Respond ONLY with valid JSON. No markdown, no explanation.",
        max_tokens=100,
    )
    info(f"Latency: {result['elapsed']:.2f}s")
    if result["status"] != 200:
        err(f"Request failed: {result['status']}")
        return

    data = json.loads(result["body"])
    raw = data["choices"][0]["message"]["content"].strip()
    info(f"Raw response: {repr(raw)}")

    clean = raw
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:]).rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(clean)
        ok(f"Valid JSON returned: {parsed}")
    except json.JSONDecodeError as e:
        err(f"Not valid JSON: {e}")
        info("Smaller local models may not follow JSON instructions reliably")

async def test_speed(key: str) -> float | None:
    hdr("6. Speed test (3 calls)")
    times = []
    for i in range(3):
        result = await call(key, [{"role": "user", "content": f"Count to {i+3} briefly."}], max_tokens=50)
        if result["status"] == 200:
            times.append(result["elapsed"])
            ok(f"Call {i+1}: {result['elapsed']:.2f}s")
        else:
            err(f"Call {i+1} failed: {result['status']}")
    if times:
        avg = sum(times) / len(times)
        info(f"Avg latency: {avg:.2f}s")
        if avg > 10:
            info("Tip: GPU acceleration speeds up Ollama significantly — check 'ollama ps'")
        return avg
    return None

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print(f"\n{YLW}{'═'*50}")
    print(f"  OLLAMA LOCAL MODEL TEST")
    print(f"{'═'*50}{RST}")

    await test_config()
    if not await test_connectivity():
        return {"provider": "Ollama", "ok": False, "latency": None, "model": MODEL}

    key = KEYS[0]
    auth_ok = await test_auth(key, 0)

    await test_model_list(key)

    avg_latency = None
    if auth_ok:
        await test_json_compliance(key)
        avg_latency = await test_speed(key)

    print(f"\n{YLW}── Summary{RST}")
    if auth_ok:
        ok(f"Ollama is working with model '{MODEL}'")
        info("No API keys or rate limits — fully local and free")
        info("For best performance: use a GPU and a quantized model (e.g. llama3:8b-q4)")
    else:
        err(f"Ollama is NOT working")
        info(f"1. Make sure Ollama is running: ollama serve")
        info(f"2. Pull your model: ollama pull {MODEL}")
        info(f"3. Check PROVIDER_OLLAMA_MODEL in .env matches a pulled model name")
    print()

    return {"provider": "Ollama", "ok": auth_ok, "latency": avg_latency, "model": MODEL}


if __name__ == "__main__":
    asyncio.run(main())
