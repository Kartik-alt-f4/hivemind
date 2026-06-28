#!/usr/bin/env python3
"""
Standalone Gemini API test — run this before using Gemini in the cluster.
Tests: connectivity, auth, model validity, JSON compliance, speed.

Usage:
    python tests/test_gemini.py
"""
import asyncio
import httpx
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = os.getenv("PROVIDER_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
MODEL    = os.getenv("PROVIDER_GEMINI_MODEL", "gemini-2.0-flash-lite")
RAW_KEYS = os.getenv("PROVIDER_GEMINI_KEYS", "")
KEYS     = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]

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
    async with httpx.AsyncClient(timeout=30.0) as client:
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
    info(f"BASE_URL : {BASE_URL}")
    info(f"MODEL    : {MODEL}")
    info(f"KEYS     : {len(KEYS)} found")
    for i, k in enumerate(KEYS):
        info(f"  key[{i}] : {k[:8]}...{k[-4:]}")
    if not KEYS:
        err("No keys found — set PROVIDER_GEMINI_KEYS in .env")
        err("Get a free key: https://aistudio.google.com/app/apikey")
        return False
    ok("Config loaded")
    return True

async def test_connectivity():
    hdr("2. Connectivity")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get("https://generativelanguage.googleapis.com")
        ok(f"generativelanguage.googleapis.com reachable (HTTP {r.status_code})")
        return True
    except Exception as e:
        err(f"Cannot reach generativelanguage.googleapis.com: {e}")
        return False

async def test_auth(key: str, idx: int):
    hdr(f"3. Auth test — key[{idx}]")
    result = await call(key, [{"role": "user", "content": "Say the word OK and nothing else."}])
    info(f"HTTP status : {result['status']}")
    info(f"Latency     : {result['elapsed']:.2f}s")

    if result["status"] == 401:
        err("401 Unauthorized — invalid API key")
        info(f"Response: {result['body'][:300]}")
        return False
    if result["status"] == 403:
        err("403 Forbidden — key may be expired or quota exceeded")
        info(f"Response: {result['body'][:300]}")
        return False
    if result["status"] == 400:
        err(f"400 Bad Request — likely wrong model name: '{MODEL}'")
        info(f"Response: {result['body'][:400]}")
        info("Try: gemini-2.0-flash-lite, gemini-1.5-flash, gemini-1.5-flash-8b")
        return False
    if result["status"] == 429:
        err("429 Rate limited — free tier is 1500 req/day, 15 req/min")
        return False
    if result["status"] != 200:
        err(f"Unexpected status {result['status']}")
        info(f"Response: {result['body'][:300]}")
        return False

    try:
        data = json.loads(result["body"])
        content = data["choices"][0]["message"]["content"]
        ok(f"Auth OK — model replied: {repr(content.strip())}")
        return True
    except Exception as e:
        err(f"Parsed response but couldn't extract content: {e}")
        info(f"Raw: {result['body'][:300]}")
        return False

async def test_model_list(key: str):
    hdr("4. Available models")
    # Gemini's OpenAI-compat endpoint may not support /models — handle gracefully
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{BASE_URL}/models",
                headers={"Authorization": f"Bearer {key}"},
            )
        if r.status_code == 404:
            info("Model list endpoint not supported on Gemini OpenAI-compat endpoint (404) — skipping")
            return
        if r.status_code != 200:
            info(f"Models endpoint returned {r.status_code} — skipping")
            return
        data = r.json()
        # Gemini returns IDs like "models/gemini-2.0-flash-lite" — strip the prefix for comparison
        models = sorted([m["id"].removeprefix("models/") for m in data.get("data", [])])
        ok(f"{len(models)} models available:")
        for m in models:
            marker = " ◄ CURRENT" if m == MODEL else ""
            info(f"    {m}{marker}")
        if models and MODEL not in models:
            err(f"'{MODEL}' not in available models — update PROVIDER_GEMINI_MODEL in .env")
    except Exception as e:
        info(f"Could not list models: {e} — skipping")

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
        info("Gemini Flash Lite occasionally wraps JSON in markdown — cluster handles this automatically")

async def test_speed(key: str):
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
        info(f"Avg latency: {sum(times)/len(times):.2f}s")

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    print(f"\n{YLW}{'═'*50}")
    print(f"  GEMINI API TEST")
    print(f"{'═'*50}{RST}")

    if not await test_config():
        return {"provider": "Gemini", "ok": False, "latency": None, "model": MODEL}
    if not await test_connectivity():
        return {"provider": "Gemini", "ok": False, "latency": None, "model": MODEL}

    key = KEYS[0]
    auth_ok = await test_auth(key, 0)

    await test_model_list(key)

    avg_latency = None
    if auth_ok:
        await test_json_compliance(key)
        await test_speed(key)

        if len(KEYS) > 1:
            hdr(f"7. Testing remaining {len(KEYS)-1} key(s)")
            for i, k in enumerate(KEYS[1:], 1):
                await test_auth(k, i)

    print(f"\n{YLW}── Summary{RST}")
    if auth_ok:
        ok("Gemini is working — safe to use in the cluster")
        info(f"Recommended .env setting: PROVIDER_GEMINI_MODEL={MODEL}")
        info("Free tier: 1500 req/day, 15 req/min — add more keys to increase throughput")
    else:
        err("Gemini is NOT working — check your key at https://aistudio.google.com/app/apikey")
    print()

    return {"provider": "Gemini", "ok": auth_ok, "latency": avg_latency, "model": MODEL}


if __name__ == "__main__":
    asyncio.run(main())
