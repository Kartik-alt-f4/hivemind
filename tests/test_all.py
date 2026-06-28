#!/usr/bin/env python3
"""
Master provider test runner.
Runs each provider's test only if that provider is configured in .env.
Prints a summary table at the end.

Usage:
    python tests/test_all.py
"""
import asyncio
import os
import time
from dotenv import load_dotenv

load_dotenv()

# ANSI colors
GRN = "\033[32m"; RED = "\033[31m"; YLW = "\033[33m"; DIM = "\033[2m"; CYN = "\033[36m"; RST = "\033[0m"; BLD = "\033[1m"

def _is_configured(prefix: str) -> bool:
    keys = os.getenv(f"PROVIDER_{prefix}_KEYS", "").strip()
    url  = os.getenv(f"PROVIDER_{prefix}_BASE_URL", "").strip()
    return bool(keys and url)


async def run_provider(name: str, prefix: str, module_path: str) -> dict:
    if not _is_configured(prefix):
        print(f"\n{DIM}── {name}: not configured in .env — skipping{RST}")
        return {"provider": name, "ok": None, "latency": None, "model": os.getenv(f"PROVIDER_{prefix}_MODEL", "—")}

    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(module_path, module_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    t0 = time.time()
    result = await mod.main()
    elapsed = time.time() - t0

    if result and result.get("latency") is None:
        result["latency"] = elapsed
    return result or {"provider": name, "ok": False, "latency": elapsed, "model": "—"}


async def main():
    print(f"\n{CYN}{'═'*56}")
    print(f"  HIVEMIND — ALL PROVIDERS TEST")
    print(f"{'═'*56}{RST}")

    import pathlib
    tests_dir = pathlib.Path(__file__).parent

    providers = [
        ("Gemini",      "GEMINI",      str(tests_dir / "test_gemini.py")),
        ("Groq",        "GROQ",        str(tests_dir / "test_groq.py")),
        ("OpenRouter",  "OPENROUTER",  str(tests_dir / "test_openrouter.py")),
        ("Ollama",      "OLLAMA",      str(tests_dir / "test_ollama.py")),
    ]

    results = []
    for name, prefix, path in providers:
        r = await run_provider(name, prefix, path)
        results.append(r)

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{YLW}{'═'*56}")
    print(f"  SUMMARY")
    print(f"{'═'*56}{RST}")
    print(f"  {'Provider':<14} {'Status':<8} {'Latency':<10} {'Model'}")
    print(f"  {'─'*14} {'─'*8} {'─'*10} {'─'*20}")

    configured_count = 0
    working_count    = 0

    for r in results:
        name     = r.get("provider", "?")
        ok_val   = r.get("ok")
        latency  = r.get("latency")
        model    = r.get("model") or "—"

        if ok_val is None:
            status_str = f"{DIM}skip{RST}"
            lat_str    = f"{DIM}—{RST}"
            model_str  = f"{DIM}{model[:28]}{RST}"
        elif ok_val:
            status_str = f"{GRN}✓ OK{RST}"
            lat_str    = f"{GRN}{latency:.1f}s{RST}" if latency else f"{DIM}—{RST}"
            model_str  = model[:28]
            configured_count += 1
            working_count    += 1
        else:
            status_str = f"{RED}✗ FAIL{RST}"
            lat_str    = f"{DIM}—{RST}"
            model_str  = model[:28]
            configured_count += 1

        print(f"  {name:<14} {status_str:<17} {lat_str:<19} {model_str}")

    print()
    if configured_count == 0:
        print(f"  {RED}No providers configured. Copy .env.example to .env and add keys.{RST}")
    elif working_count == configured_count:
        print(f"  {GRN}{BLD}All {working_count} configured provider(s) working.{RST}")
    else:
        failed = configured_count - working_count
        print(f"  {GRN}{working_count} working{RST}  {RED}{failed} failed{RST}  — check output above for details")

    print(f"\n  {DIM}Tip: run individual tests for more detail, e.g. python tests/test_groq.py{RST}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
