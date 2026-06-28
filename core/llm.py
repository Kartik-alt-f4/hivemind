"""
LLM Client - async wrapper around any OpenAI-compatible endpoint
with automatic key rotation and retry on rate-limit errors.
"""
import asyncio
import json
import httpx
from core.providers import get_pool, Provider


MAX_RETRIES = 4
RETRY_DELAY = 2.0  # seconds, doubles on each retry


async def chat(
    messages: list[dict],
    system: str = "",
    temperature: float = 0.4,
    max_tokens: int = 2048,
    provider: Provider | None = None,
) -> str:
    """
    Send a chat completion request. Returns the assistant's reply as a string.
    Rotates providers/keys automatically on 429 or connection errors.
    """
    pool = get_pool()
    delay = RETRY_DELAY
    last_error = "unknown"

    # Filter out providers with too many errors (likely misconfigured)
    ERROR_BLACKLIST_THRESHOLD = 5

    for attempt in range(MAX_RETRIES):
        # Prefer healthy providers
        healthy = [p for p in pool.providers if p.errors < ERROR_BLACKLIST_THRESHOLD]
        candidate_pool = healthy if healthy else pool.providers
        p = provider or next(iter(candidate_pool)) if len(candidate_pool) == 1 else (provider or pool.next_provider())
        key = p.next_key()

        payload = {
            "model": p.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": (
                [{"role": "system", "content": system}] + messages
                if system else messages
            ),
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{p.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

            if resp.status_code == 429:
                p.errors += 1
                await asyncio.sleep(delay)
                delay *= 2
                provider = None  # try a different provider next round
                continue

            if resp.status_code == 400:
                # Bad request from this provider (wrong model, unsupported param, etc.)
                # Log and failover to next provider rather than crashing
                p.errors += 1
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text[:200]
                last_error = f"400 from {p.name}: {detail}"
                provider = None  # force switch
                continue

            resp.raise_for_status()
            data = resp.json()
            p.calls += 1
            return data["choices"][0]["message"]["content"]

        except (httpx.ConnectError, httpx.TimeoutException, KeyError) as e:
            p.errors += 1
            last_error = str(e)
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts: {last_error}")
            await asyncio.sleep(delay)
            delay *= 2
            provider = None

    raise RuntimeError(f"LLM call failed: max retries exceeded. Last error: {last_error}")


async def chat_json(
    messages: list[dict],
    system: str = "",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> dict:
    """Like chat() but parses the response as JSON. Handles all common model formatting quirks."""
    raw = await chat(messages, system=system, temperature=temperature, max_tokens=max_tokens)
    clean = raw.strip()

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    if clean.startswith("```"):
        lines = clean.split("\n")
        # Remove first line (```json or ```) and last ``` line
        inner_lines = []
        for line in lines[1:]:
            if line.strip() == "```":
                break
            inner_lines.append(line)
        clean = "\n".join(inner_lines).strip()

    # If there's text before the first { or [, strip it
    for brace in ["{", "["]:
        idx = clean.find(brace)
        if idx > 0:
            clean = clean[idx:]
            break

    # Strip trailing text after the last } or ]
    for brace in ["}", "]"]:
        idx = clean.rfind(brace)
        if idx != -1 and idx < len(clean) - 1:
            clean = clean[:idx + 1]
            break

    import pathlib, datetime

    def _log(label, content):
        entry = f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] chat_json {label}:\n{content}\n{'─'*60}\n"
        pathlib.Path("hivemind_debug.log").open("a").write(entry)

    _log("RAW", raw)

    try:
        parsed = json.loads(clean)

        # Double-encoded: model returned a JSON string containing JSON
        if isinstance(parsed, str):
            _log("DOUBLE-ENCODED", parsed)
            parsed = json.loads(parsed)

        # Gemini wraps single object in array
        if isinstance(parsed, list):
            if len(parsed) == 1 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            elif all(isinstance(x, dict) for x in parsed):
                # Multiple dicts — merge them
                merged = {}
                for d in parsed:
                    merged.update(d)
                parsed = merged

        _log("PARSED", str(parsed))
        return parsed

    except json.JSONDecodeError as e:
        _log("PARSE FAILED", f"{e}\nCLEAN: {clean}")
        raise ValueError(f"Could not parse JSON from model response: {e}\nRaw: {raw[:300]}")