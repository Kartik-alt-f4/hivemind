"""
LLM Client - async wrapper around any OpenAI-compatible endpoint
with automatic key rotation and retry on rate-limit errors.
"""
import asyncio
import json
import httpx
from core.providers import get_pool, Provider, AllProvidersExhausted


MAX_RETRIES = 6
RETRY_DELAY = 1.0  # seconds, doubles on each retry


async def chat(
    messages: list[dict],
    system: str = "",
    temperature: float = 0.4,
    max_tokens: int = 2048,
    provider: Provider | None = None,
    depth: int = 0,
    prefer_root: bool = False,
    prefer_merge: bool = False,
) -> str:
    """
    Send a chat completion request. Returns the assistant's reply as a string.
    Picks the least-used available key across providers on each attempt.
    When depth==0 or prefer_root=True, uses root provider.
    When prefer_merge=True, uses the merge provider (heavy model for synthesis).
    Raises AllProvidersExhausted when every key is rate-limited.
    """
    import sys
    pool = get_pool()
    delay = RETRY_DELAY
    last_error = "unknown"

    # Tier routing: merge > root > pool
    if prefer_merge and pool.merge_provider is not None:
        use_provider = pool.merge_provider
    elif (depth == 0 or prefer_root) and pool.root_provider is not None:
        use_provider = pool.root_provider
    else:
        use_provider = None  # use pool rotation

    use_root = use_provider is not None
    root_tried = False

    for attempt in range(MAX_RETRIES):
        # Pick provider + key
        if use_root and not root_tried:
            p = use_provider
            root_tried = True
            api_key = p.best_key() or p.api_keys[0]
        elif provider is not None:
            p = provider
            api_key = p.best_key() or p.api_keys[0]
        else:
            if not pool.any_available():
                raise AllProvidersExhausted(
                    "All API keys are currently rate-limited. "
                    "Please wait a moment or add more keys to your .env."
                )
            p = pool.next_provider()
            api_key = p.best_key()
            if api_key is None:
                # Shouldn't happen given any_available() check, but be safe
                continue

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
                        "Authorization": f"Bearer {api_key.value}",
                        "Content-Type": "application/json",
                        **p.extra_headers,
                    },
                    json=payload,
                )

            if resp.status_code == 429:
                api_key.mark_rate_limited()
                p.errors += 1
                print(
                    f"\033[2m[llm] 429 on {p.name} key …{api_key.value[-6:]} "
                    f"({sum(1 for k in p.api_keys if k.is_available())} keys left in provider)\033[0m",
                    file=sys.stderr,
                )
                if use_root and p is use_provider:
                    print(f"\033[2m[llm] {p.name} provider 429 — falling back to pool\033[0m", file=sys.stderr)
                if not pool.any_available():
                    raise AllProvidersExhausted(
                        "All API keys are currently rate-limited. "
                        "Please wait a moment or add more keys to your .env."
                    )
                await asyncio.sleep(delay)
                delay *= 2
                provider = None
                continue

            if resp.status_code == 400:
                p.errors += 1
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text[:200]
                last_error = f"400 from {p.name}: {detail}"
                if use_root and p is use_provider:
                    print(f"\033[2m[llm] {p.name} provider 400 — falling back to pool\033[0m", file=sys.stderr)
                provider = None
                continue

            resp.raise_for_status()
            data = resp.json()
            api_key.calls += 1
            return data["choices"][0]["message"]["content"]

        except AllProvidersExhausted:
            raise

        except (httpx.ConnectError, httpx.TimeoutException, KeyError) as e:
            p.errors += 1
            last_error = str(e)
            if use_root and p is use_provider:
                print(f"\033[2m[llm] {p.name} provider error ({e.__class__.__name__}) — falling back\033[0m", file=sys.stderr)
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