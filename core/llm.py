"""
LLM Client — async wrapper around any OpenAI-compatible endpoint.

Dispatch is now class-based, not depth-based:
  chat(..., model_class=ModelClass.ORCHESTRATOR)  → strongest available
  chat(..., model_class=ModelClass.WORKER)        → light execution model
  etc.

The caller decides which class is appropriate for the task type.
Automatic key rotation and exponential backoff on 429s are preserved.
"""
import asyncio
import json
import sys
import httpx

from core.providers import get_pool, Provider, AllProvidersExhausted
from core.model_classes import ModelClass


MAX_RETRIES = 6
RETRY_DELAY = 1.0


async def chat(
    messages: list[dict],
    system: str = "",
    temperature: float = 0.4,
    max_tokens: int = 1024,
    provider: Provider | None = None,
    model_class: ModelClass = ModelClass.WORKER,
    # ── Legacy compat (ignored — kept so old call sites don't crash) ──────────
    depth: int = 0,
    max_depth: int = 6,
    prefer_root: bool = False,
    prefer_merge: bool = False,
) -> str:
    """
    Send a chat completion. Returns the assistant reply as a string.

    model_class controls which provider bucket is used.
    Legacy prefer_root=True is mapped to ORCHESTRATOR for compatibility.
    """
    pool = get_pool()
    delay = RETRY_DELAY
    last_error = "unknown"

    # Legacy compat: prefer_root → ORCHESTRATOR
    if prefer_root:
        model_class = ModelClass.ORCHESTRATOR

    # Resolve provider once; retry may escalate through pool
    if provider is not None:
        pinned = provider
    else:
        pinned = pool.for_class(model_class)

    _log_mode = model_class.value
    print(
        f"\033[2m[llm] {_log_mode} → {pinned.name}({pinned.model.split('/')[-1]})\033[0m",
        file=sys.stderr,
    )

    pinned_tried = False

    for attempt in range(MAX_RETRIES):
        if not pinned_tried:
            p = pinned
            pinned_tried = True
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
            p = pool.for_class(model_class)
            api_key = p.best_key()
            if api_key is None:
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
                    f"({sum(1 for k in p.api_keys if k.is_available())} keys left)\033[0m",
                    file=sys.stderr,
                )
                if not pool.any_available():
                    raise AllProvidersExhausted(
                        "All API keys are currently rate-limited. "
                        "Please wait a moment or add more keys to your .env."
                    )
                await asyncio.sleep(delay)
                delay *= 2
                provider = None
                continue

            if resp.status_code in (400, 401, 413, 503):
                p.errors += 1
                try:
                    detail = resp.json()
                except Exception:
                    detail = resp.text[:200]
                last_error = f"{resp.status_code} from {p.name}: {detail}"
                print(f"\033[2m[llm] {p.name} {resp.status_code} — falling back\033[0m", file=sys.stderr)
                provider = None
                continue

            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"].get("content") if data.get("choices") else None
            if not content:
                # Empty completion (e.g. Gemini max_tokens too tight) — treat as 400
                p.errors += 1
                last_error = f"empty completion from {p.name} (finish_reason={data['choices'][0].get('finish_reason','?')})"
                print(f"\033[2m[llm] {p.name} empty response — falling back\033[0m", file=sys.stderr)
                provider = None
                continue
            api_key.calls += 1
            return content

        except AllProvidersExhausted:
            raise

        except (httpx.ConnectError, httpx.TimeoutException, KeyError) as e:
            p.errors += 1
            last_error = str(e)
            print(f"\033[2m[llm] {p.name} error ({e.__class__.__name__}) — falling back\033[0m",
                  file=sys.stderr)
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"LLM call failed after {MAX_RETRIES} attempts: {last_error}"
                )
            await asyncio.sleep(delay)
            delay *= 2
            provider = None

    raise RuntimeError(f"LLM call failed: max retries exceeded. Last error: {last_error}")


async def chat_json(
    messages: list[dict],
    system: str = "",
    temperature: float = 0.2,
    max_tokens: int = 1024,
    model_class: ModelClass = ModelClass.ANALYST,
) -> dict:
    """Like chat() but parses the response as JSON."""
    raw = await chat(
        messages, system=system, temperature=temperature,
        max_tokens=max_tokens, model_class=model_class,
    )
    clean = raw.strip()

    if clean.startswith("```"):
        lines = clean.split("\n")
        inner_lines = []
        for line in lines[1:]:
            if line.strip() == "```":
                break
            inner_lines.append(line)
        clean = "\n".join(inner_lines).strip()

    for brace in ["{", "["]:
        idx = clean.find(brace)
        if idx > 0:
            clean = clean[idx:]
            break

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

        if isinstance(parsed, str):
            _log("DOUBLE-ENCODED", parsed)
            parsed = json.loads(parsed)

        if isinstance(parsed, list):
            if len(parsed) == 1 and isinstance(parsed[0], dict):
                parsed = parsed[0]
            elif all(isinstance(x, dict) for x in parsed):
                merged = {}
                for d in parsed:
                    merged.update(d)
                parsed = merged

        _log("PARSED", str(parsed))
        return parsed

    except json.JSONDecodeError as e:
        _log("PARSE FAILED", f"{e}\nCLEAN: {clean}")
        raise ValueError(
            f"Could not parse JSON from model response: {e}\nRaw: {raw[:300]}"
        )
