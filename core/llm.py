"""
LLM Client - async wrapper around any OpenAI-compatible endpoint
with automatic key rotation and retry on rate-limit errors.

V-shape tier routing:
  Going DOWN the tree:  depth 0 → tier 0, depth 1 → tier 1, depth N → tier N
  Coming UP (merges):   depth 0 merge → tier 0, depth 1 merge → tier 1, etc.
  Leaf nodes resolve to tier 0 (same as root) because their tier_idx is clamped
  to the lightest available tier, then the merge pass at the same depth rises
  back — so root and leaves both use the strongest model, middle uses lighter.

  tier_for_depth(depth, max_depth, num_tiers) maps depth to a tier index
  where 0 = strongest. Both plan (going down) and merge (going up) use this
  same function — the caller passes `merge=True` to mirror the index.
"""
import asyncio
import json
import httpx
from core.providers import get_pool, Provider, AllProvidersExhausted


MAX_RETRIES = 6
RETRY_DELAY = 1.0


def tier_for_depth(depth: int, max_depth: int, num_tiers: int, merge: bool = False) -> int:
    """
    Map tree depth to a quality tier index (0 = strongest, num_tiers-1 = lightest).

    Going down (merge=False): depth 0 → tier 0, deeper → higher tier index (lighter).
    Going up   (merge=True):  mirrors — depth 0 merge → tier 0, deeper merge → lighter.

    Both root plan (depth 0) and leaf solve (depth == max_depth) land at tier 0
    because the deepest depth maps to num_tiers-1 on the way down, but leaf
    solves use prefer_root=True which overrides to tier 0 directly.

    The V bottom is at max_depth//2 where the lightest model is used.
    """
    if num_tiers <= 1:
        return 0
    # Normalise depth to [0, 1] range
    half = max(1, max_depth)
    ratio = min(depth / half, 1.0)
    if merge:
        # Coming back up: deepest merge = lightest, shallowest merge = strongest
        tier = round(ratio * (num_tiers - 1))
    else:
        # Going down: root = strongest, deeper = lighter
        tier = round(ratio * (num_tiers - 1))
    return min(tier, num_tiers - 1)


async def chat(
    messages: list[dict],
    system: str = "",
    temperature: float = 0.4,
    max_tokens: int = 2048,
    provider: Provider | None = None,
    depth: int = 0,
    max_depth: int = 6,
    prefer_root: bool = False,   # force tier 0 (leaf nodes, root orchestration)
    prefer_merge: bool = False,  # merge pass — mirrors depth back up the V
) -> str:
    """
    Send a chat completion request. Returns the assistant's reply as a string.

    Tier selection (V-shape):
      prefer_root=True  → tier 0 (strongest) — used for root plan and leaf solves
      prefer_merge=True → tier mirrored by depth going back up — used for all merges
      default           → tier by depth going down — lighter toward the middle
    """
    import sys
    pool = get_pool()
    num_tiers = pool.num_tiers()
    delay = RETRY_DELAY
    last_error = "unknown"

    # Resolve which provider to use
    if prefer_root or depth == 0:
        use_provider = pool.provider_for_tier(0)
    elif prefer_merge:
        tier = tier_for_depth(depth, max_depth, num_tiers, merge=True)
        use_provider = pool.provider_for_tier(tier)
    else:
        tier = tier_for_depth(depth, max_depth, num_tiers, merge=False)
        use_provider = pool.provider_for_tier(tier)

    import sys as _sys
    _mode = "root" if (prefer_root or depth == 0) else ("merge" if prefer_merge else f"depth{depth}")
    print(f"\033[2m[llm] {_mode} → {use_provider.name}({use_provider.model.split('/')[-1]})\033[0m",
          file=_sys.stderr)

    pinned = use_provider  # remember the chosen provider across retries
    pinned_tried = False

    for attempt in range(MAX_RETRIES):
        # First attempt: use the resolved tier provider
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
            p = pool.next_provider()
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
                    f"({sum(1 for k in p.api_keys if k.is_available())} keys left in provider)\033[0m",
                    file=sys.stderr,
                )
                if p is pinned:
                    print(f"\033[2m[llm] {p.name} 429 — falling back to pool\033[0m",
                          file=sys.stderr)
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
                if p is pinned:
                    print(f"\033[2m[llm] {p.name} 400 — falling back to pool\033[0m",
                          file=sys.stderr)
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
            if p is pinned:
                print(f"\033[2m[llm] {p.name} error ({e.__class__.__name__}) — falling back\033[0m",
                      file=sys.stderr)
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
    """Like chat() but parses the response as JSON."""
    raw = await chat(messages, system=system, temperature=temperature, max_tokens=max_tokens)
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
        raise ValueError(f"Could not parse JSON from model response: {e}\nRaw: {raw[:300]}")
