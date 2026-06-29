"""
Provider Pool Manager
Loads providers from .env and organises them into quality tiers.

Tier ordering (0 = strongest, N = lightest):
  PROVIDER_ROOT_*   → tier 0  (root plan + leaf solves + final merge)
  PROVIDER_TIER1_*  → tier 1  (one level in from root/leaf)
  PROVIDER_TIER2_*  → tier 2  (two levels in)
  ...
  PROVIDER_*        → lowest tier (any remaining named providers)

The V-shape routing in llm.py maps depth → tier going down, and
(max_depth - depth) → tier coming back up on merges. Leaf nodes and
root node both resolve to tier 0 (same model quality, different token
budgets). Middle of the tree uses lighter models.

If a tier has no provider configured it falls back to the next available
tier (graceful degradation).
"""
import os
import time
import asyncio
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class AllProvidersExhausted(RuntimeError):
    """Raised when every key across every provider is rate-limited."""
    pass


@dataclass
class ApiKey:
    value: str
    provider_name: str
    calls: int = 0
    rate_limited: bool = False
    rate_limited_at: float = 0.0

    COOLDOWN = 60.0

    def is_available(self) -> bool:
        if not self.rate_limited:
            return True
        if time.monotonic() - self.rate_limited_at > self.COOLDOWN:
            self.rate_limited = False
        return not self.rate_limited

    def mark_rate_limited(self):
        self.rate_limited = True
        self.rate_limited_at = time.monotonic()


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    api_keys: list[ApiKey]
    extra_headers: dict[str, str] = field(default_factory=dict)
    errors: int = 0

    @property
    def calls(self) -> int:
        return sum(k.calls for k in self.api_keys)

    @property
    def keys(self) -> list[str]:
        return [k.value for k in self.api_keys]

    def best_key(self) -> Optional[ApiKey]:
        available = [k for k in self.api_keys if k.is_available()]
        if not available:
            return None
        return min(available, key=lambda k: k.calls)

    def all_exhausted(self) -> bool:
        return all(not k.is_available() for k in self.api_keys)

    def key_status(self) -> list[dict]:
        return [
            {
                "label": f"{self.name.lower()}{i + 1}",
                "key_suffix": k.value[-6:],
                "calls": k.calls,
                "rate_limited": k.rate_limited,
            }
            for i, k in enumerate(self.api_keys)
        ]


def _make_provider(name: str, base_url: str, model: str, raw_keys: str,
                   extra_headers: dict) -> Provider:
    keys = [
        ApiKey(value=v.strip(), provider_name=name)
        for v in raw_keys.split(",") if v.strip()
    ]
    return Provider(name=name, base_url=base_url, model=model,
                    api_keys=keys, extra_headers=extra_headers)


def _extra_headers(base_url: str) -> dict[str, str]:
    if "openrouter.ai" in base_url:
        return {"HTTP-Referer": "https://github.com/hivemind", "X-Title": "HiveMind"}
    return {}


class ProviderPool:
    """
    Manages providers organised into quality tiers (0 = strongest).

    tiers: list[Provider]  — index 0 is strongest (ROOT), ascending = lighter
    providers: list[Provider]  — alias for the lowest tier, used for pool rotation
    root_provider: Provider  — tier 0 (backwards compat)
    merge_provider: Provider  — tier 0 (final merge; same model, heavier budget in llm.py)
    """

    def __init__(self):
        self.tiers: list[Provider] = []   # ordered strongest → lightest
        self._lock = asyncio.Lock()
        self._load_providers()

    def _load_providers(self):
        reserved = {"ROOT", "MERGE"}

        # ── Tier 0: ROOT ──────────────────────────────────────────────────────
        root_url  = os.environ.get("PROVIDER_ROOT_BASE_URL", "").strip()
        root_keys = os.environ.get("PROVIDER_ROOT_KEYS", "").strip()
        if root_url and root_keys:
            root_model = os.environ.get("PROVIDER_ROOT_MODEL", "gpt-4o").strip()
            self.tiers.append(_make_provider("ROOT", root_url, root_model,
                                             root_keys, _extra_headers(root_url)))

        # ── Named tiers: TIER1, TIER2, … ─────────────────────────────────────
        tier_idx = 1
        while True:
            name = f"TIER{tier_idx}"
            url  = os.environ.get(f"PROVIDER_{name}_BASE_URL", "").strip()
            keys = os.environ.get(f"PROVIDER_{name}_KEYS", "").strip()
            if not url or not keys:
                break
            model = os.environ.get(f"PROVIDER_{name}_MODEL", "gpt-3.5-turbo").strip()
            self.tiers.append(_make_provider(name, url, model, keys, _extra_headers(url)))
            reserved.add(name)
            tier_idx += 1

        # ── Remaining named providers → lowest tier ───────────────────────────
        seen: set[str] = set()
        for key in os.environ:
            if key.startswith("PROVIDER_") and key.endswith("_BASE_URL"):
                name = key[len("PROVIDER_"):-len("_BASE_URL")]
                if name in reserved or name in seen:
                    continue
                seen.add(name)
                url   = os.environ.get(f"PROVIDER_{name}_BASE_URL", "").strip()
                model = os.environ.get(f"PROVIDER_{name}_MODEL", "gpt-3.5-turbo").strip()
                raw   = os.environ.get(f"PROVIDER_{name}_KEYS", "").strip()
                if url and raw:
                    self.tiers.append(_make_provider(name, url, model, raw,
                                                     _extra_headers(url)))

        if not self.tiers:
            raise RuntimeError(
                "No providers found in .env!\n"
                "Copy .env.example to .env and fill in at least one provider."
            )

    # ── Backwards-compat properties ───────────────────────────────────────────

    @property
    def root_provider(self) -> Optional[Provider]:
        return self.tiers[0] if self.tiers else None

    @property
    def merge_provider(self) -> Optional[Provider]:
        return self.tiers[0] if self.tiers else None

    @property
    def providers(self) -> list[Provider]:
        """Lowest tier providers — used for pool rotation fallback."""
        return self.tiers[-1:] if self.tiers else []

    # ── Tier resolution ───────────────────────────────────────────────────────

    def provider_for_tier(self, tier_idx: int) -> Provider:
        """
        Return the provider at tier_idx, clamped to available tiers.
        Gracefully degrades: if tier_idx > len(tiers)-1, returns the lightest.
        """
        idx = min(tier_idx, len(self.tiers) - 1)
        return self.tiers[idx]

    def num_tiers(self) -> int:
        return len(self.tiers)

    # ── Pool rotation (within the lowest tier or all tiers as fallback) ───────

    def next_provider(self) -> Provider:
        """Least-used available provider across all tiers (fallback path)."""
        candidates = [p for p in self.tiers if not p.all_exhausted()]
        if not candidates:
            raise AllProvidersExhausted(
                "All API keys are rate-limited. Please wait a moment or add more keys."
            )
        return min(candidates, key=lambda p: p.best_key().calls)  # type: ignore

    def any_available(self) -> bool:
        return any(not p.all_exhausted() for p in self.tiers)

    def stats(self) -> list[dict]:
        return [
            {
                "name": p.name, "model": p.model,
                "calls": p.calls, "errors": p.errors,
                "keys": len(p.api_keys), "key_status": p.key_status(),
            }
            for p in self.tiers
        ]


# Singleton
_pool: Optional[ProviderPool] = None

def get_pool() -> ProviderPool:
    global _pool
    if _pool is None:
        _pool = ProviderPool()
    return _pool
