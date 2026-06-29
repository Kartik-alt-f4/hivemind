"""
Provider Pool Manager
Loads any number of OpenAI-compatible providers from .env and rotates keys.
Tracks per-key call counts and rate-limit state; selects the least-used
available key across providers. When all keys are rate-limited it raises
AllProvidersExhausted so the caller can surface a user-facing message.
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

    # Auto-clear rate-limit after this many seconds (conservative — most free
    # tiers reset per-minute or per-day; we use 60s so short limits recover fast
    # and the day-limit case just accumulates until EOD reset in TokenMeter).
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

    # Legacy shim — node.py reads p.calls; sum per-key calls on access.
    @property
    def calls(self) -> int:
        return sum(k.calls for k in self.api_keys)

    @property
    def keys(self) -> list[str]:
        return [k.value for k in self.api_keys]

    def best_key(self) -> Optional[ApiKey]:
        """Return the available key with fewest calls (least-used), or None."""
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


class ProviderPool:
    """Discovers and manages all PROVIDER_* entries in .env"""

    def __init__(self):
        self.providers: list[Provider] = []
        self.root_provider: Optional[Provider] = None
        self.merge_provider: Optional[Provider] = None  # heavy model for merge/integration
        self._lock = asyncio.Lock()
        self._load_providers()

    def _load_providers(self):
        root_url  = os.environ.get("PROVIDER_ROOT_BASE_URL", "").strip()
        root_keys = os.environ.get("PROVIDER_ROOT_KEYS", "").strip()
        if root_url and root_keys:
            root_model = os.environ.get("PROVIDER_ROOT_MODEL", "gpt-4o").strip()
            extra: dict[str, str] = {}
            if "openrouter.ai" in root_url:
                extra = {"HTTP-Referer": "https://github.com/hivemind", "X-Title": "HiveMind"}
            self.root_provider = _make_provider("ROOT", root_url, root_model, root_keys, extra)

        # Optional dedicated merge provider (heavy model for synthesis passes)
        merge_url  = os.environ.get("PROVIDER_MERGE_BASE_URL", "").strip()
        merge_keys = os.environ.get("PROVIDER_MERGE_KEYS", "").strip()
        if merge_url and merge_keys:
            merge_model = os.environ.get("PROVIDER_MERGE_MODEL", "gpt-4o").strip()
            extra_m: dict[str, str] = {}
            if "openrouter.ai" in merge_url:
                extra_m = {"HTTP-Referer": "https://github.com/hivemind", "X-Title": "HiveMind"}
            self.merge_provider = _make_provider("MERGE", merge_url, merge_model, merge_keys, extra_m)
        else:
            # Fall back to root provider for merge passes
            self.merge_provider = self.root_provider

        seen: set[str] = set()
        for key in os.environ:
            if key.startswith("PROVIDER_") and key.endswith("_BASE_URL"):
                name = key[len("PROVIDER_"):-len("_BASE_URL")]
                if name in seen or name in ("ROOT", "MERGE"):
                    continue
                seen.add(name)

                base_url  = os.environ.get(f"PROVIDER_{name}_BASE_URL", "").strip()
                model     = os.environ.get(f"PROVIDER_{name}_MODEL", "gpt-3.5-turbo").strip()
                raw_keys  = os.environ.get(f"PROVIDER_{name}_KEYS", "").strip()

                if not base_url or not raw_keys:
                    continue

                extra: dict[str, str] = {}
                if "openrouter.ai" in base_url:
                    extra = {"HTTP-Referer": "https://github.com/hivemind", "X-Title": "HiveMind"}

                self.providers.append(_make_provider(name, base_url, model, raw_keys, extra))

        if not self.providers:
            raise RuntimeError(
                "No providers found in .env!\n"
                "Copy .env.example to .env and fill in at least one provider."
            )

    def next_provider(self) -> Provider:
        """Return the provider whose best available key has the fewest calls."""
        candidates = [p for p in self.providers if not p.all_exhausted()]
        if not candidates:
            raise AllProvidersExhausted(
                "All API keys are rate-limited. Please wait a moment or add more keys."
            )
        return min(candidates, key=lambda p: p.best_key().calls)  # type: ignore[union-attr]

    def any_available(self) -> bool:
        return any(not p.all_exhausted() for p in self.providers)

    def stats(self) -> list[dict]:
        rows = []
        if self.root_provider:
            p = self.root_provider
            rows.append({
                "name": p.name, "model": p.model,
                "calls": p.calls, "errors": p.errors,
                "keys": len(p.api_keys), "key_status": p.key_status(),
            })
        for p in self.providers:
            rows.append({
                "name": p.name, "model": p.model,
                "calls": p.calls, "errors": p.errors,
                "keys": len(p.api_keys), "key_status": p.key_status(),
            })
        return rows


# Singleton
_pool: Optional[ProviderPool] = None

def get_pool() -> ProviderPool:
    global _pool
    if _pool is None:
        _pool = ProviderPool()
    return _pool
