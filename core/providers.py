"""
Provider Pool Manager
Loads any number of OpenAI-compatible providers from .env and rotates keys.
"""
import os
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional
from itertools import cycle
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    keys: list[str]
    extra_headers: dict[str, str] = field(default_factory=dict)
    _key_cycle: object = field(init=False, repr=False)
    _lock: asyncio.Lock = field(init=False, repr=False)
    calls: int = 0
    errors: int = 0

    def __post_init__(self):
        self._key_cycle = cycle(self.keys)
        self._lock = asyncio.Lock()

    def next_key(self) -> str:
        return next(self._key_cycle)


class ProviderPool:
    """Discovers and manages all PROVIDER_* entries in .env"""

    def __init__(self):
        self.providers: list[Provider] = []
        self._provider_cycle = None
        self._lock = asyncio.Lock()
        self._load_providers()

    def _load_providers(self):
        seen = set()
        for key in os.environ:
            if key.startswith("PROVIDER_") and key.endswith("_BASE_URL"):
                name = key[len("PROVIDER_"):-len("_BASE_URL")]
                if name in seen:
                    continue
                seen.add(name)

                base_url = os.environ.get(f"PROVIDER_{name}_BASE_URL", "").strip()
                model = os.environ.get(f"PROVIDER_{name}_MODEL", "gpt-3.5-turbo").strip()
                raw_keys = os.environ.get(f"PROVIDER_{name}_KEYS", "").strip()
                keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

                if not base_url or not keys:
                    continue

                extra_headers: dict[str, str] = {}
                if "openrouter.ai" in base_url:
                    extra_headers = {
                        "HTTP-Referer": "https://github.com/hivemind",
                        "X-Title": "HiveMind",
                    }

                self.providers.append(Provider(
                    name=name,
                    base_url=base_url,
                    model=model,
                    keys=keys,
                    extra_headers=extra_headers,
                ))

        if not self.providers:
            raise RuntimeError(
                "No providers found in .env!\n"
                "Copy .env.example to .env and fill in at least one provider."
            )

        self._provider_cycle = cycle(self.providers)

    def next_provider(self) -> Provider:
        """Round-robin across providers"""
        return next(self._provider_cycle)

    def stats(self) -> list[dict]:
        return [
            {"name": p.name, "model": p.model, "keys": len(p.keys),
             "calls": p.calls, "errors": p.errors}
            for p in self.providers
        ]


# Singleton
_pool: Optional[ProviderPool] = None

def get_pool() -> ProviderPool:
    global _pool
    if _pool is None:
        _pool = ProviderPool()
    return _pool
