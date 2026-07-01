"""
Provider Pool Manager
Loads providers from .env and maps them to ModelClass buckets.

.env convention:
  CLASS_<NAME>_BASE_URL  = https://...
  CLASS_<NAME>_MODEL     = model-id
  CLASS_<NAME>_KEYS      = key1,key2,...
  CLASS_<NAME>_CLASS     = orchestrator | analyst | worker | fast

  Legacy PROVIDER_ROOT_* is imported as ORCHESTRATOR for backwards compat.
  Legacy PROVIDER_TIER1_* → ANALYST, TIER2_* → WORKER, the rest → WORKER.
  Any PROVIDER_<NAME>_* without a CLASS_ prefix → WORKER (safe default).

At runtime, pool.for_class(ModelClass.ORCHESTRATOR) returns the best
available provider for that class, falling back up the capability ladder
if the preferred class is exhausted or unconfigured:
  FAST → WORKER → ANALYST → ORCHESTRATOR (escalation on exhaustion)
"""
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

from core.model_classes import ModelClass

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
    model_class: ModelClass = ModelClass.WORKER
    extra_headers: dict[str, str] = field(default_factory=dict)
    errors: int = 0
    weight: int = 1   # dispatch weight — higher = more calls routed here

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
                   model_class: ModelClass, extra_headers: dict,
                   weight: int = 1,
                   key_registry: dict | None = None) -> Provider:
    keys = []
    for v in raw_keys.split(","):
        v = v.strip()
        if not v:
            continue
        if key_registry is not None and v in key_registry:
            keys.append(key_registry[v])   # reuse shared object
        else:
            ak = ApiKey(value=v, provider_name=name)
            if key_registry is not None:
                key_registry[v] = ak
            keys.append(ak)
    return Provider(name=name, base_url=base_url, model=model,
                    api_keys=keys, model_class=model_class,
                    extra_headers=extra_headers, weight=weight)


def _extra_headers(base_url: str) -> dict[str, str]:
    if "openrouter.ai" in base_url:
        return {"HTTP-Referer": "https://github.com/hivemind", "X-Title": "HiveMind"}
    return {}


def _parse_class(raw: str) -> ModelClass:
    mapping = {
        "orchestrator": ModelClass.ORCHESTRATOR,
        "analyst":      ModelClass.ANALYST,
        "worker":       ModelClass.WORKER,
        "fast":         ModelClass.FAST,
    }
    return mapping.get(raw.strip().lower(), ModelClass.WORKER)


# Escalation order when preferred class is exhausted
_ESCALATION: dict[ModelClass, list[ModelClass]] = {
    ModelClass.FAST:         [ModelClass.FAST, ModelClass.WORKER, ModelClass.ANALYST, ModelClass.ORCHESTRATOR],
    ModelClass.WORKER:       [ModelClass.WORKER, ModelClass.ANALYST, ModelClass.ORCHESTRATOR, ModelClass.FAST],
    ModelClass.ANALYST:      [ModelClass.ANALYST, ModelClass.ORCHESTRATOR, ModelClass.WORKER, ModelClass.FAST],
    ModelClass.ORCHESTRATOR: [ModelClass.ORCHESTRATOR, ModelClass.ANALYST, ModelClass.WORKER, ModelClass.FAST],
}


class ProviderPool:
    """
    Manages providers organised by ModelClass.
    Each class can have multiple providers (for key rotation).
    """

    def __init__(self):
        self._by_class: dict[ModelClass, list[Provider]] = {mc: [] for mc in ModelClass}
        # Shared ApiKey objects keyed by raw value — same physical key across
        # multiple provider entries shares one call counter and rate-limit flag.
        self._key_registry: dict[str, ApiKey] = {}
        self._load_providers()

    def _load_providers(self):
        loaded: list[tuple[str, Provider]] = []

        # ── New-style CLASS_<NAME>_* ──────────────────────────────────────────
        seen: set[str] = set()
        for key in os.environ:
            if key.startswith("CLASS_") and key.endswith("_BASE_URL"):
                name = key[len("CLASS_"):-len("_BASE_URL")]
                if name in seen:
                    continue
                seen.add(name)
                url   = os.environ.get(f"CLASS_{name}_BASE_URL", "").strip()
                model = os.environ.get(f"CLASS_{name}_MODEL", "gpt-3.5-turbo").strip()
                raw   = os.environ.get(f"CLASS_{name}_KEYS", "").strip()
                cls_s = os.environ.get(f"CLASS_{name}_CLASS", "worker").strip()
                if url and raw:
                    mc  = _parse_class(cls_s)
                    raw_w = os.environ.get(f"CLASS_{name}_WEIGHT", "1").strip()
                    w   = int(raw_w) if raw_w.isdigit() else 1
                    p   = _make_provider(name, url, model, raw, mc, _extra_headers(url),
                                         weight=w, key_registry=self._key_registry)
                    loaded.append((name, p))

        # ── Legacy PROVIDER_ROOT_* → ORCHESTRATOR ────────────────────────────
        root_url  = os.environ.get("PROVIDER_ROOT_BASE_URL", "").strip()
        root_keys = os.environ.get("PROVIDER_ROOT_KEYS", "").strip()
        if root_url and root_keys:
            root_model = os.environ.get("PROVIDER_ROOT_MODEL", "gpt-4o").strip()
            p = _make_provider("ROOT", root_url, root_model, root_keys,
                               ModelClass.ORCHESTRATOR, _extra_headers(root_url),
                               key_registry=self._key_registry)
            loaded.append(("ROOT", p))

        # ── Legacy PROVIDER_TIER1_* → ANALYST ────────────────────────────────
        tier1_url  = os.environ.get("PROVIDER_TIER1_BASE_URL", "").strip()
        tier1_keys = os.environ.get("PROVIDER_TIER1_KEYS", "").strip()
        if tier1_url and tier1_keys:
            tier1_model = os.environ.get("PROVIDER_TIER1_MODEL", "gpt-3.5-turbo").strip()
            p = _make_provider("TIER1", tier1_url, tier1_model, tier1_keys,
                               ModelClass.ANALYST, _extra_headers(tier1_url),
                               key_registry=self._key_registry)
            loaded.append(("TIER1", p))

        # ── Legacy PROVIDER_TIER2_* → WORKER ─────────────────────────────────
        tier2_url  = os.environ.get("PROVIDER_TIER2_BASE_URL", "").strip()
        tier2_keys = os.environ.get("PROVIDER_TIER2_KEYS", "").strip()
        if tier2_url and tier2_keys:
            tier2_model = os.environ.get("PROVIDER_TIER2_MODEL", "gpt-3.5-turbo").strip()
            p = _make_provider("TIER2", tier2_url, tier2_model, tier2_keys,
                               ModelClass.WORKER, _extra_headers(tier2_url),
                               key_registry=self._key_registry)
            loaded.append(("TIER2", p))

        # ── Legacy PROVIDER_<NAME>_* (everything else) → WORKER ──────────────
        legacy_reserved = {"ROOT", "MERGE", "TIER1", "TIER2", "TIER3", "TIER4"}
        legacy_seen: set[str] = set()
        for key in os.environ:
            if key.startswith("PROVIDER_") and key.endswith("_BASE_URL"):
                name = key[len("PROVIDER_"):-len("_BASE_URL")]
                if name in legacy_reserved or name in legacy_seen:
                    continue
                legacy_seen.add(name)
                url   = os.environ.get(f"PROVIDER_{name}_BASE_URL", "").strip()
                model = os.environ.get(f"PROVIDER_{name}_MODEL", "gpt-3.5-turbo").strip()
                raw   = os.environ.get(f"PROVIDER_{name}_KEYS", "").strip()
                if url and raw:
                    p = _make_provider(name, url, model, raw,
                                       ModelClass.WORKER, _extra_headers(url),
                                       key_registry=self._key_registry)
                    loaded.append((name, p))

        if not loaded:
            raise RuntimeError(
                "No providers found in .env!\n"
                "Add at least one CLASS_<NAME>_* or PROVIDER_* entry."
            )

        for _name, p in loaded:
            self._by_class[p.model_class].append(p)

    # ── Primary dispatch ──────────────────────────────────────────────────────

    def for_class(self, mc: ModelClass) -> Provider:
        """
        Return an available provider for the given ModelClass using weighted
        dispatch, then escalate through the capability ladder if exhausted.

        Weighted dispatch: each provider's selection probability is proportional
        to (weight / (calls + 1)) — weight biases toward high-limit providers,
        call count naturally load-balances as usage accumulates.
        Within the chosen provider, the least-used available key is picked.
        """
        for candidate_class in _ESCALATION[mc]:
            available = [p for p in self._by_class[candidate_class]
                         if not p.all_exhausted() and p.weight > 0]
            if not available:
                continue
            if len(available) == 1:
                return available[0]
            # Weighted random: weight / (calls + 1) so busier providers cool off
            scores = [p.weight / (p.calls + 1) for p in available]
            total  = sum(scores)
            pick   = random.random() * total
            cumulative = 0.0
            for p, score in zip(available, scores):
                cumulative += score
                if pick <= cumulative:
                    return p
            return available[-1]   # float rounding fallback
        raise AllProvidersExhausted(
            "All API keys are rate-limited. Please wait or add more keys."
        )

    def any_available(self) -> bool:
        return any(
            not p.all_exhausted() and p.weight > 0
            for providers in self._by_class.values()
            for p in providers
        )

    # ── Backwards-compat properties (used by widget_server, hm.py) ───────────

    @property
    def tiers(self) -> list[Provider]:
        """Flat list: orchestrator first, fast last."""
        out = []
        for mc in [ModelClass.ORCHESTRATOR, ModelClass.ANALYST,
                   ModelClass.WORKER, ModelClass.FAST]:
            out.extend(self._by_class[mc])
        return out

    @property
    def root_provider(self) -> Optional[Provider]:
        providers = self._by_class[ModelClass.ORCHESTRATOR]
        return providers[0] if providers else (self.tiers[0] if self.tiers else None)

    @property
    def merge_provider(self) -> Optional[Provider]:
        return self.root_provider

    @property
    def providers(self) -> list[Provider]:
        """Lowest-class providers — backwards compat."""
        for mc in [ModelClass.FAST, ModelClass.WORKER]:
            ps = self._by_class[mc]
            if ps:
                return ps
        return self.tiers[-1:] if self.tiers else []

    def num_tiers(self) -> int:
        return len(self.tiers)

    def next_provider(self) -> Provider:
        return self.for_class(ModelClass.WORKER)

    def provider_for_tier(self, tier_idx: int) -> Provider:
        order = [ModelClass.ORCHESTRATOR, ModelClass.ANALYST,
                 ModelClass.WORKER, ModelClass.FAST]
        mc = order[min(tier_idx, len(order) - 1)]
        return self.for_class(mc)

    def stats(self) -> list[dict]:
        return [
            {
                "name": p.name, "model": p.model,
                "class": p.model_class.value,
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
