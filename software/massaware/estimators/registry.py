"""Name -> Estimator class registry.

Estimator modules self-register at import time, e.g.:

    from massaware.estimators.registry import register
    register("pid_error", PIDErrorEstimator)

`build()` is called once at startup with the chosen name and full config dict;
each estimator decides which keys it needs.
"""

from __future__ import annotations

from massaware.estimators.base import Estimator

_REGISTRY: dict[str, type[Estimator]] = {}


def register(name: str, cls: type[Estimator]) -> None:
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"Estimator '{name}' is already registered to {_REGISTRY[name]!r}")
    _REGISTRY[name] = cls


def build(name: str, cfg: dict) -> Estimator:
    if name not in _REGISTRY:
        raise KeyError(
            f"Estimator '{name}' not registered. Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](cfg)


def available() -> list[str]:
    return sorted(_REGISTRY)
