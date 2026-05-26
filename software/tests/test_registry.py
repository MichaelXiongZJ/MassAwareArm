"""Estimator registry: register / build / duplicate guard."""

import pytest

from massaware.estimators.base import EstimateResult, Estimator
from massaware.estimators.registry import _REGISTRY, available, build, register


class _StubEstimator(Estimator):
    name = "_stub"

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self): pass
    def update(self, obs): pass
    def estimate(self): return EstimateResult(m_hat=0.0)


@pytest.fixture(autouse=True)
def _restore_registry():
    """Snapshot/restore the registry around each test so re-registration races
    across tests don't bleed."""
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


def test_register_and_build():
    register("_stub", _StubEstimator)
    est = build("_stub", {"foo": 1})
    assert isinstance(est, _StubEstimator)
    assert est.cfg == {"foo": 1}


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        build("_nonexistent_estimator", {})


def test_duplicate_registration_with_same_class_is_ok():
    register("_stub", _StubEstimator)
    # Re-registering the SAME class is idempotent.
    register("_stub", _StubEstimator)


def test_duplicate_registration_with_different_class_raises():
    class _Other(_StubEstimator):
        pass
    register("_stub", _StubEstimator)
    with pytest.raises(ValueError):
        register("_stub", _Other)


def test_available_lists_registered():
    register("_stub", _StubEstimator)
    names = available()
    assert "_stub" in names
    # Concrete estimators auto-register via massaware/estimators/__init__.py
    # if their module has been imported — don't assert on those here to keep
    # this test isolated.
