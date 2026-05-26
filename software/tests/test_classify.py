"""Threshold classifier boundary behavior."""

from massaware.classify import classify_mass


def test_below_threshold_is_light():
    assert classify_mass(0.1, 0.35) == "light"


def test_above_threshold_is_heavy():
    assert classify_mass(0.5, 0.35) == "heavy"


def test_exact_threshold_is_heavy():
    # Convention: >= threshold counts as heavy.
    assert classify_mass(0.35, 0.35) == "heavy"


def test_negative_m_hat_is_light():
    # A negative estimate (degenerate measurement) must not crash.
    assert classify_mass(-0.1, 0.35) == "light"
