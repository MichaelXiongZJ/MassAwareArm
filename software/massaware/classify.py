"""Threshold-based mass classifier."""

from __future__ import annotations


def classify_mass(m_hat: float, threshold: float) -> str:
    """Return 'heavy' if m_hat >= threshold else 'light'."""
    return "heavy" if m_hat >= threshold else "light"
