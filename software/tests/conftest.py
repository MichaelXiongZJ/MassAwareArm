"""Shared pytest fixtures and path setup."""

import sys
from pathlib import Path

# Make `massaware` importable when running `pytest software/tests` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
