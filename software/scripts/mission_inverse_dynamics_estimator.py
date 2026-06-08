"""Run the main mission pipeline with the inverse-dynamics estimator.

This wrapper leaves the stock mission.py unchanged. It imports the
inverse_dynamics estimator module so it self-registers, then delegates to the
normal mission entry point. If no estimator is specified, it defaults to
``--estimator inverse_dynamics``.
"""

from __future__ import annotations

import sys
from pathlib import Path

SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
if str(SOFTWARE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOFTWARE_ROOT))

# Registration side effect.
from massaware.estimators import inverse_dynamics as _inverse_dynamics  # noqa: F401

from mission import main as mission_main


def main() -> int:
    if "--estimator" not in sys.argv:
        sys.argv.extend(["--estimator", "inverse_dynamics"])
    return mission_main()


if __name__ == "__main__":
    raise SystemExit(main())
