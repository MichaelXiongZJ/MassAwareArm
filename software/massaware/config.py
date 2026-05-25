"""Simple YAML configuration loader."""

from __future__ import annotations

import yaml
from pathlib import Path
import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    """Load YAML config and convert degree poses to radians."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Convert pose arrays from degrees to radians
    if "poses" in cfg:
        for key in ["home_qpos", "weigh_qpos"]:
            if key in cfg["poses"]:
                cfg["poses"][key] = np.deg2rad(cfg["poses"][key])
        
        for key in ["light_bin_drop", "heavy_bin_drop"]:
            if key in cfg["poses"]:
                cfg["poses"][key] = np.array(cfg["poses"][key])

    return cfg
