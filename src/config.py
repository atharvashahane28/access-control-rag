"""Load the project config and fix the random seed.

Every script should start with:

    from src.config import load_config, set_seed
    cfg = load_config()
    set_seed(cfg["seed"])
"""
import os
import random
from pathlib import Path

import numpy as np
import yaml

# The repo root is the folder that contains src/.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "config.yaml"


def load_config(path=DEFAULT_CONFIG):
    """Read the YAML config and turn the paths into full paths."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for name, value in cfg["paths"].items():
        cfg["paths"][name] = REPO_ROOT / value
    return cfg


def set_seed(seed):
    """Seed every random number generator the project uses."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass  # torch is only needed from Step 2 onwards