#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration utilities and project path constants.

Centralizes config loading logic and defines the project directory layout
so that all other modules can resolve paths consistently.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from easydict import EasyDict as edict

# ---------------------------------------------------------------------------
# Project directory layout
# ---------------------------------------------------------------------------
# All paths are anchored to the demo/ root (two levels up from this file).
DEMO_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DEMO_DIR.parent
MOEFACE_PROJECT_DIR = PROJECT_ROOT / "Moeface_project"

# Make Moeface_project importable (needed by model.py, result_parser.py, etc.)
if str(MOEFACE_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(MOEFACE_PROJECT_DIR))


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> None:
    """Create a directory (and parents) if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def to_edict(obj: Any) -> Any:
    """Recursively convert nested dicts / lists into EasyDict objects."""
    if isinstance(obj, dict):
        return edict({k: to_edict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_edict(v) for v in obj]
    return obj


def load_yaml_as_edict(path: str | Path) -> edict:
    """Load a YAML file and return its contents as a nested EasyDict.

    Any relative path values under ``paths.*`` and ``runtime.output_dir``
    will be resolved against the YAML file's parent directory (the demo/ root).
    """
    yaml_path = Path(path).resolve()
    yaml_dir = yaml_path.parent

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = to_edict(yaml.safe_load(f))

    def _resolve(section):
        if not hasattr(cfg, section):
            return
        for key, val in vars(getattr(cfg, section)).items():
            if isinstance(val, str) and not Path(val).is_absolute():
                setattr(getattr(cfg, section), key, str(yaml_dir / val))

    _resolve("paths")
    if hasattr(cfg, "runtime"):
        _resolve("runtime")

    return cfg
