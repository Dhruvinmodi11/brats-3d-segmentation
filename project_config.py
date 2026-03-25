from __future__ import annotations

import os
from pathlib import Path


def _resolve_project_root() -> Path:
    # Priority: explicit PROJECT_ROOT, then DATA_ROOT, else repo root.
    root = os.environ.get("PROJECT_ROOT") or os.environ.get("DATA_ROOT")
    if root:
        return Path(root).resolve()
    return Path(__file__).resolve().parent


PROJECT_ROOT = _resolve_project_root()

# Canonical paths
MODELS_DIR = PROJECT_ROOT / "models"
V2_MODELS_DIR = MODELS_DIR / "v2"
MIXED_MODELS_DIR = MODELS_DIR / "mixed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PATCHES_DIR = PROJECT_ROOT / "patches"
TRAINING_DIR = PROJECT_ROOT / "training"
INFERENCE_DIR = PROJECT_ROOT / "inference"

# Model defaults (override with env vars when needed)
MODEL_IN_CH = int(os.environ.get("MODEL_IN_CH", "4"))
MODEL_NUM_CLASSES = int(os.environ.get("MODEL_NUM_CLASSES", "4"))
MODEL_V1_BASE = int(os.environ.get("MODEL_V1_BASE", "24"))
MODEL_V2_BASE = int(os.environ.get("MODEL_V2_BASE", "24"))

