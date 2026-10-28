from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

import torch


def load_fg2_components(repository: str | Path) -> dict[str, Any]:
    """Import the released FG2 KITTI model without editing its source tree.

    The upstream modules read ``config.ini`` relative to the current directory
    and use top-level package names (``models``, ``utils``, ``DINO_modules``).
    This loader establishes the upstream path and working directory only while
    those modules are imported.
    """

    repository = Path(repository).expanduser().resolve()
    if not (repository / "models" / "model_kitti.py").is_file():
        raise FileNotFoundError(f"FG2 repository not found or incomplete: {repository}")
    for name, expected in (("models", repository / "models"), ("utils", repository / "utils")):
        loaded = sys.modules.get(name)
        loaded_file = getattr(loaded, "__file__", None) if loaded is not None else None
        if loaded_file and Path(loaded_file).resolve().parent != expected.resolve():
            raise RuntimeError(
                f"Python package {name!r} is already loaded from {loaded_file}; "
                "run this FG2 entry point in a fresh Python process."
            )

    repository_string = str(repository)
    if repository_string in sys.path:
        sys.path.remove(repository_string)
    sys.path.insert(0, repository_string)
    previous_cwd = Path.cwd()
    try:
        os.chdir(repository)
        model_module = importlib.import_module("models.model_kitti")
        feature_module = importlib.import_module("models.modules")
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("mmcv"):
            raise RuntimeError(
                "FG2 requires MMCV with the MultiScaleDeformableAttention CUDA op. "
                "Install a build compatible with the active PyTorch/CUDA environment."
            ) from error
        raise
    finally:
        os.chdir(previous_cwd)

    return {
        "CVM": model_module.CVM,
        "DinoExtractor": feature_module.DinoExtractor,
    }


def load_dinov2_weights(path: str | Path | None) -> Any:
    if path is None or str(path).strip().lower() in {"", "none", "null"}:
        return None
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("teacher", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"DINOv2 checkpoint must contain a state dict: {path}")
    return checkpoint

