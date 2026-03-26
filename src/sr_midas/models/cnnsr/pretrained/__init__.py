"""Provides access to bundled pre-trained CNNSR model weights."""

import importlib.resources as ilr
from pathlib import Path

MODEL_NAMES = {
    "x1_x2": "x1_x2-pst60000-240117_arch3-itrOut",
    "x2_x4": "x2pred_x4-pst60000-240117_arch3-itrOut",
    "x4_x8": "x4Pred_x8-pst60000-240117_arch3-itrOut",
    "x1_x8": "x1_x8-pst60000-250606_arch3-itrOut",
}

MODEL_ITRS = {
    "x1_x2": 4975,
    "x2_x4": 999,
    "x4_x8": 999,
    "x1_x8": 1300,
}


def get_model_dir(name: str) -> Path:
    """Return the absolute path to a bundled pre-trained model directory.

    Args:
        name: one of 'x1_x2', 'x2_x4', 'x4_x8', 'x1_x8'

    Returns:
        Path to the model directory containing mod-it*.pth and _train_args.json
    """
    if name not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: {list(MODEL_NAMES)}"
        )
    pkg = ilr.files("sr_midas.models.cnnsr.pretrained") / MODEL_NAMES[name]
    return Path(str(pkg))
