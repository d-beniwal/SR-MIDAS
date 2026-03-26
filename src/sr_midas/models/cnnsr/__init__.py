"""CNNSR model sub-package: architecture, training, inference, and pretrained weights."""

from sr_midas.models.cnnsr.architecture import CNNSR
from sr_midas.models.cnnsr.load import load_trained_CNNSR
from sr_midas.models.cnnsr.predict import predict_CNNSR, predict_CNNSR_singleMod, err_from_log
from sr_midas.models.cnnsr.train import train_cnnsr
from sr_midas.models.cnnsr.hp_optimize import run_hp_optimize
from sr_midas.models.cnnsr.pretrained import get_model_dir, MODEL_NAMES, MODEL_ITRS

__all__ = [
    "CNNSR",
    "load_trained_CNNSR",
    "predict_CNNSR",
    "predict_CNNSR_singleMod",
    "err_from_log",
    "train_cnnsr",
    "run_hp_optimize",
    "get_model_dir",
    "MODEL_NAMES",
    "MODEL_ITRS",
]
