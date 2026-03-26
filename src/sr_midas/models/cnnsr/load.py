"""Functions for loading pre-trained CNNSR models.

Canonical source: SRML-HEDM/CNNSR/cnnsr_functions.py
"""

import json
import os

import torch
import torch.nn as nn

from sr_midas.models.cnnsr.architecture import CNNSR

SEP = os.sep


# ----------------------------
def load_trained_CNNSR(mod_dir, mod_itr, torch_devs):
    """Loads a trained CNNSR model from a directory.
    The model directory must contain:
        - mod-it{mod_itr}.pth: model weights
        - _train_args.json: model training arguments
    Args:
        mod_dir (str): full path to directory containing the trained model
        mod_itr (int): model iteration number
        torch_devs (torch.device): device for loading the model
    Returns:
        mod (torch model): loaded model instance with weights
        mod_args (dict): arguments used for model training
        X_channels (list): input channel indices (0=Intensity, 1=Radius, 2=Eta)
    """

    base = mod_dir

    mod_args_filepath = f"{base}{SEP}_train_args.json"
    with open(mod_args_filepath, 'r') as f:
        mod_args = json.load(f)

    X_channels = [0]
    if mod_args["useRch"] in ["True", "true", True, "Yes", "yes", "Y", "y"]:
        X_channels.append(1)
    if mod_args["useEtach"] in ["True", "true", True, "Yes", "yes", "Y", "y"]:
        X_channels.append(2)

    in_ch = len(X_channels)

    mod_itr_path = f"{base}{SEP}mod-it{mod_itr}.pth"

    l_layer_params = mod_args["arch"].split("_")
    l_ch_nrs = [int(i.split("-")[0]) for i in l_layer_params]
    l_ker_size = [int(i.split("-")[1]) for i in l_layer_params]
    l_act_func = [str(i.split("-")[2]) for i in l_layer_params]

    mod = CNNSR(l_ch_nrs, l_ker_size, l_act_func, in_ch)
    mod = nn.DataParallel(mod)
    mod.load_state_dict(torch.load(mod_itr_path, map_location=torch_devs, weights_only=True))
    mod = mod.to(torch_devs)

    return (mod, mod_args, X_channels)
