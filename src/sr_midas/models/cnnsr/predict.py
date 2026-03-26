"""Inference functions for CNNSR models: batch prediction and error extraction.

Source: SRML-HEDM/CNNSR/cnnsr_functions.py
"""

import time

import numpy as np
import torch
from copy import deepcopy

from sr_midas.data.upscale import upscale
from sr_midas.models.cnnsr.load import load_trained_CNNSR


# ----------------------------
def predict_CNNSR(patch_x1, mods_to_use, print_run_time=True, batch_size=200):
    """Predict SR patches at x2, x4, x8 using cascaded pre-trained models.
    Args:
        patch_x1 (arr): input patches at native x1 resolution (N, C, H, W)
        mods_to_use (dict): paths to pre-trained models, e.g.:
            {
                'SRx2': {'mod_dir': path, 'mod_itr': itr},
                'SRx4': {'mod_dir': path, 'mod_itr': itr},
                'SRx8': {'mod_dir': path, 'mod_itr': itr}
            }
        print_run_time (bool; default=True): print timing info
        batch_size (int; default=200): inference batch size
    Returns:
        SRx2_pred (arr): SRx2 predicted patches
        SRx4_pred (arr): SRx4 predicted patches
        SRx8_pred (arr): SRx8 predicted patches
    """

    if torch.cuda.is_available():
        torch_devs = torch.device("cuda")
    elif torch.backends.mps.is_available():
        torch_devs = torch.device("mps")
    else:
        torch_devs = torch.device("cpu")

    if print_run_time: print("Loading models...", end="", flush=True)
    t0 = time.time()
    x2mod, x2mod_args, x2mod_ch = load_trained_CNNSR(mods_to_use["SRx2"]["mod_dir"], mods_to_use["SRx2"]["mod_itr"], torch_devs)
    x4mod, x4mod_args, x4mod_ch = load_trained_CNNSR(mods_to_use["SRx4"]["mod_dir"], mods_to_use["SRx4"]["mod_itr"], torch_devs)
    x8mod, x8mod_args, x8mod_ch = load_trained_CNNSR(mods_to_use["SRx8"]["mod_dir"], mods_to_use["SRx8"]["mod_itr"], torch_devs)
    t1 = time.time()
    if print_run_time: print(f"DONE. (Time taken = {round(t1-t0, 3)} sec)")

    # ---- SRx2 MODEL ----
    if print_run_time: print("Running SRx2 predictions...", end="", flush=True)
    X_in = patch_x1[:, x2mod_ch, :, :]
    del patch_x1

    X = np.zeros(shape=(X_in.shape[0], len(x2mod_ch), X_in.shape[2] * 2, X_in.shape[3] * 2))
    for i in range(len(X)):
        Xi_upscaled = upscale(X_in[i, 0, :, :], srfac_in=1, srfac_out=2)
        X[i, 0, :, :] = Xi_upscaled / np.max(Xi_upscaled)

    del X_in, Xi_upscaled

    n_patches = len(X)
    n_batches = n_patches // batch_size

    for i in range(0, n_batches + 1):
        i_s, i_f = i * batch_size, min((i + 1) * batch_size, n_patches)
        if i_s < n_patches:
            X_batch = torch.from_numpy(X[i_s:i_f].astype(np.float32)).to(torch_devs)
            SRx2_pred_batch = x2mod.forward(X_batch).detach().cpu().numpy()
            if i == 0: SRx2_pred = deepcopy(SRx2_pred_batch)
            else: SRx2_pred = np.append(SRx2_pred, SRx2_pred_batch, axis=0)

    del X, X_batch, SRx2_pred_batch

    SRx2_pred = SRx2_pred / np.max(SRx2_pred, axis=(2, 3), keepdims=True)
    t2 = time.time()
    if print_run_time: print(f"DONE. (Time taken = {round(t2-t1, 3)} sec)")

    # ---- SRx4 MODEL ----
    if print_run_time: print("Running SRx4 predictions...", end="", flush=True)
    X = np.zeros(shape=(SRx2_pred.shape[0], len(x4mod_ch), SRx2_pred.shape[2] * 2, SRx2_pred.shape[3] * 2))
    for i in range(len(X)):
        Xi_upscaled = upscale(SRx2_pred[i, 0, :, :], srfac_in=2, srfac_out=4)
        X[i, 0, :, :] = Xi_upscaled / np.max(Xi_upscaled)

    del Xi_upscaled

    for i in range(0, n_batches + 1):
        i_s, i_f = i * batch_size, min((i + 1) * batch_size, n_patches)
        if i_s < n_patches:
            X_batch = torch.from_numpy(X[i_s:i_f].astype(np.float32)).to(torch_devs)
            SRx4_pred_batch = x4mod.forward(X_batch).detach().cpu().numpy()
            if i == 0: SRx4_pred = deepcopy(SRx4_pred_batch)
            else: SRx4_pred = np.append(SRx4_pred, SRx4_pred_batch, axis=0)

    del X, X_batch, SRx4_pred_batch

    SRx4_pred = SRx4_pred / np.max(SRx4_pred, axis=(2, 3), keepdims=True)
    t3 = time.time()
    if print_run_time: print(f"DONE. (Time taken = {round(t3-t2, 3)} sec)")

    # ---- SRx8 MODEL ----
    if print_run_time: print("Running SRx8 predictions...", end="", flush=True)
    X = np.zeros(shape=(SRx4_pred.shape[0], len(x8mod_ch), SRx4_pred.shape[2] * 2, SRx4_pred.shape[3] * 2))
    for i in range(len(X)):
        Xi_upscaled = upscale(SRx4_pred[i, 0, :, :], srfac_in=4, srfac_out=8)
        X[i, 0, :, :] = Xi_upscaled / np.max(Xi_upscaled)

    del Xi_upscaled

    for i in range(0, n_batches + 1):
        i_s, i_f = i * batch_size, min((i + 1) * batch_size, n_patches)
        if i_s < n_patches:
            X_batch = torch.from_numpy(X[i_s:i_f].astype(np.float32)).to(torch_devs)
            SRx8_pred_batch = x8mod.forward(X_batch).detach().cpu().numpy()
            if i == 0: SRx8_pred = deepcopy(SRx8_pred_batch)
            else: SRx8_pred = np.append(SRx8_pred, SRx8_pred_batch, axis=0)

    del X, X_batch, SRx8_pred_batch

    SRx8_pred = SRx8_pred / np.max(SRx8_pred, axis=(2, 3), keepdims=True)
    t4 = time.time()
    if print_run_time: print(f"DONE. (Time taken = {round(t4-t3, 3)} sec)")

    if print_run_time:
        print("-----------")
        print(f"TOTAL TIME TAKEN = {round(t4-t0, 3)} sec")
        print("--------------------------\n")

    return (SRx2_pred, SRx4_pred, SRx8_pred)


# ----------------------------
def predict_CNNSR_singleMod(patch_x1, mods_to_use, print_run_time=True, batch_size=200):
    """Predict SR patches at x8 using a single x1→x8 pre-trained model.
    Args:
        patch_x1 (arr): input patches at native x1 resolution (N, C, H, W)
        mods_to_use (dict): path to the pre-trained SRx8 model, e.g.:
            {'SRx8': {'mod_dir': path, 'mod_itr': itr}}
        print_run_time (bool; default=True): print timing info
        batch_size (int; default=200): inference batch size
    Returns:
        SRx8_pred (arr): SRx8 predicted patches
    """

    if torch.cuda.is_available():
        torch_devs = torch.device("cuda")
    elif torch.backends.mps.is_available():
        torch_devs = torch.device("mps")
    else:
        torch_devs = torch.device("cpu")

    t0 = time.time()
    if print_run_time: print("Loading models...", end="", flush=True)
    x8mod, x8mod_args, x8mod_ch = load_trained_CNNSR(mods_to_use["SRx8"]["mod_dir"], mods_to_use["SRx8"]["mod_itr"], torch_devs)
    t1 = time.time()
    if print_run_time: print(f"DONE. (Time taken = {round(t1-t0, 3)} sec)")

    if print_run_time: print("Running SRx8 predictions...", end="", flush=True)
    X_in = patch_x1[:, x8mod_ch, :, :]
    del patch_x1

    X = np.zeros(shape=(X_in.shape[0], len(x8mod_ch), X_in.shape[2] * 8, X_in.shape[3] * 8))
    for i in range(len(X)):
        Xi_upscaled = upscale(X_in[i, 0, :, :], srfac_in=1, srfac_out=8)
        X[i, 0, :, :] = Xi_upscaled / np.max(Xi_upscaled)

    del Xi_upscaled

    n_patches = len(X)
    n_batches = n_patches // batch_size

    for i in range(0, n_batches + 1):
        i_s, i_f = i * batch_size, min((i + 1) * batch_size, n_patches)
        if i_s < n_patches:
            X_batch = torch.from_numpy(X[i_s:i_f].astype(np.float32)).to(torch_devs)
            SRx8_pred_batch = x8mod.forward(X_batch).detach().cpu().numpy()
            if i == 0: SRx8_pred = deepcopy(SRx8_pred_batch)
            else: SRx8_pred = np.append(SRx8_pred, SRx8_pred_batch, axis=0)

    del X, X_batch, SRx8_pred_batch

    SRx8_pred = SRx8_pred / np.max(SRx8_pred, axis=(2, 3), keepdims=True)
    t2 = time.time()
    if print_run_time: print(f"DONE. (Time taken = {round(t2-t1, 3)} sec)")

    if print_run_time:
        print("-----------")
        print(f"TOTAL TIME TAKEN = {round(t2-t0, 3)} sec")
        print("--------------------------\n")

    return SRx8_pred


# ----------------------------
def err_from_log(log_filepath):
    """Extract training and validation L2 norm error from a training log file.
    Args:
        log_filepath (str): path to log file
    Returns:
        train_l2norm_avg (list): training L2 norm error per epoch
        valid_l2norm_avg (list): validation L2 norm error per epoch
    """

    log_content = open(log_filepath, "r").read()
    train_l2norm_avg, valid_l2norm_avg = [], []

    for line in log_content.split("\n"):
        if "INFO:root:[Train] @ " in line:
            train_l2norm_avg.append(float(line.split("Avg.: ")[1].split(",")[0]))
        if "INFO:root:[Valid] @ " in line:
            valid_l2norm_avg.append(float(line.split("Avg.: ")[1].split(",")[0]))

    return train_l2norm_avg, valid_l2norm_avg
