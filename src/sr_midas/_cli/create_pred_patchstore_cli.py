"""CLI for creating a predicted patchstore HDF5 from a pre-trained CNNSR model.

Refactored from: SRML-HEDM/CNNSR/create_pred_patchstore_hdf5.py
"""

import argparse
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import h5py

from sr_midas.models.cnnsr import load_trained_CNNSR
from sr_midas.data.patchstore import load_patchstore_h5data
from sr_midas.data.upscale import upscale

SEP = os.sep


def main():
    parser = argparse.ArgumentParser(
        description="Create a patchstore HDF5 file with predicted patches from a "
                    "pre-trained CNNSR model."
    )
    parser.add_argument("-pstPath", type=str, required=True,
                        help="Path to input patchstore")
    parser.add_argument("-saveDir", type=str, required=True,
                        help="Path to directory where output patchstore will be saved")
    parser.add_argument("-saveName", type=str, required=True,
                        help="Filename for output patchstore")
    parser.add_argument("-trainedModDir", type=str, required=True,
                        help="Path to directory containing pre-trained CNNSR model")
    parser.add_argument("-trainedModItr", type=str, required=True,
                        help="Iteration of pre-trained CNNSR model")
    parser.add_argument("-srfacIn", type=int, required=True,
                        help="Super-resolution factor of input patchstore")
    parser.add_argument("-srfacOut", type=int, required=True,
                        help="Super-resolution factor of predicted patchstore")
    parser.add_argument("-bsz", type=int, default=500,
                        help="Batch size for prediction")
    args = parser.parse_args()

    print("Selecting device...", end="", flush=True)
    if torch.cuda.is_available():
        torch_devs = torch.device("cuda")
    elif torch.backends.mps.is_available():
        torch_devs = torch.device("mps")
    else:
        torch_devs = torch.device("cpu")
    print(f"DONE. Device: {torch_devs}", flush=True)

    print("Loading super-resolution model...", end="", flush=True)
    sr_mod, sr_mod_args, sr_mod_ch = load_trained_CNNSR(
        mod_dir=args.trainedModDir, mod_itr=args.trainedModItr, torch_devs=torch_devs
    )
    print("DONE.", flush=True)

    print("Loading input patchstore...", end="", flush=True)
    patch_arr, *_ = load_patchstore_h5data(args.pstPath)
    X_in = patch_arr[f"SRx{args.srfacIn}"]
    print("DONE.", flush=True)

    print("Upscaling input patchstore...", end="", flush=True)
    X_in = X_in[:, sr_mod_ch, :, :]
    upscale_factor = int(args.srfacOut / args.srfacIn)
    X = np.zeros(shape=(X_in.shape[0], len(sr_mod_ch),
                         X_in.shape[2] * upscale_factor,
                         X_in.shape[3] * upscale_factor))

    for i in range(len(X)):
        Xi_upscaled = upscale(X_in[i, 0, :, :], args.srfacIn, args.srfacOut)
        X[i, 0, :, :] = Xi_upscaled / np.max(Xi_upscaled)

    del X_in
    print("DONE.", flush=True)

    n_patches = len(X)
    batch_size = args.bsz
    n_batches = n_patches // batch_size

    print("Making predictions...", end="", flush=True)
    SRx_pred = None
    for i in range(n_batches + 1):
        i_s, i_f = i * batch_size, min((i + 1) * batch_size, n_patches)
        if i_s < n_patches:
            X_batch = torch.from_numpy(X[i_s:i_f].astype(np.float32)).to(torch_devs)
            SRx_pred_batch = sr_mod.forward(X_batch).detach().cpu().numpy()
            if SRx_pred is None:
                SRx_pred = deepcopy(SRx_pred_batch)
            else:
                SRx_pred = np.append(SRx_pred, SRx_pred_batch, axis=0)
    print("DONE.", flush=True)

    del X

    if not os.path.exists(args.saveDir):
        os.mkdir(args.saveDir)
        print(f"Directory created: {args.saveDir}", flush=True)

    print("Compiling patchstore as HDF5 file...", end="", flush=True)
    pred_pst_savepath = f"{args.saveDir}{SEP}{args.saveName}"
    with h5py.File(pred_pst_savepath, "w") as h5file:
        group = h5file.create_group("patchArr")
        group.create_dataset(f"SRx{args.srfacOut}", data=SRx_pred, dtype=np.float32)
    print("DONE.", flush=True)
    print(f"Saved: {pred_pst_savepath}", flush=True)


if __name__ == "__main__":
    main()
