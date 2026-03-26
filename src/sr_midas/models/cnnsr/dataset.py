"""PyTorch Dataset class for CNN super-resolution training.

Source: SRML-HEDM/CNNSR/dataset.py (trainData_CNNSR class)
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from sr_midas.data.patchstore import load_patchstore_h5data
from sr_midas.data.upscale import upscale


# ----------------------------
class trainData_CNNSR(Dataset):
    """PyTorch Dataset for loading patchstore data for CNNSR training."""

    def __init__(self, pst_path, srfac_out, srfac_in,
                 use_R_channel=False, use_Eta_channel=False,
                 normalize_R_channel=True, normalize_Eta_channel=True,
                 use="train", train_frac=0.8,
                 pst_path_X=None, pst_path_Y=None):
        """Initialization function.
        Args:
            pst_path (str): path to the .h5 patchstore file
            srfac_out (int): SR factor of model output patch
            srfac_in (int): SR factor of model input patch
            use_R_channel (str/bool; default=False): include R-channel in input
            use_Eta_channel (str/bool; default=False): include Eta-channel in input
            normalize_R_channel (bool; default=True): normalize R channel
            normalize_Eta_channel (bool; default=True): normalize Eta channel
            use (str; default="train"): "train" or "test"
            train_frac (float; default=0.8): fraction of data for training
            pst_path_X (str or None): input patchstore path (if different from pst_path)
            pst_path_Y (str or None): target patchstore path (if different from pst_path)
        """

        self.pst_path = pst_path
        self.srfac_out = srfac_out
        self.srfac_in = srfac_in
        self.use = use
        self.train_frac = train_frac
        self.use_R_channel = use_R_channel
        self.use_Eta_channel = use_Eta_channel
        self.normalize_R_channel = normalize_R_channel
        self.normalize_Eta_channel = normalize_Eta_channel

        Y_channels = [0]
        X_channels = [0]
        if self.use_R_channel in ["True", "true", True, "Yes", "yes", "Y", "y"]:
            X_channels.append(1)
        if self.use_Eta_channel in ["True", "true", True, "Yes", "yes", "Y", "y"]:
            X_channels.append(2)

        nr_channels_in_X = len(X_channels)

        if (pst_path_Y in [None, "None", "none"]) and (pst_path_X in [None, "None", "none"]):
            self.pst_path_Y, self.pst_path_X = pst_path, pst_path
        else:
            if pst_path_Y is None: self.pst_path_Y = pst_path
            else: self.pst_path_Y = pst_path_Y
            if pst_path_X is None: self.pst_path_X = pst_path
            else: self.pst_path_X = pst_path_X

        Y = np.array(load_patchstore_h5data(self.pst_path_Y, only_patch_arrays=True)[f"SRx{srfac_out}"])
        X_in = np.array(load_patchstore_h5data(self.pst_path_X, only_patch_arrays=True)[f"SRx{srfac_in}"])

        X = np.zeros(shape=(X_in.shape[0], nr_channels_in_X, Y.shape[2], Y.shape[3]))

        for i in range(len(X)):
            Xi_upscaled = upscale(X_in[i, 0, :, :], self.srfac_in, self.srfac_out)
            X[i, 0, :, :] = Xi_upscaled / np.max(Xi_upscaled)

        if nr_channels_in_X >= 2:
            X[:, 1, :, :] = Y[:, 1, :, :]
            X[:, 2, :, :] = Y[:, 2, :, :]

            ind_zeroI = np.where(X[:, [0], :, :] == 0)
            X[ind_zeroI[0], 1, ind_zeroI[2], ind_zeroI[3]] = 0
            X[ind_zeroI[0], 2, ind_zeroI[2], ind_zeroI[3]] = 0

            masked_X = np.ma.masked_equal(X, 0)

            if normalize_R_channel in ["True", "true", True, "Yes", "yes", "Y", "y"]:
                max_arr_R = np.max(masked_X[:, [1], :, :], axis=(2, 3), keepdims=True)
                min_arr_R = np.min(masked_X[:, [1], :, :], axis=(2, 3), keepdims=True)
                X[:, [1], :, :] = (masked_X[:, [1], :, :] - min_arr_R) / (max_arr_R - min_arr_R)

            if normalize_Eta_channel in ["True", "true", True, "Yes", "yes", "Y", "y"]:
                max_arr_Eta = np.max(np.abs(masked_X[:, [2], :, :]), axis=(2, 3), keepdims=True)
                min_arr_Eta = np.min(np.abs(masked_X[:, [2], :, :]), axis=(2, 3), keepdims=True)
                X[:, [2], :, :] = (np.abs(masked_X[:, [2], :, :]) - min_arr_Eta) / (max_arr_Eta - min_arr_Eta)

            X = X[:, X_channels, :, :]

        X = torch.from_numpy(X.astype(np.float32))

        Y = Y[:, Y_channels, :, :]
        Y = torch.from_numpy(Y.astype(np.float32))

        if use in ["train", "Train", "TRAIN"]:
            si, fi = 0, int(self.train_frac * len(Y))
        elif use in ["test", "Test", "TEST", "val", "Val"]:
            si, fi = int(self.train_frac * len(Y)), None
        else:
            print(f"Error: use='{self.use}' is not supported. Accepted: ['train', 'test']")

        self.X = X[si:fi]
        self.Y = Y[si:fi]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
