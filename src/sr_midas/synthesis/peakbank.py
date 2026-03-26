"""Peakbank creation from MIDAS-fitted peaks.

Refactored from: SIMR-xrd/SIMR_xrd/create_peakbank.py
"""

import os
import json

import numpy as np
import pandas as pd

from sr_midas.data.patchstore import midas_Zarr_zip
from sr_midas.synthesis import peak_artist


# ----------------------------
def create_peakbank(config):
    """Create a peakbank of MIDAS-fitted peaks.
    Loads peak shapes, reconstructs each peak, computes reconstruction error,
    filters by threshold, and saves the result as a CSV file.
    Args:
        config (dict or str): configuration dict or path to JSON config file with keys:
            midas_dir (list of str): list of paths to MIDAS data directories
            peakbank_savedir (str): directory to save the peakbank CSV
            peakbank_savename (str): filename for the peakbank CSV
            peak_recon_err_threshold (float or None): max allowed reconstruction error
            cvsz (int): canvas size for patch reconstruction
            srfac (int): super-res factor for patch reconstruction
            I_thresh (float): intensity threshold
            save_exp_patches (bool): whether to save experimental patches
            dir_ignore (list): directory names to ignore
            save_frame_gen (bool): whether to save synthetic frames
    Returns:
        df_peakbank (pandas DataFrame): filtered peakbank
    """

    if isinstance(config, str):
        with open(config, "r") as f:
            config = json.load(f)

    midas_dir_list = config["midas_dir"]
    peakbank_savedir = config["peakbank_savedir"]
    peakbank_savename = config["peakbank_savename"]
    peak_recon_err_threshold = config["peak_recon_err_threshold"]
    cvsz = config["cvsz"]
    srfac = config["srfac"]
    I_thresh = config["I_thresh"]
    save_exp_patches = config["save_exp_patches"]
    dir_ignore = config["dir_ignore"]
    save_frame_gen = config["save_frame_gen"]

    if str(save_frame_gen).lower() in ["true", "True", "t", "1"]:
        frame_gen_savedir = os.path.join(peakbank_savedir, "frame_gen")
        if not os.path.exists(frame_gen_savedir):
            os.mkdir(frame_gen_savedir)

    print("--- CREATING PEAKBANK USING MIDAS-FITTED PEAKS ---", flush=True)
    df_peakbank = pd.DataFrame()

    for sample_dir in midas_dir_list:

        df_sample_peaks = pd.DataFrame()

        print("Midas data directory running: ", sample_dir)
        midas_peaks_csv_dir = os.path.join(sample_dir, "Temp")
        midas_zip_filename = [i for i in os.listdir(sample_dir) if i.endswith(".MIDAS.zip")][0]
        midas_zip_filepath = os.path.join(sample_dir, midas_zip_filename)

        csv_filenames = sorted([i for i in os.listdir(midas_peaks_csv_dir) if i.endswith(".csv")])

        zf, exData, exDark, exBright = midas_Zarr_zip(midas_zip_filepath)
        exData[exData < I_thresh] = 0

        zf_process_params = zf["analysis"]["process"]["analysis_parameters"]
        Ypx_BC = zf_process_params["YCen"][0]
        Zpx_BC = zf_process_params["ZCen"][0]

        ImTransOpt = 0
        if "ImTransOpt" in zf_process_params:
            ImTransOpt = int(zf_process_params["ImTransOpt"][0])

        Padding = 6
        if "Padding" in zf_process_params:
            Padding = int(zf_process_params["Padding"][0])

        SkipFrame = 0
        if "SkipFrame" in zf_process_params:
            SkipFrame = int(zf_process_params["SkipFrame"][0])

        exData = exData[SkipFrame:]
        nr_frames = exData.shape[0]

        for frame_i in range(0, nr_frames):

            frame_exp = exData[frame_i]

            if ImTransOpt == 1: frame_exp = np.flip(frame_exp, axis=1)
            if ImTransOpt == 2: frame_exp = np.flip(frame_exp, axis=0)
            if ImTransOpt == 3: frame_exp = np.transpose(frame_exp)

            csv_fn = [i for i in csv_filenames if i.endswith(f"{str(frame_i+1).zfill(Padding)}_PS.csv")][0]
            midas_peaks_csv_filepath = os.path.join(midas_peaks_csv_dir, csv_fn)
            df_frame_peaks = pd.read_csv(midas_peaks_csv_filepath, delimiter="\t")

            spot_max_peaks = 1
            df_frame_peaks = df_frame_peaks[df_frame_peaks["nPeaks"] <= spot_max_peaks].reset_index(drop=True)

            Eta_exclude = 3.0
            df_frame_peaks = df_frame_peaks[abs(df_frame_peaks["Eta(degrees)"]) - Eta_exclude > 0].reset_index(drop=True)
            df_frame_peaks = df_frame_peaks[abs(df_frame_peaks["Eta(degrees)"]) < (180 - Eta_exclude)].reset_index(drop=True)

            if len(df_frame_peaks) >= 1:
                frame_gen = peak_artist.draw_detector_frame(df_frame_peaks, Ypx_BC, Zpx_BC,
                                                            frame_shape=frame_exp.shape,
                                                            cvsz=int(cvsz), srfac=int(srfac), I_thresh=I_thresh)

                if str(save_frame_gen).lower() in ["true", "True", "t", "1"]:
                    frame_gen_filepath = os.path.join(frame_gen_savedir, f"frame_gen_{str(frame_i).zfill(6)}.npy")
                    np.save(frame_gen_filepath, frame_gen)

                peak_recon_err = peak_artist.peak_reconstruction_err(df_frame_peaks, frame_exp, frame_gen, cvsz=int(cvsz))

                nr_peaks_frame = len(df_frame_peaks)
                df_frame_peaks.insert(loc=1, column="frame_id", value=[str(frame_i).zfill(6)] * nr_peaks_frame)
                df_frame_peaks.insert(loc=1, column="error_reconstruction", value=peak_recon_err)

                df_sample_peaks = pd.concat([df_sample_peaks, df_frame_peaks]).reset_index(drop=True)

        df_peakbank = pd.concat([df_peakbank, df_sample_peaks]).reset_index(drop=True)

    if peak_recon_err_threshold not in [None, "None"]:
        df_peakbank = df_peakbank[df_peakbank["error_reconstruction"] < peak_recon_err_threshold].reset_index(drop=True)

    if not os.path.exists(peakbank_savedir):
        os.mkdir(peakbank_savedir)

    save_path = os.path.join(peakbank_savedir, peakbank_savename)
    print("Saving peakbank at: ", f"'{save_path}' ... ", end="", flush=True)
    df_peakbank.to_csv(save_path)
    print("DONE!\n\n", flush=True)

    return df_peakbank
