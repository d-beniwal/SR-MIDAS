"""Synthetic patchstore creation at multiple resolutions.

Refactored from: SIMR-xrd/SIMR_xrd/create_patchstore.py
"""

import os
import json
import time

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from scipy.signal import find_peaks
from sklearn.cluster import MeanShift, estimate_bandwidth

from sr_midas.physics import coord_transform
from sr_midas.synthesis import patch_methods, peak_artist
from sr_midas.data.patchstore import df_to_sarray


# ----------------------------
def create_patchstore(args):
    """Create a synthetic patchstore with peaks at multiple resolutions.
    Args:
        args: namespace or dict with keys matching the CLI arguments:
            peakbank (str): path to peakbank CSV
            saveName (str): output filename (.h5)
            saveDir (str): output directory
            nPatch (int): number of patches
            lrsz (int): low-resolution patch size
            cvsz (int): canvas size for initial drawing
            srfacSource (int): super-res factor for source patch
            srfacAll (str): all SR factors e.g. "1-2-4-8"
            nPeak (str): possible peak counts e.g. "1-2-3-4-5"
            pSepMin (float): minimum peak separation (pixels)
            pSepMax (float): maximum peak separation (pixels)
            varR (float): max R variation from ring radius
            errCut (float): reconstruction error cutoff
            integIntCut (float): integrated intensity cutoff
            midasIthresh (float): MIDAS intensity threshold
            peakImin (float): minimum peak intensity
            peakImax (float): maximum peak intensity
            Ypx_BC (float): Beam Center Y pixel coordinate
            Zpx_BC (float): Beam Center Z pixel coordinate
            srIthreshFac (float): source patch threshold factor
            config (str or None): optional JSON config path to override args
    """

    if isinstance(args, dict):
        import argparse
        ns = argparse.Namespace(**args)
        args = ns

    # Merge JSON config if provided
    pst_args = vars(args).copy()
    if getattr(args, 'config', None):
        with open(args.config, "r") as f:
            config_data = json.load(f)
        for key, value in config_data.items():
            if key in pst_args:
                pst_args[key] = value
            else:
                print(f"Warning: Unrecognized key '{key}' in JSON config. Ignoring.")

    t0 = time.time()

    n_peaks_possible = [int(i) for i in pst_args["nPeak"].split("-")]
    srfac_list = [int(i) for i in pst_args["srfacAll"].split("-")]

    Ypx_BC, Zpx_BC = pst_args["Ypx_BC"], pst_args["Zpx_BC"]

    pst_dir = pst_args["saveDir"]
    if not os.path.exists(pst_dir):
        os.mkdir(pst_dir)
        print(f"Directory created: {pst_dir}", flush=True)

    pst_savepath = os.path.join(pst_dir, pst_args["saveName"])

    print(f"Loading & filtering peakbank ({pst_args['peakbank']})...", end="", flush=True)
    df_peakbank = pd.read_csv(pst_args["peakbank"])
    df_peakbank = df_peakbank[df_peakbank["error_reconstruction"] < pst_args["errCut"]]
    df_peakbank = df_peakbank[df_peakbank["IntegratedIntensity"] > pst_args["integIntCut"]]
    df_peakbank = df_peakbank[
        (df_peakbank["SigmaGR"] / df_peakbank["SigmaGEta"] < 10) &
        (df_peakbank["SigmaGR"] / df_peakbank["SigmaGEta"] > (1 / 10)) &
        (df_peakbank["SigmaLR"] / df_peakbank["SigmaLEta"] < 10) &
        (df_peakbank["SigmaLR"] / df_peakbank["SigmaLEta"] > (1 / 10)) &
        (df_peakbank["SigmaGR"] > 0.1) & (df_peakbank["SigmaGR"] < 2) &
        (df_peakbank["SigmaLR"] > 0.1) & (df_peakbank["SigmaLR"] < 2) &
        (df_peakbank["SigmaGEta"] > 0.1) & (df_peakbank["SigmaGEta"] < 2) &
        (df_peakbank["SigmaLEta"] > 0.1) & (df_peakbank["SigmaLEta"] < 2)
    ]
    print("DONE.", flush=True)

    print("Finding ring radii from peakbank...", end="", flush=True)
    min_points_per_cluster = max(n_peaks_possible)
    peaks_ringR = np.array(df_peakbank["Radius(px)"])
    if peaks_ringR.ndim == 1:
        peaks_ringR = peaks_ringR.reshape(-1, 1)

    bandwidth = estimate_bandwidth(peaks_ringR, quantile=0.1)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(peaks_ringR)

    initial_labels = ms.labels_
    initial_centers = ms.cluster_centers_

    unique_labels, cluster_counts = np.unique(initial_labels, return_counts=True)
    valid_cluster_labels = unique_labels[cluster_counts >= min_points_per_cluster]
    valid_cluster_centers = initial_centers[valid_cluster_labels]

    nr_rings = len(valid_cluster_labels)
    ringR = valid_cluster_centers.flatten().tolist()
    print("DONE.", flush=True)
    print(f"Ring radii identified: {ringR}", flush=True)

    print("Randomly selecting peak counts & ring radii for patches...", end="", flush=True)
    pst_nPeaks = np.random.choice(n_peaks_possible, pst_args["nPatch"])
    pst_RingR = np.random.choice(ringR, pst_args["nPatch"])
    print("DONE.", flush=True)

    print("Creating empty placeholders...", end="", flush=True)
    pst_patchSumI = np.full(pst_nPeaks.shape, np.nan)
    peaks_data_shape = (pst_args["nPatch"], max(n_peaks_possible))

    peaks_parameters_dict = {
        "Ypx": np.full(peaks_data_shape, np.nan),
        "Zpx": np.full(peaks_data_shape, np.nan),
        "Rpx": np.full(peaks_data_shape, np.nan),
        "EtaDeg": np.full(peaks_data_shape, np.nan),
        "sigGR": np.full(peaks_data_shape, np.nan),
        "sigGEta": np.full(peaks_data_shape, np.nan),
        "sigLR": np.full(peaks_data_shape, np.nan),
        "sigLEta": np.full(peaks_data_shape, np.nan),
        "IMax": np.full(peaks_data_shape, np.nan),
        "LGmix": np.full(peaks_data_shape, np.nan),
        "IntegInt": np.full(peaks_data_shape, np.nan),
        "BG": np.full(peaks_data_shape, np.nan),
        "peakPointer": np.full(peaks_data_shape, np.nan)
    }
    print("DONE.", flush=True)

    print("\nFinding peak positions from peakbank sampling...", flush=True)
    for (nPeaks, patchRingR, patch_i) in tqdm(zip(pst_nPeaks, pst_RingR, range(pst_args["nPatch"]))):

        df_forSampling = df_peakbank[
            (df_peakbank["Radius(px)"] < (patchRingR + pst_args["varR"])) &
            (df_peakbank["Radius(px)"] > (patchRingR - pst_args["varR"]))
        ]

        print(f"RingR: {patchRingR} | Sampling {nPeaks} peaks from {len(df_forSampling)} peaks", flush=True)
        df_sampledPeaks = df_forSampling.sample(nPeaks)
        df_sampledPeaks = df_sampledPeaks.sort_values(by=["IMax"], ascending=False)

        peaks_parameters_dict["peakPointer"][patch_i][:nPeaks] = (df_sampledPeaks.index).tolist()
        peaks_parameters_dict["sigGR"][patch_i][:nPeaks] = df_sampledPeaks["SigmaGR"].tolist()
        peaks_parameters_dict["sigGEta"][patch_i][:nPeaks] = df_sampledPeaks["SigmaGEta"].tolist()
        peaks_parameters_dict["sigLR"][patch_i][:nPeaks] = df_sampledPeaks["SigmaLR"].tolist()
        peaks_parameters_dict["sigLEta"][patch_i][:nPeaks] = df_sampledPeaks["SigmaLEta"].tolist()
        peaks_IMax = df_sampledPeaks["IMax"].tolist()
        peaks_parameters_dict["IMax"][patch_i][:nPeaks] = [max(min(pst_args["peakImax"], i), pst_args["peakImin"]) for i in peaks_IMax]
        peaks_parameters_dict["LGmix"][patch_i][:nPeaks] = df_sampledPeaks["MU"].tolist()
        peaks_parameters_dict["IntegInt"][patch_i][:nPeaks] = df_sampledPeaks["IntegratedIntensity"].tolist()
        peaks_parameters_dict["BG"][patch_i][:nPeaks] = df_sampledPeaks["BG"].tolist()

        pst_patchSumI[patch_i] = float(np.sum(df_sampledPeaks["IntegratedIntensity"]))

        peak1R = np.random.uniform(patchRingR - pst_args["varR"], patchRingR + pst_args["varR"])
        peak1Eta = np.random.choice([np.random.uniform(-177, -3), np.random.uniform(3, 177)])
        peak1Ypx, peak1Zpx = coord_transform.YZ_from_REta(peak1R, peak1Eta, Ypx_BC, Zpx_BC)

        peakPlaceFinish = False

        if nPeaks == 1:
            peaks_R, peaks_Eta, peaks_Ypx, peaks_Zpx = [peak1R], [peak1Eta], [peak1Ypx], [peak1Zpx]
            peakPlaceFinish = True
        else:
            while peakPlaceFinish == False:
                peaks_R, peaks_Eta, peaks_Ypx, peaks_Zpx = [peak1R], [peak1Eta], [peak1Ypx], [peak1Zpx]
                peaksDist = []
                for peak_i_inner in range(1, nPeaks):
                    peak_i_Rcheck = False
                    while peak_i_Rcheck == False:
                        peak_i_Ypx = np.random.uniform(peak1Ypx - pst_args["pSepMax"], peak1Ypx + pst_args["pSepMax"])
                        peak_i_Zpx = np.random.uniform(peak1Zpx - pst_args["pSepMax"], peak1Zpx + pst_args["pSepMax"])
                        peak_i_R, peak_i_Eta = coord_transform.REta_from_YZ(peak_i_Ypx, peak_i_Zpx, Ypx_BC, Zpx_BC)
                        if np.abs(peak_i_R - patchRingR) < pst_args["varR"]:
                            peak_i_Rcheck = True
                            peaks_Ypx.append(peak_i_Ypx)
                            peaks_Zpx.append(peak_i_Zpx)
                            peaks_R.append(peak_i_R)
                            peaks_Eta.append(peak_i_Eta)

                for i in range(0, nPeaks - 1):
                    for j in range(i + 1, nPeaks):
                        dist_ij = ((peaks_Ypx[i] - peaks_Ypx[j])**2 + (peaks_Zpx[i] - peaks_Zpx[j])**2)**0.5
                        peaksDist.append(dist_ij)

                if (np.min(peaksDist) > pst_args["pSepMin"]) & (np.max(peaksDist) < pst_args["pSepMax"]):
                    peakPlaceFinish = True

        peaks_parameters_dict["Ypx"][patch_i][:nPeaks] = peaks_Ypx
        peaks_parameters_dict["Zpx"][patch_i][:nPeaks] = peaks_Zpx
        peaks_parameters_dict["Rpx"][patch_i][:nPeaks] = peaks_R
        peaks_parameters_dict["EtaDeg"][patch_i][:nPeaks] = peaks_Eta

    print(f"\nDrawing normalized patches at SR factors: {srfac_list}...", flush=True)

    pst_Y00 = np.full(pst_nPeaks.shape, np.nan)
    pst_Z00 = np.full(pst_nPeaks.shape, np.nan)

    SRx_src_size = int(pst_args["srfacSource"] * pst_args["lrsz"])

    patch_arr_dict = {}
    peaks_loc_in_patch_dict = {}

    for xi in srfac_list:
        patch_arr_dict[f"SRx{xi}"] = np.zeros(shape=(pst_args["nPatch"], 3, int(xi * pst_args["lrsz"]), int(xi * pst_args["lrsz"])))
        peaks_loc_in_patch_dict[f"SRx{xi}"] = {
            "Ypx": np.full(peaks_data_shape, np.nan),
            "Zpx": np.full(peaks_data_shape, np.nan)
        }

    SRx_thresholdInt = pst_args["srIthreshFac"] * (pst_args["midasIthresh"] / (pst_args["srfacSource"] ** 2))

    delete_patches_i = []

    for patch_i in tqdm(range(pst_args["nPatch"])):
        nPeaks = pst_nPeaks[patch_i]
        peaksBG_SRx = peaks_parameters_dict["BG"][patch_i][:nPeaks] / (pst_args["srfacSource"] ** 2)
        peaksIMax_SRx = peaks_parameters_dict["IMax"][patch_i][:nPeaks] / (pst_args["srfacSource"] ** 2)

        grid_YY, grid_ZZ, grid_RR, grid_EE = patch_methods.patch_grid_fromYZpos(
            peaks_parameters_dict["Ypx"][patch_i][0], peaks_parameters_dict["Zpx"][patch_i][0],
            pst_args["cvsz"], pst_args["srfacSource"], Ypx_BC, Zpx_BC)

        RE_coords = (grid_RR, grid_EE)
        SRx = peak_artist.draw_peaks_diffLGwidth(RE_coords,
                                                  R=peaks_parameters_dict["Rpx"][patch_i][:nPeaks],
                                                  Eta=peaks_parameters_dict["EtaDeg"][patch_i][:nPeaks],
                                                  sigGR=peaks_parameters_dict["sigGR"][patch_i][:nPeaks],
                                                  sigGE=peaks_parameters_dict["sigGEta"][patch_i][:nPeaks],
                                                  sigLR=peaks_parameters_dict["sigLR"][patch_i][:nPeaks],
                                                  sigLE=peaks_parameters_dict["sigLEta"][patch_i][:nPeaks],
                                                  LGmix=peaks_parameters_dict["LGmix"][patch_i][:nPeaks],
                                                  IMax=peaksIMax_SRx,
                                                  BG=peaksBG_SRx,
                                                  I_thresh=0)

        SRx = SRx * (pst_patchSumI[patch_i] / np.sum(SRx))

        if np.max(SRx) <= SRx_thresholdInt:
            delete_patches_i.append(patch_i)
        else:
            SRx[SRx < SRx_thresholdInt] = 0
            SRx = SRx * (pst_patchSumI[patch_i] / np.sum(SRx))

            SRx1 = patch_methods.downscale(SRx, srfac_in=pst_args["srfacSource"], srfac_out=1)
            SRx1[SRx1 < pst_args["midasIthresh"]] = 0
            SRx1 = SRx1 * (pst_patchSumI[patch_i] / np.sum(SRx1))

            SRx1_max_Zpx, SRx1_max_Ypx = patch_methods.max_px_loc(SRx1)

            rs = int(SRx1_max_Zpx * pst_args["srfacSource"] - int(SRx_src_size / 2))
            cs = int(SRx1_max_Ypx * pst_args["srfacSource"] - int(SRx_src_size / 2))
            rf, cf = rs + SRx_src_size, cs + SRx_src_size
            SRx_crop = SRx[rs:rf, cs:cf]

            if SRx_crop.size == 0:
                delete_patches_i.append(patch_i)
                continue

            pst_Y00[patch_i] = int(grid_YY[rs, cs])
            pst_Z00[patch_i] = int(grid_ZZ[rs, cs])

            for xi in srfac_list:
                SRxi = patch_methods.downscale(SRx_crop, srfac_in=pst_args["srfacSource"], srfac_out=xi)
                if np.max(SRxi) > (pst_args["midasIthresh"] / (xi ** 2)):
                    SRxi[SRxi < (pst_args["midasIthresh"] / (xi ** 2))] = 0
                SRxi = SRxi * (pst_patchSumI[patch_i] / np.sum(SRxi))
                SRxi_norm = SRxi / np.max(SRxi)
                patch_arr_dict[f"SRx{xi}"][patch_i][0] = SRxi_norm

                _, _, SRxi_RR, SRxi_EE = patch_methods.patch_grid_fromYZ00(
                    pst_Y00[patch_i], pst_Z00[patch_i], pst_args["lrsz"], xi, Ypx_BC, Zpx_BC)
                patch_arr_dict[f"SRx{xi}"][patch_i][1] = SRxi_RR
                patch_arr_dict[f"SRx{xi}"][patch_i][2] = SRxi_EE

                peaks_loc_SRxi_Ypx = (peaks_parameters_dict["Ypx"][patch_i][:nPeaks] - pst_Y00[patch_i]) * xi
                peaks_loc_SRxi_Zpx = (peaks_parameters_dict["Zpx"][patch_i][:nPeaks] - pst_Z00[patch_i]) * xi
                peaks_loc_in_patch_dict[f"SRx{xi}"]["Ypx"][patch_i][:nPeaks] = peaks_loc_SRxi_Ypx
                peaks_loc_in_patch_dict[f"SRx{xi}"]["Zpx"][patch_i][:nPeaks] = peaks_loc_SRxi_Zpx

                if xi == pst_args["srfacSource"]:
                    for i in range(0, nPeaks):
                        r1, r2 = int(peaks_loc_SRxi_Zpx[i] - 5), int(peaks_loc_SRxi_Zpx[i] + 5)
                        c1, c2 = int(peaks_loc_SRxi_Ypx[i] - 5), int(peaks_loc_SRxi_Ypx[i] + 5)
                        if np.sum(SRxi_norm[r1:r2, c1:c2]) < 0.1:
                            print(f"SRx{xi} | patch: {patch_i}; peak: {i} not drawn")

    # Delete blank patches
    for key in peaks_parameters_dict.keys():
        peaks_parameters_dict[key] = np.delete(peaks_parameters_dict[key], delete_patches_i, axis=0)

    pst_nPeaks = np.delete(pst_nPeaks, delete_patches_i, axis=0)
    pst_patchSumI = np.delete(pst_patchSumI, delete_patches_i, axis=0)
    pst_Y00 = np.delete(pst_Y00, delete_patches_i, axis=0)
    pst_Z00 = np.delete(pst_Z00, delete_patches_i, axis=0)

    for key in patch_arr_dict.keys():
        patch_arr_dict[key] = np.delete(patch_arr_dict[key], delete_patches_i, axis=0)

    for key1 in peaks_loc_in_patch_dict.keys():
        for key2 in peaks_loc_in_patch_dict[key1].keys():
            peaks_loc_in_patch_dict[key1][key2] = np.delete(peaks_loc_in_patch_dict[key1][key2], delete_patches_i, axis=0)

    df_patch_info = pd.DataFrame()
    df_patch_info["nPeaks"] = pst_nPeaks
    df_patch_info["ISum"] = pst_patchSumI
    df_patch_info["Y00"] = pst_Y00
    df_patch_info["Z00"] = pst_Z00

    if len(delete_patches_i) >= 1:
        print(f"{len(delete_patches_i)} patches deleted (blank after threshold)", flush=True)

    print("\nCompiling patchstore as HDF5 file...", end="", flush=True)
    with h5py.File(pst_savepath, "w") as h5file:

        group = h5file.create_group("patchArr")
        for key, array in patch_arr_dict.items():
            group.create_dataset(key, data=array, dtype=np.float32)

        group = h5file.create_group("peaksLocInPatch")
        for key1, dict1 in peaks_loc_in_patch_dict.items():
            subgroup = group.create_group(key1)
            for (key2, array) in dict1.items():
                subgroup.create_dataset(key2, data=array, dtype=np.float32)

        sa, saType = df_to_sarray(df_patch_info)
        h5file.create_dataset('patchInfo', data=sa, dtype=saType)

        group = h5file.create_group("peaksParameters")
        for key, array in peaks_parameters_dict.items():
            group.create_dataset(key, data=array, dtype=np.float32)

        group = h5file.create_group("pstCreationArgs")
        for key, value in pst_args.items():
            group.create_dataset(key, data=value)

    print("DONE.", flush=True)
    print(f"\n----- PATCHSTORE CREATION COMPLETED -----")
    print(f"Patchstore saved as: {pst_savepath}")
    print("-----------------------------------------\n\n")
