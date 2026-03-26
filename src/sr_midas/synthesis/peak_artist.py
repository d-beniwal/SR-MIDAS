"""Functions to draw synthetic detector patches and frames.

Source: SIMR-xrd/SIMR_xrd/utils/peak_artist.py
"""

import numpy as np

from sr_midas.physics import peaks2d
from sr_midas.synthesis import patch_methods


# ----------------------------
def draw_peaks_diffLGwidth(RE_coords, R, Eta, sigGR, sigGE, sigLR, sigLE, LGmix, IMax, BG=0.0, I_thresh=0):
    """Draw a patch with pseudo-Voigt peaks using different Gaussian and Lorentzian widths.
    Args:
        RE_coords (tuple): (RR, EE) coordinate grids (R in pixels, Eta in degrees)
        R (list of floats): peak positions - R coordinate
        Eta (list of floats): peak positions - Eta coordinate
        sigGR (list of floats): Gaussian peak widths along R
        sigGE (list of floats): Gaussian peak widths along Eta
        sigLR (list of floats): Lorentzian peak widths along R
        sigLE (list of floats): Lorentzian peak widths along Eta
        LGmix (list of floats): LGmix factors
        IMax (list of floats): Maximum peak intensities
        BG (float or list; default=0.0): Background intensities
        I_thresh (float; default=0): Intensity threshold for patch
    Returns:
        patch_I (arr): patch with all input peaks
    """

    grid_RR, grid_EE = RE_coords
    patch_I = np.zeros(shape=grid_RR.shape)

    n_peaks = len(R)

    if isinstance(BG, (int, float)):
        BG = [BG] * n_peaks

    for i in range(0, n_peaks):
        peak_i = peaks2d.pseudoVoigt2d_diffLGwidth(grid_RR, grid_EE,
                                                    R[i], Eta[i],
                                                    sigGR[i], sigGE[i],
                                                    sigLR[i], sigLE[i],
                                                    LGmix[i], IMax[i])

        peak_i[peak_i > 0.0] = peak_i[peak_i > 0.0] + BG[i]
        patch_I += peak_i

    if np.max(patch_I) > I_thresh:
        patch_I[patch_I <= I_thresh] = 0

    return (patch_I)


# ----------------------------
def draw_peaks_sameLGwidth(RE_coords, R, Eta, sigR, sigE, LGmix, IMax, BG=0.0, I_thresh=0):
    """Draw a patch with pseudo-Voigt peaks using the same Gaussian and Lorentzian widths.
    Args:
        RE_coords (tuple): (RR, EE) coordinate grids
        R (list of floats): peak positions - R coordinate
        Eta (list of floats): peak positions - Eta coordinate
        sigR (list of floats): peak widths along R
        sigE (list of floats): peak widths along Eta
        LGmix (list of floats): LGmix factors
        IMax (list of floats): Maximum peak intensities
        BG (float or list; default=0.0): Background intensities
        I_thresh (float; default=0): Intensity threshold for patch
    Returns:
        patch_I (arr): patch with all input peaks
    """

    grid_RR, grid_EE = RE_coords
    patch_I = np.zeros(shape=grid_RR.shape)

    n_peaks = len(R)

    if isinstance(BG, (int, float)):
        BG = [BG] * n_peaks

    for i in range(0, n_peaks):
        peak_i = peaks2d.pseudoVoigt2d_sameLGwidth(grid_RR, grid_EE,
                                                    R[i], Eta[i],
                                                    sigR[i], sigE[i],
                                                    LGmix[i], IMax[i])

        peak_i[peak_i > 0.0] = peak_i[peak_i > 0.0] + BG[i]
        patch_I += peak_i

    if np.max(patch_I) > I_thresh:
        patch_I[patch_I <= I_thresh] = 0

    return (patch_I)


# ----------------------------
def draw_detector_frame(df_frame_peaks, Ypx_BC, Zpx_BC, frame_shape=(2048, 2048), cvsz=int(10), srfac=int(1), I_thresh=0.0):
    """Draw a complete synthetic detector frame from all peak parameters.
    Args:
        df_frame_peaks (pandas df): dataframe with all peak parameters for the frame
        Ypx_BC (float): Beam Center Y pixel coordinate
        Zpx_BC (float): Beam Center Z pixel coordinate
        frame_shape (tuple; default=(2048,2048)): shape of detector frame
        cvsz (int; default=10): canvas size for drawing peaks
        srfac (int; default=1): super-res factor
        I_thresh (float; default=0.0): intensity threshold for final frame
    Returns:
        frame_gen (arr): synthetic frame
    """

    cvsz = int(10)
    srfac = int(1)

    frame_gen_shape = (frame_shape[0] * srfac, frame_shape[1] * srfac)
    frame_gen = np.zeros(shape=frame_gen_shape)

    nr_peaks = len(df_frame_peaks)

    for peak_i in range(0, nr_peaks):

        peak_param_i = df_frame_peaks.iloc[peak_i]
        Ypx, Zpx = peak_param_i["YCen(px)"], peak_param_i["ZCen(px)"]
        R, Eta = peak_param_i["Radius(px)"], peak_param_i["Eta(degrees)"]
        sigGR, sigGE = peak_param_i["SigmaGR"], peak_param_i["SigmaGEta"]
        sigLR, sigLE = peak_param_i["SigmaLR"], peak_param_i["SigmaLEta"]
        LGmix, IMax = peak_param_i["MU"], peak_param_i["IMax"]
        ISum, BG = peak_param_i["IntegratedIntensity"], peak_param_i["BG"]

        _, _, grid_RR, grid_EE = patch_methods.patch_grid_fromYZpos(Ypx, Zpx, cvsz, srfac, Ypx_BC, Zpx_BC)
        RE_coords = (grid_RR, grid_EE)

        peak_i_gen = draw_peaks_diffLGwidth(RE_coords,
                                            [R], [Eta],
                                            [sigGR], [sigGE], [sigLR], [sigLE],
                                            [LGmix], [IMax],
                                            BG=0.0, I_thresh=0)

        peak_i_gen[peak_i_gen > (0.05 * np.max(peak_i_gen))] += BG

        Y00, Z00 = int(srfac * (int(Ypx) - cvsz / 2)), int(srfac * (int(Zpx) - cvsz / 2))
        ci, cf = Y00, int(Y00 + srfac * cvsz)
        ri, rf = Z00, int(Z00 + srfac * cvsz)

        frame_gen[ri:rf, ci:cf] += peak_i_gen

    frame_gen[frame_gen < I_thresh] = 0

    return (frame_gen)


# ----------------------------
def peak_reconstruction_err(df_frame_peaks, frame_exp, frame_gen, cvsz=int(10)):
    """Calculate reconstruction error for each peak by comparing experimental and synthetic frames.
    Args:
        df_frame_peaks (pandas df): dataframe with peak parameters
        frame_exp (arr): experimental detector frame
        frame_gen (arr): synthetic frame
        cvsz (int; default=10): canvas size for drawing peaks
    Returns:
        peak_recon_err (list of floats): reconstruction error per peak
    """

    peak_recon_err = []
    nr_peaks = len(df_frame_peaks)

    for peak_i in range(0, nr_peaks):

        peak_param_i = df_frame_peaks.iloc[peak_i]
        Ypx, Zpx = peak_param_i["YCen(px)"], peak_param_i["ZCen(px)"]

        Y00, Z00 = int(int(Ypx) - cvsz / 2), int(int(Zpx) - cvsz / 2)
        ri, rf = Z00, int(Z00 + cvsz)
        ci, cf = Y00, int(Y00 + cvsz)

        patch_gen = frame_gen[ri:rf, ci:cf]
        patch_exp = frame_exp[ri:rf, ci:cf]

        err = np.sum(np.abs(patch_gen - patch_exp)) / np.sum(patch_exp)
        peak_recon_err.append(err)

    return (peak_recon_err)
