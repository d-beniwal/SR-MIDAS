"""Patch extraction and peak-fitting operations for the SR-MIDAS pipeline.

Extracted from: SR-MIDAS/super_res_process.py
"""

import numpy as np
from scipy.ndimage import label, find_objects, gaussian_filter, median_filter
from scipy.optimize import curve_fit
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.feature import peak_local_max


# ----------------------------
def patches_from_detector_frame(frame_arr, sr_config, connectivity_dim=8):
    """Extracts patches from input detector frame using connected component algorithm.

    First a labelled array is created where different spots are marked with different
    labels. Each spot is then isolated and pasted onto a patch of size=patch_size such
    that the max intensity pixel is at the center of the patch.

    Args:
        frame_arr (arr): experimental detector frame with good pixels
            (pixels that belong to a ring and exceed the threshold for that ring)
        sr_config (dict): dictionary containing the configuration for the
            super-resolution process
        connectivity_dim (int; default=8): if set to 8: accepts continuity with
            diagonal cells; otherwise uses 4-connectivity only
    Returns:
        patches (arr): extracted patch array (N, patch_size, patch_size)
        patches_Z00 (list of int): Z coordinates of (0,0) pixels in patches
        patches_Y00 (list of int): Y coordinates of (0,0) pixels in patches
        nr_pixels_in_patch (list of int): no. of non-zero pixels in the patch
    """

    patch_size = sr_config["spot_find_args"]["patch_size"]

    if connectivity_dim in [8, 8.0, "8"]:
        structure = np.array([[1, 1, 1],
                              [1, 1, 1],
                              [1, 1, 1]])
        labeled_array, num_features = label(frame_arr, structure)
    else:
        labeled_array, num_features = label(frame_arr)

    bounding_boxes = find_objects(labeled_array)

    spots = []
    patches = []
    patches_Y00, patches_Z00 = [], []
    nr_pixels_in_patch = []

    for i, bbox in enumerate(bounding_boxes):
        if bbox is not None:
            spot = frame_arr[bbox]
            spot = np.where(labeled_array[bbox] == (i + 1), spot, 0)
            nr_pixels_in_spot = np.count_nonzero(spot)

            if nr_pixels_in_spot >= sr_config["minPxCount"]:

                r_max_spot, c_max_spot = np.unravel_index(np.argmax(spot), spot.shape)
                Zpx_max_spot = bbox[0].start + r_max_spot
                Ypx_max_spot = bbox[1].start + c_max_spot

                Z00_patch = Zpx_max_spot - int(patch_size / 2)
                Y00_patch = Ypx_max_spot - int(patch_size / 2)

                if (Z00_patch >= 0) and \
                   (Z00_patch < frame_arr.shape[0] - patch_size) and \
                   (Y00_patch >= 0) and \
                   (Y00_patch < frame_arr.shape[1] - patch_size):

                    patch = np.pad(spot, pad_width=patch_size, mode='constant', constant_values=0)

                    r_max_patch, c_max_patch = np.unravel_index(np.argmax(patch), patch.shape)

                    rs = int(r_max_patch - int(patch_size / 2))
                    cs = int(c_max_patch - int(patch_size / 2))
                    rf, cf = rs + patch_size, cs + patch_size
                    patch = patch[rs:rf, cs:cf]
                    patches.append(patch)

                    patches_Z00.append(Z00_patch)
                    patches_Y00.append(Y00_patch)

                    spots.append(spot)
                    nr_pixels_in_patch.append(np.count_nonzero(spot))

    patches = np.array(patches)

    return (patches, patches_Z00, patches_Y00, nr_pixels_in_patch)


# ----------------------------
def weighted_center_of_mass(patch, threshold=0.5):
    """Computes the weighted center of mass (centroid) of a 2D patch.

    Args:
        patch (ndarray): 2D NumPy array representing the patch.
        threshold (float): fraction of max intensity; pixels below are zeroed.
    Returns:
        tuple: (Z_center, Y_center) coordinates of the centroid.
            Z represents row, Y represents col
    """

    patch = np.array(patch)
    patch[patch < threshold * np.max(patch)] = 0

    height, width = patch.shape
    Z_indices, Y_indices = np.indices((height, width))

    total_mass = np.sum(patch)

    if total_mass == 0:
        Y_center, Z_center = width / 2, height / 2
    else:
        Y_center = np.sum(Y_indices * patch) / total_mass
        Z_center = np.sum(Z_indices * patch) / total_mass

    return (Z_center, Y_center)


# ----------------------------
def com_peak_coords(patch, locmax_peak_coords, threshold=0.5, peak_crop_size=20):
    """Calculate center of mass peak coordinates in a patch.

    Args:
        patch (arr): input patch (with single or multiple peaks)
        locmax_peak_coords (arr): peak coords obtained from local maxima
        threshold (float; default=0.5): fraction value to threshold cropped peak patches
        peak_crop_size (int): size of peak patches cropped around each local max peak
            position; center of mass calculation is done in the cropped patch
    Returns:
        com_peak_coord (arr): array with com peak coordinates; column0 has Z (row)
            coordinates, column1 has Y (column) coordinates
    """

    n_locmax_peaks = len(locmax_peak_coords)
    com_peakY, com_peakZ = [], []

    for i in range(n_locmax_peaks):
        peak_patch_Y00 = locmax_peak_coords[i][1] - int(peak_crop_size / 2)
        peak_patch_Z00 = locmax_peak_coords[i][0] - int(peak_crop_size / 2)

        row_T = max(0, peak_patch_Z00)
        row_B = min(peak_patch_Z00 + peak_crop_size, patch.shape[0])
        col_L = max(0, peak_patch_Y00)
        col_R = min(peak_patch_Y00 + peak_crop_size, patch.shape[1])

        peak_patch = patch[row_T:row_B, col_L:col_R]
        Z_center, Y_center = weighted_center_of_mass(peak_patch, threshold)

        com_peakY.append(peak_patch_Y00 + Y_center)
        com_peakZ.append(peak_patch_Z00 + Z_center)

    result = np.zeros(shape=(n_locmax_peaks, 2))
    result[:, 0] = np.array(com_peakZ)
    result[:, 1] = np.array(com_peakY)

    return result


# ----------------------------
def watershed_peaks(patch, peak_coords, mask_thresh):
    """Apply watershed algorithm to allocate regions.

    Args:
        patch (arr): input patch
        peak_coords (arr): coordinates of peaks in patch
        mask_thresh (float): fraction of max intensity used as threshold for mask
    Returns:
        labelled_patch (arr): labelled patch
        peak_label_values (arr): label values of peaks
        peaks_Isum (arr): sum of intensities of peaks
        peaks_NrPixels (arr): number of pixels per peak region
    """

    n_peaks_in_patch = len(peak_coords)
    markers = np.zeros_like(patch, dtype=np.int32)
    for i in range(n_peaks_in_patch):
        markers[peak_coords[i][0], peak_coords[i][1]] = i + 1

    mask_thresh = mask_thresh * np.max(patch)
    labelled_patch = watershed(-patch, markers, mask=patch > mask_thresh)

    label_props = regionprops(labelled_patch, intensity_image=patch)
    peak_label_values = [prop.label for prop in label_props]
    peaks_Isum = [prop.mean_intensity * prop.area for prop in label_props]
    peaks_NrPixels = [prop.num_pixels for prop in label_props]

    return (labelled_patch, peak_label_values, peaks_Isum, peaks_NrPixels)


# ----------------------------
def multi_pv_fit(patch, Y00, Z00, Ypx_BC, Zpx_BC, lrsz, srfac,
                 min_distance, threshold_rel,
                 gauss_filter_sigma=0, median_filter_size=1, lr_int_thresh=10.0):
    """Performs multiple pseudo-Voigt fit on input patch.

    No. of peaks and their approximate location is identified using local maxima.

    Args:
        patch (arr): input patch (with single or multiple peaks)
        Y00 (int): Y coord of (0,0) px of patch in detector frame in native x1 resolution
        Z00 (int): Z coord of (0,0) px of patch in detector frame in native x1 resolution
        Ypx_BC (float): Beam center Y coord in detector frame
        Zpx_BC (float): Beam center Z coord in detector frame
        lrsz (int): low-res native patch size
        srfac (int): super-res factor of patch being fitted
        min_distance (float): minimum distance (in pixels) between two peaks
        threshold_rel (float): threshold value to count a pixel as a peak
        gauss_filter_sigma (float; default=0): standard deviation for Gaussian kernel
        median_filter_size (float; default=1): size for median filter
        lr_int_thresh (float; default=10.0): threshold used to filter out pixels
    Returns:
        fit_peaks (list of dict): properties of fitted peaks; peak positions in
            detector frame coordinates
        fit_peak_coords (arr): pv-fit peak coordinates; column0 has Z (row)
            coordinates, column1 has Y (column) coordinates in patch coordinates
        fit_patch (arr): fitted patch
    """

    SRx_thresholdInt = lr_int_thresh / (srfac * srfac)

    def multiPV2d_diffLGwidth(RE_coords, *params):
        RR, EE = RE_coords
        n_paramsPerPeak = 8
        n_peaks = len(params) // n_paramsPerPeak
        patchI = np.zeros_like(RR, dtype=np.float32)

        for i in range(n_peaks):
            Rpx, EtaDeg, sigGR, sigGEta, sigLR, sigLEta, LGmix, IMax = \
                params[i * n_paramsPerPeak: (i + 1) * n_paramsPerPeak]
            G = IMax * np.exp(-0.5 * ((RR - Rpx) / sigGR)**2 - 0.5 * ((EE - EtaDeg) / sigGEta)**2)
            L = IMax * 1 / ((1 + ((RR - Rpx) / sigLR)**2) * (1 + ((EE - EtaDeg) / sigLEta)**2))
            patchI += (LGmix * L) + ((1 - LGmix) * G)

        patchI[patchI < SRx_thresholdInt] = 0
        return patchI

    def multiPV2d_diffLGwidth_forFit(RE_coords, *params):
        return multiPV2d_diffLGwidth(RE_coords, *params).ravel()

    dR, dEta, dIMax = 2, 0.1, 400 / srfac
    dpx_sr = 1 / srfac

    Ypx = np.arange(Y00, Y00 + lrsz, dpx_sr)
    Zpx = np.arange(Z00, Z00 + lrsz, dpx_sr)

    grid_YY, grid_ZZ = np.meshgrid(Ypx, Zpx)
    grid_RR = ((Ypx_BC - grid_YY)**2 + (Zpx_BC - grid_ZZ)**2)**0.5
    grid_EE = np.rad2deg(np.arccos((grid_ZZ - Zpx_BC) / grid_RR))
    grid_EE = grid_EE * ((grid_YY - Ypx_BC) / np.abs(grid_YY - Ypx_BC))

    patch_for_locmax = gaussian_filter(patch, sigma=gauss_filter_sigma)
    patch_for_locmax = median_filter(patch_for_locmax, size=median_filter_size)

    locmax_coords = peak_local_max(patch_for_locmax, min_distance=min_distance,
                                   threshold_rel=threshold_rel)

    nPeaks_locmax = len(locmax_coords)
    locmax_Y, locmax_Z = locmax_coords[:, 1], locmax_coords[:, 0]

    fit_params_names = ["R(px)", "Eta(deg)", "SigGR", "SigGEta", "SigLR", "SigLEta", "LGmix", "IMax"]

    init_guess = [0, 0, 0.3, 0.3, 0.3, 0.3, 0.5, 0] * nPeaks_locmax
    lower_bounds = [0, 0, 0.1, 0.05, 0.1, 0.05, 0, 0] * nPeaks_locmax
    upper_bounds = [0, 0, 3.0, 3.0, 3.0, 3.0, 1, 0] * nPeaks_locmax

    n_paramsPerPeak = len(fit_params_names)

    for i in range(nPeaks_locmax):
        Rpx_guess = grid_RR[locmax_Z[i], locmax_Y[i]]
        Eta_guess = grid_EE[locmax_Z[i], locmax_Y[i]]
        IMax_guess = patch[locmax_Z[i], locmax_Y[i]]

        init_guess[i * n_paramsPerPeak] = Rpx_guess
        lower_bounds[i * n_paramsPerPeak] = Rpx_guess - dR
        upper_bounds[i * n_paramsPerPeak] = Rpx_guess + dR

        init_guess[i * n_paramsPerPeak + 1] = Eta_guess
        lower_bounds[i * n_paramsPerPeak + 1] = Eta_guess - dEta
        upper_bounds[i * n_paramsPerPeak + 1] = Eta_guess + dEta

        init_guess[i * n_paramsPerPeak + 7] = IMax_guess
        upper_bounds[i * n_paramsPerPeak + 7] = IMax_guess + dIMax

    popt, _ = curve_fit(
        multiPV2d_diffLGwidth_forFit, (grid_RR, grid_EE), patch.ravel(),
        p0=init_guess, bounds=(lower_bounds, upper_bounds),
        maxfev=1000, method="trf"
    )

    fit_patch = multiPV2d_diffLGwidth((grid_RR, grid_EE), *popt).reshape(grid_RR.shape)

    fit_peaks = []
    for i in range(nPeaks_locmax):
        params = popt[i * n_paramsPerPeak:(i + 1) * n_paramsPerPeak]
        fit_peaks.append({fit_params_names[j]: params[j] for j in range(n_paramsPerPeak)})

        Y0_fit = Ypx_BC + (fit_peaks[i]['R(px)'] * np.sin(np.deg2rad(fit_peaks[i]['Eta(deg)'])))
        Z0_fit = Zpx_BC + (fit_peaks[i]['R(px)'] * np.cos(np.deg2rad(fit_peaks[i]['Eta(deg)'])))

        fit_peaks[i]['Y(px)'] = Y0_fit
        fit_peaks[i]['Z(px)'] = Z0_fit

    fit_peak_coords = np.zeros(shape=(len(fit_peaks), 2))
    fit_peak_coords[:, 0] = np.array([fit_peaks[i]['Z(px)'] for i in range(len(fit_peak_coords))])
    fit_peak_coords[:, 1] = np.array([fit_peaks[i]['Y(px)'] for i in range(len(fit_peak_coords))])

    fit_peak_coords[:, 0] = (fit_peak_coords[:, 0] - Z00) * srfac
    fit_peak_coords[:, 1] = (fit_peak_coords[:, 1] - Y00) * srfac

    return fit_peaks, fit_peak_coords, fit_patch
