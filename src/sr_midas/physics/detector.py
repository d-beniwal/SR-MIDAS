"""Detector geometry functions: rotation matrices, distortion map, ring number map.

Canonical source: SR-MIDAS/super_res_process.py
"""

import numpy as np
from sr_midas.physics.coord_transform import REta_from_YZ


# ----------------------------
def create_rotation_matrices(tx, ty, tz):
    """Create the rotation matrices for the given rotation angles
    Args:
        tx (float): rotation about x-axis in degrees
        ty (float): rotation about y-axis in degrees
        tz (float): rotation about z-axis in degrees
    Returns:
        Rx (np.ndarray): rotation matrix about x-axis
        Ry (np.ndarray): rotation matrix about y-axis
        Rz (np.ndarray): rotation matrix about z-axis
    """

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(np.deg2rad(tx)), -np.sin(np.deg2rad(tx))],
        [0, np.sin(np.deg2rad(tx)), np.cos(np.deg2rad(tx))]
        ])

    Ry = np.array([
        [np.cos(np.deg2rad(ty)), 0, np.sin(np.deg2rad(ty))],
        [0, 1, 0],
        [-np.sin(np.deg2rad(ty)), 0, np.cos(np.deg2rad(ty))]
        ])

    Rz = np.array([
        [np.cos(np.deg2rad(tz)), -np.sin(np.deg2rad(tz)), 0],
        [np.sin(np.deg2rad(tz)), np.cos(np.deg2rad(tz)), 0],
        [0, 0, 1]
        ])

    return Rx, Ry, Rz


# ----------------------------
def create_distortion_map(RR, EE, sr_params):
    """Full distortion model matching PeaksFittingOMPZarrRefactor.c:2721-2739.

    Uses all 15 distortion parameters (p0-p14) with RhoD normalization.
    When p4-p14 are 0 (default), this reduces to the basic 4-parameter model.

    Args:
        RR (np.ndarray): radial map (in microns)
        EE (np.ndarray): polar angle map (degrees, spans 180 to -180 clockwise)
        sr_params (dict): parameter dictionary containing p0-p14 and RhoD
    Returns:
        dist_fRE (np.ndarray): distortion factor map
    """
    RhoD = sr_params["RhoD"]
    RR_N = RR / RhoD
    EE_T = np.deg2rad(90 - EE)

    RN2 = RR_N ** 2
    RN3 = RR_N ** 3
    RN4 = RN2 * RN2
    RN5 = RN4 * RR_N
    RN6 = RN4 * RN2

    p = sr_params
    dist_fRE = (
        p["p0"]  * RN2 * np.cos(2.0 * EE_T + np.deg2rad(p["p6"]))
      + p["p1"]  * RN4 * np.cos(4.0 * EE_T + np.deg2rad(p["p3"]))
      + p["p2"]  * RN2
      + p["p4"]  * RN6
      + p["p5"]  * RN4
      + p["p7"]  * RN4 * np.cos(EE_T + np.deg2rad(p["p8"]))
      + p["p9"]  * RN3 * np.cos(3.0 * EE_T + np.deg2rad(p["p10"]))
      + p["p11"] * RN5 * np.cos(5.0 * EE_T + np.deg2rad(p["p12"]))
      + p["p13"] * RN6 * np.cos(6.0 * EE_T + np.deg2rad(p["p14"]))
      + 1.0
    )

    return dist_fRE


# ----------------------------
def ringNr_map_on_detector(sr_params, residual_corr_map=None):
    """Create the ring number map on the detector. Pixels that don't belong to any ring are marked with -1.
    Pixels that belong to a ring in sr_params["ringsToUse"] are marked with the ring index (starting from 0).
    Args:
        sr_params (dict): dictionary containing the parameters for the super-resolution process
        residual_corr_map (np.ndarray or None): optional per-pixel residual correction map
    Returns:
        RingNrmap (arr): ring number map on the detector
    """

    det_shape = (sr_params["numPxZ"], sr_params["numPxY"])

    YYpx, ZZpx = np.meshgrid(np.arange(0, sr_params["numPxY"]), np.arange(0, sr_params["numPxZ"]))

    yyd = (-YYpx + sr_params["Ypx_BC"])*sr_params["pxSize"]
    zzd = (ZZpx - sr_params["Zpx_BC"])*sr_params["pxSize"]

    xyz_comb = np.array([
        np.zeros_like(yyd.flatten()),
        yyd.flatten(),
        zzd.flatten()
    ])

    Rx, Ry, Rz = create_rotation_matrices(sr_params["tx"], sr_params["ty"], sr_params["tz"])

    xyz_comb_rot = (Rx @ (Ry @ Rz)) @ xyz_comb

    XX = (sr_params["Lsd"] + xyz_comb_rot[0]).reshape(det_shape)
    YY = xyz_comb_rot[1].reshape(det_shape)
    ZZ = xyz_comb_rot[2].reshape(det_shape)

    RR = (sr_params["Lsd"]/XX) * np.sqrt(YY**2 + ZZ**2)
    EE = np.zeros_like(YY)
    EE[YY <= 0] = np.rad2deg(np.arccos(ZZ[YY<=0] / (np.sqrt(YY[YY<=0]**2 + ZZ[YY<=0]**2))))
    EE[YY > 0] = np.rad2deg(- np.arccos(ZZ[YY>0] / (np.sqrt(YY[YY>0]**2 + ZZ[YY>0]**2))))

    dist_fRE = create_distortion_map(RR, EE, sr_params)

    Rmap = RR * dist_fRE / sr_params["pxSize"]

    if residual_corr_map is not None:
        Rmap += residual_corr_map

    RingNrmap = np.full(det_shape, -1, dtype=int)

    for i, R in enumerate(sr_params["rings_to_use_Rpx"]):
        ring_mask = np.abs(Rmap - R) <= (sr_params["ring_width"]/sr_params["pxSize"])
        RingNrmap[ring_mask] = i

    return RingNrmap
