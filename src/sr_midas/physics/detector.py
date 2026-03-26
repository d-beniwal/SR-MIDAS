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
def create_distortion_map(RR, EE, p0, p1, p2, p3, px):
    """Create the distortion map for the given parameters
    Args:
        RR (np.ndarray): radial map (in microns)
        EE (np.ndarray): polar angle map (spans from 180 to -180 degrees in clockwise direction)
        p0 (float): distortion parameter
        p1 (float): distortion parameter
        p2 (float): distortion parameter
        p3 (float): distortion parameter
        px (float): pixel size in microns
    Returns:
        dist_fRE (np.ndarray): distortion factor map
    """

    RR_N = RR / np.max(RR) # Normalized radial map
    EE_T = 90 - EE # Transformed polar angle map

    # Distortion terms
    dist_gRE = p0 * (RR_N**2) * np.cos(np.deg2rad(2*EE_T)) # 2-fold angular
    dist_hRE = p1 * (RR_N**4) * np.cos(np.deg2rad(4*EE_T + p3)) # 4-fold angular
    dist_kRE = p2 * (RR_N**2) # radial distortion
    dist_fRE = 1 + dist_gRE + dist_hRE + dist_kRE # combined distortion factor

    return dist_fRE


# ----------------------------
def ringNr_map_on_detector(sr_params):
    """Create the ring number map on the detector. Pixels that don't belong to any ring are marked with -1.
    Pixels that belong to a ring in sr_params["ringsToUse"] are marked with the ring index (starting from 0).
    Args:
        sr_params (dict): dictionary containing the parameters for the super-resolution process
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

    dist_fRE = create_distortion_map(RR, EE,
                                    sr_params["p0"], sr_params["p1"], sr_params["p2"], sr_params["p3"],
                                    sr_params["pxSize"])

    Rmap = RR * dist_fRE / sr_params["pxSize"]

    RingNrmap = np.full(det_shape, -1, dtype=int)

    for i, R in enumerate(sr_params["rings_to_use_Rpx"]):
        ring_mask = np.abs(Rmap - R) <= (sr_params["ring_width"]/sr_params["pxSize"])
        RingNrmap[ring_mask] = i

    return (RingNrmap)
