"""2D peak shape functions: Gaussian, Lorentzian, pseudo-Voigt.

Canonical source for pseudoVoigt2d_diffLGwidth, gaussian2d, lorentzian2d:
    SR-MIDAS/super_res_process.py

pseudoVoigt2d_sameLGwidth from SIMR-xrd/SIMR_xrd/utils/peaks2D.py (not present in super_res_process.py)

Terminology: Y->horizontal, Z->vertical in detector frame
"""

import numpy as np


# ----------------------------
def gaussian2d(yy, zz, y0, z0, ySig, zSig, IMax, *args):
    """Create 2D Gaussian spread over yy and zz coordinates
    Args:
        yy (arr): Y coord (pixel) grid array that maps Y distribution over patch
        zz (arr): Z coord (pixel) grid array that maps Z distribution over patch
        y0 (float): Peak position Y coord (pixel)
        z0 (float): Peak position Z coord (pixel)
        ySig (float): Peak width in Y-axis
        zSig (float): Peak width in Z-axis
        IMax (float): Max intensity
        *args (list): [rot_deg (rotation angle in degrees)]
    Returns:
        G (arr): Gaussian peak over Y-Z grid
    """

    if len(args) > 0: rot_deg = args[0]
    else: rot_deg = 0

    rot_rad = np.deg2rad(rot_deg)

    yyRot = (yy - y0) * np.cos(rot_rad) - (zz - z0) * np.sin(rot_rad)
    zzRot = (yy - y0) * np.sin(rot_rad) + (zz - z0) * np.cos(rot_rad)

    G = IMax * np.exp(-0.5 * (yyRot / ySig)**2 - 0.5 * (zzRot / zSig)**2)

    return (G)


# ----------------------------
def lorentzian2d(yy, zz, y0, z0, ySig, zSig, IMax, *args):
    """Create 2D Lorentzian spread over yy and zz coordinates
    Args:
        yy (arr): Y coord (pixel) grid array that maps Y distribution over patch
        zz (arr): Z coord (pixel) grid array that maps Z distribution over patch
        y0 (float): Peak position Y coord (pixel)
        z0 (float): Peak position Z coord (pixel)
        ySig (float): Peak width in Y-axis
        zSig (float): Peak width in Z-axis
        IMax (float): Max intensity
        *args (list): [rot_deg (rotation angle in degrees)]
    Returns:
        L (arr): Lorentzian peak over Y-Z grid
    """

    if len(args) > 0: rot_deg = args[0]
    else: rot_deg = 0

    rot_rad = np.deg2rad(rot_deg)

    yyRot = (yy - y0) * np.cos(rot_rad) - (zz - z0) * np.sin(rot_rad)
    zzRot = (yy - y0) * np.sin(rot_rad) + (zz - z0) * np.cos(rot_rad)

    L = IMax * 1 / ((1 + (yyRot / ySig)**2) * (1 + (zzRot / zSig)**2))

    return (L)


# ----------------------------
def pseudoVoigt2d_diffLGwidth(yy, zz, y0, z0, ySigG, zSigG, ySigL, zSigL, LGmix, IMax, *args):
    """Create 2D pseudo-Voigt spread over yy and zz coordinates using different Gaussian and Lorentzian widths.
    Canonical source: SR-MIDAS/super_res_process.py
    Args:
        yy (arr): Y coord (pixel) grid array that maps Y distribution over patch
        zz (arr): Z coord (pixel) grid array that maps Z distribution over patch
        y0 (float): Peak position Y coord (pixel)
        z0 (float): Peak position Z coord (pixel)
        ySigG (float): Gaussian peak width in Y-axis
        zSigG (float): Gaussian peak width in Z-axis
        ySigL (float): Lorentzian peak width in Y-axis
        zSigL (float): Lorentzian peak width in Z-axis
        LGmix (float): LGmix factor
        IMax (float): Max intensity
        *args (list): [rot_deg (rotation angle in degrees)]
    Returns:
        PV (arr): pseudo-Voigt peak over Y-Z grid
    """

    if len(args) > 0: rot_deg = args[0]
    else: rot_deg = 0

    G = gaussian2d(yy, zz, y0, z0, ySigG, zSigG, IMax, *[rot_deg])
    L = lorentzian2d(yy, zz, y0, z0, ySigL, zSigL, IMax, *[rot_deg])

    PV = (LGmix * L) + ((1 - LGmix) * G)

    return (PV)


# ----------------------------
def pseudoVoigt2d_sameLGwidth(yy, zz, y0, z0, ySig, zSig, LGmix, IMax, *args):
    """Create 2D pseudo-Voigt spread over yy and zz coordinates using same Gaussian and Lorentzian widths.
    Source: SIMR-xrd/SIMR_xrd/utils/peaks2D.py
    Args:
        yy (arr): Y coord (pixel) grid array that maps Y distribution over patch
        zz (arr): Z coord (pixel) grid array that maps Z distribution over patch
        y0 (float): Peak position Y coord (pixel)
        z0 (float): Peak position Z coord (pixel)
        ySig (float): Peak width in Y-axis
        zSig (float): Peak width in Z-axis
        LGmix (float): LGmix factor
        IMax (float): Max intensity
        *args (list): [rot_deg (rotation angle in degrees)]
    Returns:
        PV (arr): pseudo-Voigt peak over Y-Z grid
    """

    if len(args) > 0: rot_deg = args[0]
    else: rot_deg = 0

    G = gaussian2d(yy, zz, y0, z0, ySig, zSig, IMax, *[rot_deg])
    L = lorentzian2d(yy, zz, y0, z0, ySig, zSig, IMax, *[rot_deg])

    PV = (LGmix * L) + ((1 - LGmix) * G)

    return (PV)
