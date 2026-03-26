"""Functions to create or modify patches.

Source: SIMR-xrd/SIMR_xrd/utils/patch_methods.py
"""

import numpy as np

from sr_midas.physics import coord_transform


# ----------------------------
def patch_grid_fromYZpos(Ypx_patchCen, Zpx_patchCen, cvsz, srfac, Ypx_BC, Zpx_BC):
    """Creates patchgrid in Y-Z and R-Eta coordinate space from Y-Z patch center.
    Args:
        Ypx_patchCen (float): Patch center Y pixel coordinate
        Zpx_patchCen (float): Patch center Z pixel coordinate
        cvsz (int): canvas size in detector pixels (use even values)
        srfac (int): super resolution factor
        Ypx_BC (float): Beam Center Y pixel coordinate
        Zpx_BC (float): Beam Center Z pixel coordinate
    Returns:
        grid_YY (arr): Y coord grid over patch
        grid_ZZ (arr): Z coord grid over patch
        grid_RR (arr): R coord grid over patch (pixels)
        grid_EE (arr): Eta coord grid over patch (degrees)
    """

    dpx_sr = 1 / srfac

    Y00, Z00 = int(int(Ypx_patchCen) - cvsz / 2), int(int(Zpx_patchCen) - cvsz / 2)

    Ypx = np.arange(Y00, Y00 + cvsz, dpx_sr)
    Zpx = np.arange(Z00, Z00 + cvsz, dpx_sr)

    grid_YY, grid_ZZ = np.meshgrid(Ypx, Zpx)
    grid_RR, grid_EE = coord_transform.REta_from_YZ(grid_YY, grid_ZZ, Ypx_BC, Zpx_BC)

    return (grid_YY, grid_ZZ, grid_RR, grid_EE)


# ----------------------------
def patch_grid_fromREpos(R_patchCen, Eta_patchCen, cvsz, srfac, Ypx_BC, Zpx_BC):
    """Creates patchgrid in Y-Z and R-Eta coordinate space from R-Eta patch center.
    Args:
        R_patchCen (float): Patch center R coordinate (pixels)
        Eta_patchCen (float): Patch center Eta coordinate (degrees)
        cvsz (int): canvas size in detector pixels
        srfac (int): super resolution factor
        Ypx_BC (float): Beam Center Y pixel coordinate
        Zpx_BC (float): Beam Center Z pixel coordinate
    Returns:
        grid_YY (arr): Y coord grid over patch
        grid_ZZ (arr): Z coord grid over patch
        grid_RR (arr): R coord grid over patch (pixels)
        grid_EE (arr): Eta coord grid over patch (degrees)
    """

    Ypx_patchCen, Zpx_patchCen = coord_transform.YZ_from_REta(R_patchCen, Eta_patchCen, Ypx_BC, Zpx_BC)

    grid_YY, grid_ZZ, grid_RR, grid_EE = patch_grid_fromYZpos(Ypx_patchCen, Zpx_patchCen, cvsz, srfac, Ypx_BC, Zpx_BC)

    return (grid_YY, grid_ZZ, grid_RR, grid_EE)


# ----------------------------
def patch_grid_fromYZ00(Y00, Z00, cvsz, srfac, Ypx_BC, Zpx_BC):
    """Creates patchgrid in Y-Z and R-Eta coordinate space from top-left (0,0) pixel.
    Args:
        Y00 (float): Y-Pixel coordinate of (0,0) cell
        Z00 (float): Z-Pixel coordinate of (0,0) cell
        cvsz (int): canvas size in detector pixels
        srfac (int): super resolution factor
        Ypx_BC (float): Beam Center Y pixel coordinate
        Zpx_BC (float): Beam Center Z pixel coordinate
    Returns:
        grid_YY (arr): Y coord grid over patch
        grid_ZZ (arr): Z coord grid over patch
        grid_RR (arr): R coord grid over patch (pixels)
        grid_EE (arr): Eta coord grid over patch (degrees)
    """

    dpx_sr = 1 / srfac

    Ypx = np.arange(Y00, Y00 + cvsz, dpx_sr)
    Zpx = np.arange(Z00, Z00 + cvsz, dpx_sr)

    grid_YY, grid_ZZ = np.meshgrid(Ypx, Zpx)
    grid_RR, grid_EE = coord_transform.REta_from_YZ(grid_YY, grid_ZZ, Ypx_BC, Zpx_BC)

    return (grid_YY, grid_ZZ, grid_RR, grid_EE)


# ----------------------------
def upscale(input_patch, srfac_in, srfac_out):
    """Increase patch size by redistributing pixel intensities uniformly (loop version).
    Args:
        input_patch (arr): input patch
        srfac_in (int): super-res factor of input patch
        srfac_out (int): super-res factor for output patch
    Returns:
        out_patch (arr): upscaled output patch
    """

    if srfac_out >= srfac_in:

        upscale_fac = int(srfac_out / srfac_in)

        out_patch = np.zeros(shape=(input_patch.shape[0] * upscale_fac,
                                    input_patch.shape[1] * upscale_fac))

        for i in range(0, input_patch.shape[0]):
            for j in range(0, input_patch.shape[1]):
                rs, rf = i * upscale_fac, (i + 1) * upscale_fac
                cs, cf = j * upscale_fac, (j + 1) * upscale_fac
                out_patch[rs:rf, cs:cf] = np.sum(input_patch[i, j]) / (upscale_fac * upscale_fac)

        return (out_patch)

    elif srfac_out == srfac_in:
        return (input_patch)

    else:
        print("ERROR: 'srfac_out' must be larger than 'srfac_in' for upscaling.")


# ----------------------------
def downscale(input_patch, srfac_in, srfac_out):
    """Decrease patch size by pooling pixel intensities (loop version).
    Args:
        input_patch (arr): input patch
        srfac_in (int): super-res factor of input patch
        srfac_out (int): super-res factor for output patch
    Returns:
        out_patch (arr): downscaled output patch
    """

    if srfac_out <= srfac_in:

        downscale_fac = int(srfac_in / srfac_out)
        out_patch = np.zeros(shape=(int(input_patch.shape[0] / downscale_fac),
                                    int(input_patch.shape[1] / downscale_fac)))

        for i in range(0, out_patch.shape[0]):
            for j in range(0, out_patch.shape[1]):
                rs, rf = i * downscale_fac, (i + 1) * downscale_fac
                cs, cf = j * downscale_fac, (j + 1) * downscale_fac
                out_patch[i, j] = np.sum(input_patch[rs:rf, cs:cf])

        return (out_patch)

    elif srfac_out == srfac_in:
        return (input_patch)

    else:
        print("ERROR: 'srfac_out' must be smaller than 'srfac_in' for downscaling.")


# ----------------------------
def max_px_loc(in_patch):
    """Find location of max intensity pixel in a patch.
    Returns:
        (row, col) index of max value pixel as a tuple
    """

    return np.unravel_index(np.argmax(in_patch), in_patch.shape)
