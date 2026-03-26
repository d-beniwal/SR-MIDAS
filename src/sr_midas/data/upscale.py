"""Upscaling functions for patch resolution increase.

Canonical source for upscale_fast: SR-MIDAS/super_res_process.py (vectorized np.repeat variant)
upscale (loop version) retained for compatibility with synthesis and training pipelines.
"""

import numpy as np


# ----------------------------
def upscale(input_patch, srfac_in, srfac_out):
    """Increase size of patch by redistributing the pixel intensities uniformly (loop version).
    Args:
        input_patch (arr): input patch
        srfac_in (int): super-res factor of input patch
        srfac_out (int): super-res factor for output patch
    Returns:
        out_patch (arr): upscaled output patch
    """

    if srfac_out > srfac_in:

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
def upscale_fast(patches_subset, upscale_fac):
    """Increase size of a batch of patches using vectorized np.repeat (fast version).
    Canonical source: SR-MIDAS/super_res_process.py
    Args:
        patches_subset (arr): batch of patches with shape (N, C, H, W)
        upscale_fac (int): upscale factor (e.g. 2 to double resolution)
    Returns:
        upscaled_patches (arr): upscaled patch batch, intensity scaled by 1/(upscale_fac^2)
    """
    upscaled_patches = np.repeat(np.repeat(patches_subset, upscale_fac, axis=2), upscale_fac, axis=3)
    upscaled_patches = upscaled_patches / (upscale_fac * upscale_fac)
    return upscaled_patches
