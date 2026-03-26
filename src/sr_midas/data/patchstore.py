"""Patchstore I/O functions.

Canonical source for load_patchstore_h5data (5-tuple): SRML-HEDM/CNNSR/dataset.py
midas_Zarr_zip and df_to_sarray from SIMR-xrd/SIMR_xrd/utils/data_reader.py
"""

import zarr
import numpy as np
import h5py
import pandas as pd


# ----------------------------
def midas_Zarr_zip(filepath):
    """Loads detector data from midas zip file
    Args:
        filepath (str): path to the data file
    Returns:
        zf (Zarr): full z-array object with all detector data
        exData (arr): exchange data array with pixel intensities for all frames
        exDark (arr): exchange data (dark) array
        exBright (arr): exchange data (bright) array
    """

    zf = zarr.open(filepath, "r")
    zfEx = zf["exchange"]
    exData = np.array(zfEx["data"])
    exDark = np.array(zfEx["dark"])
    exBright = np.array(zfEx["bright"])

    return (zf, exData, exDark, exBright)


# ----------------------------
def df_to_sarray(df):
    """Converts a pandas DataFrame to a numpy structured array for storage in h5 files.
    String columns are converted to bytes of length = max(len(col)).
    Args:
        df (pandas dataframe): the data frame to convert
    Returns:
        z (arr): a numpy structured array representation of df
        dtype (str): datatype
    """

    def make_col_type(col_type, col):
        try:
            if 'numpy.object_' in str(col_type.type):
                maxlens = col.dropna().str.len()
                if maxlens.any():
                    maxlen = maxlens.max().astype(int)
                    col_type = ('S%s' % maxlen, 1)
                else:
                    col_type = 'f2'
            return col.name, col_type
        except:
            print(col.name, col_type, col_type.type, type(col))
            raise

    v = df.values
    types = df.dtypes
    numpy_struct_types = [make_col_type(types[col], df.loc[:, col]) for col in df.columns]
    dtype = np.dtype(numpy_struct_types)
    z = np.zeros(v.shape[0], dtype)
    for (i, k) in enumerate(z.dtype.names):
        try:
            if dtype[i].str.startswith('|S'):
                z[k] = df[k].str.encode('latin').astype('S')
            else:
                z[k] = v[:, i]
        except:
            print(k, v[:, i])
            raise

    return z, dtype


# ----------------------------
def load_patchstore_h5data(pst_path, only_patch_arrays=False):
    """Loads data from a patchstore file in .h5 format.
    Canonical source: SRML-HEDM/CNNSR/dataset.py (5-tuple return).
    Args:
        pst_path (str): path to the .h5 patchstore file
        only_patch_arrays (bool; default=False): If True, only patch arrays are loaded;
            other information (peak shapes, locations, coordinates) is skipped.
    Returns:
        patch_arr (dict of arrays): patches of different resolutions
        df_patch_info (pandas dataframe): information on each patch
        peaks_loc_in_patch (dict of dict of arrays): peak locations relative to patch
        peak_params (dict of arrays): peak parameters
        pst_creation_args (dict): patchstore creation arguments (empty dict if absent)
    """

    pst = h5py.File(pst_path)

    # Extract patch arrays at all resolutions
    patch_arr = {i: [] for i in pst["patchArr"].keys()}
    for i in patch_arr.keys():
        patch_arr[i] = np.array(pst["patchArr"][i])

    if only_patch_arrays in ["False", "false", False, "No", "no", "N", "n"]:

        df_patch_info = pd.DataFrame(np.array(pst["patchInfo"]))

        peaks_loc_in_patch = {i: {"Ypx": [], "Zpx": []} for i in pst["peaksLocInPatch"].keys()}
        for i in peaks_loc_in_patch.keys():
            peaks_loc_in_patch[i]["Ypx"] = np.array(pst["peaksLocInPatch"][i]["Ypx"])
            peaks_loc_in_patch[i]["Zpx"] = np.array(pst["peaksLocInPatch"][i]["Zpx"])

        peak_params = {i: [] for i in pst["peaksParameters"].keys()}
        for i in peak_params.keys():
            peak_params[i] = np.array(pst["peaksParameters"][i])

        # pstCreationArgs absent in older SIMR-xrd patchstores — default to empty dict
        if "pstCreationArgs" in pst:
            pst_creation_args = {i: [] for i in pst["pstCreationArgs"].keys()}
            for i in pst_creation_args.keys():
                pst_creation_args[i] = np.array(pst["pstCreationArgs"][i])
        else:
            pst_creation_args = {}

        del pst

        return (patch_arr, df_patch_info, peaks_loc_in_patch, peak_params, pst_creation_args)

    else:
        return (patch_arr)
