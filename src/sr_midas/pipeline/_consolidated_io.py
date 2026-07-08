"""
Binary I/O for MIDAS consolidated peak files.

Replicates the binary format defined in PeaksFittingConsolidatedIO.h
so that downstream MIDAS stages (MergeOverlappingPeaksAllZarr, etc.)
can read SR-MIDAS output directly.
"""

import struct
import numpy as np

N_PEAK_COLS = 29

# Column order of AllPeaks_PS.bin / per-frame *_PS.csv, matching the header
# written by MIDAS (PeaksFittingOMPZarrRefactor.c) and SR-MIDAS
# (see sr_process.py).  Kept here so the reader can build DataFrames with the
# exact same column names the CSV path produces.
PS_COLUMNS = [
    "SpotID", "IntegratedIntensity", "Omega(degrees)", "YCen(px)", "ZCen(px)",
    "IMax", "Radius(px)", "Eta(degrees)", "SigmaR", "SigmaEta", "NrPixels",
    "TotalNrPixelsInPeakRegion", "nPeaks", "maxY", "maxZ", "diffY", "diffZ",
    "rawIMax", "returnCode", "retVal", "BG", "SigmaGR", "SigmaLR", "SigmaGEta",
    "SigmaLEta", "MU", "RawSumIntensity", "maskTouched", "FitRMSE",
]
assert len(PS_COLUMNS) == N_PEAK_COLS


def write_allpeaks_ps_bin(filepath, nr_frames, frame_peak_data):
    """Write AllPeaks_PS.bin in MIDAS consolidated format.

    Args:
        filepath: Output file path.
        nr_frames: Total number of frames in the dataset.
        frame_peak_data: List of length nr_frames.  Each element is either
            a np.ndarray of shape (n_peaks, 29) with dtype float64, or
            None for frames with 0 peaks.

    Binary layout (matches PeaksFittingConsolidatedIO.h):
        Header:
            int32   nFrames
            int32   nPeaks[nFrames]
            int64   offsets[nFrames]
        Data:
            For each frame with peaks: nPeaks * 29 float64 values (row-major).
    """
    n_peaks_arr = np.zeros(nr_frames, dtype=np.int32)
    for f in range(nr_frames):
        if frame_peak_data[f] is not None:
            n_peaks_arr[f] = frame_peak_data[f].shape[0]

    # header_size = sizeof(int32) + nFrames*sizeof(int32) + nFrames*sizeof(int64)
    header_size = 4 + nr_frames * 4 + nr_frames * 8

    offsets = np.zeros(nr_frames, dtype=np.int64)
    data_off = header_size
    for f in range(nr_frames):
        offsets[f] = data_off
        data_off += int(n_peaks_arr[f]) * N_PEAK_COLS * 8  # 8 bytes per float64

    with open(filepath, 'wb') as fh:
        fh.write(struct.pack('<i', nr_frames))
        fh.write(n_peaks_arr.tobytes())
        fh.write(offsets.tobytes())
        for f in range(nr_frames):
            if frame_peak_data[f] is not None and n_peaks_arr[f] > 0:
                data = np.ascontiguousarray(frame_peak_data[f], dtype=np.float64)
                fh.write(data.tobytes())


def write_allpeaks_px_bin(filepath, nr_frames, nr_pixels, frame_pixel_data):
    """Write AllPeaks_PX.bin in MIDAS consolidated format.

    Args:
        filepath: Output file path.
        nr_frames: Total number of frames.
        nr_pixels: Detector dimension (max of height, width).
        frame_pixel_data: List of length nr_frames.  Each element is either
            a list of (pixel_y, pixel_z) tuples (one per peak), or None.
            pixel_y and pixel_z are 1-D int16 arrays of detector-row and
            detector-column indices for non-zero pixels in the peak region.

    Binary layout (matches PeaksFittingConsolidatedIO.h):
        Header:
            int32   nFrames
            int32   NrPixels
            int32   nPeaks[nFrames]
            int64   offsets[nFrames]
        Data:
            For each frame, for each peak:
                int32   nPixels
                int16   y,z pairs × nPixels  (interleaved: y0,z0,y1,z1,...)
    """
    n_peaks_arr = np.zeros(nr_frames, dtype=np.int32)
    for f in range(nr_frames):
        if frame_pixel_data[f] is not None:
            n_peaks_arr[f] = len(frame_pixel_data[f])

    # header_size = 2*sizeof(int32) + nFrames*sizeof(int32) + nFrames*sizeof(int64)
    header_size = 4 + 4 + nr_frames * 4 + nr_frames * 8

    offsets = np.zeros(nr_frames, dtype=np.int64)
    data_off = header_size
    for f in range(nr_frames):
        offsets[f] = data_off
        if frame_pixel_data[f] is not None:
            for (py, pz) in frame_pixel_data[f]:
                n_px = len(py)
                # int32 nPixels + nPx * 2 * sizeof(int16)
                data_off += 4 + n_px * 2 * 2

    with open(filepath, 'wb') as fh:
        fh.write(struct.pack('<i', nr_frames))
        fh.write(struct.pack('<i', nr_pixels))
        fh.write(n_peaks_arr.tobytes())
        fh.write(offsets.tobytes())
        for f in range(nr_frames):
            if frame_pixel_data[f] is not None:
                for (py, pz) in frame_pixel_data[f]:
                    n_px = len(py)
                    fh.write(struct.pack('<i', n_px))
                    interleaved = np.empty(n_px * 2, dtype=np.int16)
                    interleaved[0::2] = np.asarray(py, dtype=np.int16)
                    interleaved[1::2] = np.asarray(pz, dtype=np.int16)
                    fh.write(interleaved.tobytes())


# ── Readers ───────────────────────────────────────────────────────────────────
# The modern MIDAS FF-HEDM pipeline no longer writes one *_PS.csv per frame; it
# writes a single consolidated AllPeaks_PS.bin (+ AllPeaks_PX.bin).  These
# readers recover the per-frame peak tables from that binary so downstream
# consumers (e.g. peakbank creation for SR training) work regardless of whether
# the analysis directory holds per-frame CSVs or the consolidated binary.
# Layout is the inverse of write_allpeaks_ps_bin above and matches
# MIDAS/utils/UnpackConsolidatedPeaks.py.

def read_allpeaks_ps_bin(filepath):
    """Read AllPeaks_PS.bin into per-frame peak arrays.

    Args:
        filepath: Path to AllPeaks_PS.bin.

    Returns:
        (nr_frames, frame_peaks) where frame_peaks is a list of length
        nr_frames.  Each element is either an ndarray of shape (n_peaks, 29)
        (dtype float64) for frames that have peaks, or None for empty frames.
    """
    with open(filepath, "rb") as fh:
        nr_frames = struct.unpack("<i", fh.read(4))[0]
        n_peaks_arr = np.frombuffer(fh.read(4 * nr_frames), dtype=np.int32).copy()
        # offsets are recomputable from n_peaks; read past them for correctness.
        _offsets = np.frombuffer(fh.read(8 * nr_frames), dtype=np.int64)
        raw = fh.read()

    ps_data = np.frombuffer(raw, dtype=np.float64)
    expected = int(n_peaks_arr.sum()) * N_PEAK_COLS
    if ps_data.size < expected:
        # Partial/truncated file: only trust complete rows.
        n_complete_rows = ps_data.size // N_PEAK_COLS
        ps_data = ps_data[: n_complete_rows * N_PEAK_COLS]
    ps_data = ps_data.reshape(-1, N_PEAK_COLS)

    frame_peaks = [None] * nr_frames
    idx = 0
    for f in range(nr_frames):
        n = int(n_peaks_arr[f])
        if n <= 0:
            continue
        if idx + n > ps_data.shape[0]:
            n = max(0, ps_data.shape[0] - idx)
            if n == 0:
                break
        frame_peaks[f] = ps_data[idx: idx + n].copy()
        idx += n
    return nr_frames, frame_peaks


def read_allpeaks_ps_frame_dfs(filepath):
    """Read AllPeaks_PS.bin into a dict of per-frame pandas DataFrames.

    The DataFrame columns match the per-frame *_PS.csv header exactly, so the
    result is a drop-in substitute for `pd.read_csv(<frame>_PS.csv, sep='\\t')`.

    Args:
        filepath: Path to AllPeaks_PS.bin.

    Returns:
        dict mapping 0-based frame index -> DataFrame.  Frames with no peaks
        are omitted (mirroring how empty frames yield empty CSVs).
    """
    import pandas as pd

    nr_frames, frame_peaks = read_allpeaks_ps_bin(filepath)
    out = {}
    for f in range(nr_frames):
        if frame_peaks[f] is None:
            continue
        out[f] = pd.DataFrame(frame_peaks[f], columns=PS_COLUMNS)
    return out
