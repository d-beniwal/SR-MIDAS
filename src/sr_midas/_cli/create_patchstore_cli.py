"""CLI wrapper for create_patchstore."""

import argparse

from sr_midas.synthesis.patchstore_gen import create_patchstore


def main():
    parser = argparse.ArgumentParser(
        description="Create a synthetic patchstore HDF5 file."
    )
    parser.add_argument("-config", type=str, default=None,
                        help="Path to JSON config file (merged with CLI args)")
    parser.add_argument("-peakbankPath", type=str, help="Path to peakbank CSV")
    parser.add_argument("-savedir", type=str, help="Directory to save patchstore")
    parser.add_argument("-savename", type=str, help="Filename for patchstore (.h5)")
    parser.add_argument("-cvsz", type=int, help="Canvas size")
    parser.add_argument("-srfac_list", type=int, nargs="+",
                        help="List of SR factors e.g. 1 2 4 8")
    parser.add_argument("-n_patches", type=int, help="Number of patches to generate")
    parser.add_argument("-patch_size", type=int, help="Patch size in low-res pixels")
    parser.add_argument("-I_thresh", type=float, help="Intensity threshold")
    parser.add_argument("-n_peaks_per_patch", type=int, default=1,
                        help="Max peaks per patch")
    parser.add_argument("-useRch", type=str, default="false",
                        help="Include R channel (true/false)")
    parser.add_argument("-useEtach", type=str, default="false",
                        help="Include Eta channel (true/false)")
    parser.add_argument("-nwork", type=int, default=1,
                        help="Number of parallel workers")

    args = parser.parse_args()
    create_patchstore(args)


if __name__ == "__main__":
    main()
