"""CLI wrapper for run_sr_process."""

import argparse

from sr_midas.pipeline.sr_process import run_sr_process


def main():
    parser = argparse.ArgumentParser(
        description="Run the MIDAS super-resolution pipeline."
    )
    parser.add_argument("-midasZarrDir", type=str, required=True,
                        help="Path to MIDAS .zip zarr directory")
    parser.add_argument("-srfac", type=int, default=8,
                        help="Super-resolution factor (2, 4, or 8)")
    parser.add_argument("-SRconfig", type=str, default=None,
                        help="Path to SR config .json or .txt file (default: bundled cnnsr_sr_config.json)")
    parser.add_argument("-saveSRpatches", type=int, default=1,
                        help="Save SR patches (1=yes, 0=no)")
    parser.add_argument("-saveFrameGoodCoords", type=int, default=1,
                        help="Save frame good coordinates (1=yes, 0=no)")

    args = parser.parse_args()

    run_sr_process(
        midasZarrDir=args.midasZarrDir,
        srfac=args.srfac,
        SRconfig_path=args.SRconfig,
        saveSRpatches=args.saveSRpatches,
        saveFrameGoodCoords=args.saveFrameGoodCoords,
    )


if __name__ == "__main__":
    main()
