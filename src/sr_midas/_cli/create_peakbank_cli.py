"""CLI wrapper for create_peakbank."""

import argparse
import json

from sr_midas.synthesis.peakbank import create_peakbank


def main():
    parser = argparse.ArgumentParser(
        description="Create a peakbank of MIDAS-fitted peaks."
    )
    parser.add_argument(
        "-config", type=str, required=True,
        help="Path to JSON config file (or use individual flags below)"
    )
    args = parser.parse_args()

    create_peakbank(args.config)


if __name__ == "__main__":
    main()
