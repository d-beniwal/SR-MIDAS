"""CLI for training super-resolution models.

Usage:
    sr-midas-train cnnsr -expName myexp -pst data.h5 -arch 64-5-r_1-5-s ...

To add a new model type, add a subparser below and a corresponding dispatch branch in main().
"""

import argparse


# ---------------------------------------------------------------------------
# Per-model argument definitions
# ---------------------------------------------------------------------------

def _add_cnnsr_args(p):
    p.add_argument("-expName", type=str, required=True,
                   help="Experiment name (used as output directory name)")
    p.add_argument("-pst", type=str, required=True,
                   help="Path to patchstore .h5 file")
    p.add_argument("-inSRx", type=int, required=True,
                   help="Input patch SR factor")
    p.add_argument("-outSRx", type=int, required=True,
                   help="Output patch SR factor")
    p.add_argument("-useRch", type=str, default="false",
                   help="Include Radius channel (true/false)")
    p.add_argument("-useEtach", type=str, default="false",
                   help="Include Eta channel (true/false)")
    p.add_argument("-arch", type=str, required=True,
                   help="CNN architecture string e.g. 64-5-r_32-5-r_1-5-s")
    p.add_argument("-lr", type=float, required=True,
                   help="Learning rate")
    p.add_argument("-lossF", type=str, default="mse",
                   help="Loss function: mse or mae")
    p.add_argument("-mbsz", type=int, required=True,
                   help="Mini batch size")
    p.add_argument("-maxItr", type=int, required=True,
                   help="Max training iterations (epochs)")
    p.add_argument("-trainFrac", type=float, default=0.8,
                   help="Training dataset fraction")
    p.add_argument("-nwork", type=int, default=1,
                   help="Number of DataLoader workers")
    p.add_argument("-ecVal", type=float, required=True,
                   help="Error convergence value")
    p.add_argument("-ecItr", type=int, required=True,
                   help="Convergence check interval (epochs)")
    p.add_argument("-inPstPath", type=str, default=None,
                   help="Input patchstore path (if separate from -pst)")
    p.add_argument("-outPstPath", type=str, default=None,
                   help="Target patchstore path (if separate from -pst)")
    p.add_argument("-loadChkpt", type=str, default=None,
                   help="Path to pre-trained checkpoint for initialization")
    p.add_argument("-trainOutDir", type=str, required=True,
                   help="Output directory for trained models")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a super-resolution model.",
        epilog="Example: sr-midas-train cnnsr -expName myexp -pst data.h5 ..."
    )
    sub = parser.add_subparsers(dest="model", required=True, metavar="MODEL",
                                help="model type to train")

    _add_cnnsr_args(sub.add_parser("cnnsr", help="Train a CNNSR super-resolution model"))

    # New model example (uncomment and fill in when adding a new model):
    # _add_unet_args(sub.add_parser("unet", help="Train a UNet super-resolution model"))

    args = parser.parse_args()

    if args.model == "cnnsr":
        from sr_midas.models.cnnsr import train_cnnsr
        train_cnnsr(args)


if __name__ == "__main__":
    main()
