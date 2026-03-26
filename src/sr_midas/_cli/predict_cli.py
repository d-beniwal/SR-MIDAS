"""CLI for running super-resolution prediction on a patchstore.

Single-model mode (default):
    sr-midas-predict cnnsr -pstPath data.h5 -inSRx 1 -outSRx 8 \
        -modDir /path/to/model -modItr 1300 -saveDir /out

Cascaded mode (x1→x2→x4→x8 via three separate models):
    sr-midas-predict cnnsr -pstPath data.h5 -inSRx 1 -outSRx 8 -cascade \
        -x2ModDir /model_x12 -x2ModItr 4975 \
        -x4ModDir /model_x24 -x4ModItr 999  \
        -x8ModDir /model_x48 -x8ModItr 999  \
        -saveDir /out

To add a new model type, add a subparser below and a corresponding dispatch branch in main().
"""

import argparse
import os

import numpy as np


# ---------------------------------------------------------------------------
# Per-model argument definitions
# ---------------------------------------------------------------------------

def _add_cnnsr_args(p):
    # --- common ---
    p.add_argument("-pstPath", type=str, required=True,
                   help="Path to input patchstore (.h5)")
    p.add_argument("-inSRx", type=int, required=True,
                   help="SR factor of input patches (e.g. 1)")
    p.add_argument("-outSRx", type=int, required=True,
                   help="Target SR factor of output predictions (e.g. 8)")
    p.add_argument("-saveDir", type=str, required=True,
                   help="Directory to save predictions (.npy)")
    p.add_argument("-saveName", type=str, default="predictions.npy",
                   help="Filename for saved predictions")
    p.add_argument("-bsz", type=int, default=200,
                   help="Batch size for prediction")
    p.add_argument("-cascade", action="store_true",
                   help="Use cascaded x1→x2→x4→x8 prediction (requires -x2/x4/x8ModDir args)")

    # --- single-model args (used when -cascade is not set) ---
    p.add_argument("-modDir", type=str, default=None,
                   help="Path to trained model directory (single-model mode)")
    p.add_argument("-modItr", type=int, default=None,
                   help="Model checkpoint iteration (single-model mode)")

    # --- cascade-specific args ---
    p.add_argument("-x2ModDir", type=str, default=None,
                   help="x1→x2 model directory (cascade mode)")
    p.add_argument("-x2ModItr", type=int, default=None,
                   help="x1→x2 model iteration (cascade mode)")
    p.add_argument("-x4ModDir", type=str, default=None,
                   help="x2→x4 model directory (cascade mode)")
    p.add_argument("-x4ModItr", type=int, default=None,
                   help="x2→x4 model iteration (cascade mode)")
    p.add_argument("-x8ModDir", type=str, default=None,
                   help="x4→x8 model directory (cascade mode)")
    p.add_argument("-x8ModItr", type=int, default=None,
                   help="x4→x8 model iteration (cascade mode)")


# ---------------------------------------------------------------------------
# Per-model dispatch functions
# ---------------------------------------------------------------------------

def _run_cnnsr(args):
    import torch
    from sr_midas.models.cnnsr import predict_CNNSR, predict_CNNSR_singleMod
    from sr_midas.data.patchstore import load_patchstore_h5data

    if torch.cuda.is_available():
        torch_devs = torch.device("cuda")
    elif torch.backends.mps.is_available():
        torch_devs = torch.device("mps")
    else:
        torch_devs = torch.device("cpu")

    patch_arr, *_ = load_patchstore_h5data(args.pstPath)
    X = patch_arr[f"SRx{args.inSRx}"]

    if args.cascade:
        missing = [f for f, v in [("-x2ModDir", args.x2ModDir), ("-x2ModItr", args.x2ModItr),
                                   ("-x4ModDir", args.x4ModDir), ("-x4ModItr", args.x4ModItr),
                                   ("-x8ModDir", args.x8ModDir), ("-x8ModItr", args.x8ModItr)]
                   if v is None]
        if missing:
            raise ValueError(f"Cascade mode requires: {', '.join(missing)}")

        mods_to_use = {
            "SRx2": {"mod_dir": args.x2ModDir, "mod_itr": args.x2ModItr},
            "SRx4": {"mod_dir": args.x4ModDir, "mod_itr": args.x4ModItr},
            "SRx8": {"mod_dir": args.x8ModDir, "mod_itr": args.x8ModItr},
        }
        SRx2, SRx4, SRx8 = predict_CNNSR(X, mods_to_use, print_run_time=True, batch_size=args.bsz)

        os.makedirs(args.saveDir, exist_ok=True)
        stem = args.saveName.replace(".npy", "")
        for tag, arr in [("SRx2", SRx2), ("SRx4", SRx4), ("SRx8", SRx8)]:
            path = os.path.join(args.saveDir, f"{stem}_{tag}.npy")
            np.save(path, arr)
            print(f"Saved: {path}", flush=True)

    else:
        if args.modDir is None or args.modItr is None:
            raise ValueError("Single-model mode requires -modDir and -modItr")

        mods_to_use = {
            "SRx8": {"mod_dir": args.modDir, "mod_itr": args.modItr},
        }
        predictions = predict_CNNSR_singleMod(X, mods_to_use, print_run_time=True, batch_size=args.bsz)

        os.makedirs(args.saveDir, exist_ok=True)
        save_path = os.path.join(args.saveDir, args.saveName)
        np.save(save_path, predictions)
        print(f"Saved: {save_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run super-resolution prediction on a patchstore.",
        epilog="Example: sr-midas-predict cnnsr -pstPath data.h5 -inSRx 1 -outSRx 8 -modDir /model -modItr 1300 -saveDir /out"
    )
    sub = parser.add_subparsers(dest="model", required=True, metavar="MODEL",
                                help="model type to use for prediction")

    _add_cnnsr_args(sub.add_parser("cnnsr", help="Predict using a trained CNNSR model"))

    # New model example:
    # _add_unet_args(sub.add_parser("unet", help="Predict using a trained UNet model"))

    args = parser.parse_args()

    if args.model == "cnnsr":
        _run_cnnsr(args)


if __name__ == "__main__":
    main()
