"""CLI for hyperparameter optimization of super-resolution models.

Usage:
    sr-midas-hp-optimize cnnsr -pst data.h5 -inSRx 1 -outSRx 8 -n_trials 50 -n_itrs 20

To add a new model type, add a subparser below and a corresponding dispatch branch in main().
"""

import argparse


# ---------------------------------------------------------------------------
# Per-model argument definitions
# ---------------------------------------------------------------------------

def _add_cnnsr_args(p):
    p.description = "Hyperparameter optimization for CNNSR using Optuna. Requires: pip install sr-midas[optuna]"
    p.add_argument("-pst", type=str, required=True,
                   help="Path to patchstore")
    p.add_argument("-outSRx", type=int, required=True,
                   help="Output SR factor")
    p.add_argument("-inSRx", type=int, required=True,
                   help="Input SR factor")
    p.add_argument("-n_trials", type=int, required=True,
                   help="Number of Optuna trials")
    p.add_argument("-n_itrs", type=int, required=True,
                   help="Epochs per trial")
    p.add_argument("-trainFrac", type=float, default=0.8,
                   help="Training fraction")
    p.add_argument("-useRch", type=str, default="false",
                   help="Use R channel (true/false)")
    p.add_argument("-useEtach", type=str, default="false",
                   help="Use Eta channel (true/false)")
    p.add_argument("-inPstPath", type=str, default="none",
                   help="Input patchstore path")
    p.add_argument("-outPstPath", type=str, default="none",
                   help="Output patchstore path")
    p.add_argument("-nwork", type=int, default=1,
                   help="DataLoader workers")
    p.add_argument("-patience", type=int, default=5,
                   help="Early stopping patience")
    p.add_argument("-init_method", type=str, default="kaiming_normal",
                   help="Weight init method")
    p.add_argument("-study_name", type=str, default="cnnsr_optimization",
                   help="Optuna study name")
    p.add_argument("-n_startup_trials", type=int, default=5,
                   help="Pruner startup trials")
    p.add_argument("-n_warmup_steps", type=int, default=10,
                   help="Pruner warmup steps")
    p.add_argument("-save_results", type=str, default="true",
                   help="Whether to save results (true/false)")
    p.add_argument("-output_base_dir", type=str, default="optuna_results",
                   help="Base output directory")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for super-resolution models.",
        epilog="Example: sr-midas-hp-optimize cnnsr -pst data.h5 -inSRx 1 -outSRx 8 -n_trials 50 -n_itrs 20"
    )
    sub = parser.add_subparsers(dest="model", required=True, metavar="MODEL",
                                help="model type to optimize")

    _add_cnnsr_args(sub.add_parser("cnnsr", help="Optimize CNNSR (requires sr-midas[optuna])"))

    # New model example:
    # _add_unet_args(sub.add_parser("unet", help="Optimize UNet (requires sr-midas[optuna])"))

    args = parser.parse_args()

    if args.model == "cnnsr":
        from sr_midas.models.cnnsr import run_hp_optimize
        run_hp_optimize(args)


if __name__ == "__main__":
    main()
