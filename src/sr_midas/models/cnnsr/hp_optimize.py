"""Hyperparameter optimization for CNNSR using Optuna.

Refactored from: SRML-HEDM/CNNSR/hp_optimize.py
Requires: pip install sr-midas[optuna]
"""

import os
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sr_midas.models.cnnsr.dataset import trainData_CNNSR

SEP = os.sep

# ============================================================================
# Hyperparameter search space — modify as needed
# ============================================================================

MODEL_HYPERPARAMS = {
    "n_layers": {"min": 3, "max": 8},
    "kernel_size": {"choices": [3, 5, 7]},
    "out_channels": {"min": 16, "max": 128}
}

TRAINING_HYPERPARAMS = {
    "optimizer": {"choices": ["Adam", "RMSprop", "SGD"]},
    "lr": {"min": 1e-5, "max": 1e-1, "log": True},
    "batch_size": {"choices": [32, 64, 128, 256, 512]},
    "weight_decay": {"min": 1e-6, "max": 1e-3, "log": True},
    "momentum": 0.9
}

# ============================================================================


def run_hp_optimize(args):
    """Run Optuna hyperparameter optimization for CNNSR.
    Requires optuna: pip install sr-midas[optuna]
    Args:
        args: namespace or dict with keys:
            pst (str): path to patchstore
            outSRx (int): output SR factor
            inSRx (int): input SR factor
            n_trials (int): number of Optuna trials
            n_itrs (int): epochs per trial
            trainFrac (float; default=0.8): training fraction
            useRch (str; default="false"): use R channel
            useEtach (str; default="false"): use Eta channel
            inPstPath (str; default="none"): input patchstore path
            outPstPath (str; default="none"): output patchstore path
            nwork (int; default=1): DataLoader workers
            patience (int; default=5): early stopping patience
            init_method (str; default="kaiming_normal"): weight init method
            study_name (str; default="cnnsr_optimization"): Optuna study name
            n_startup_trials (int; default=5): pruner startup trials
            n_warmup_steps (int; default=10): pruner warmup steps
            save_results (str; default="true"): whether to save results
            output_base_dir (str; default="optuna_results"): base output directory
    """

    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError:
        raise ImportError(
            "optuna is required for hyperparameter optimization. "
            "Install with: pip install sr-midas[optuna]"
        ) from None

    if isinstance(args, dict):
        import argparse
        args = argparse.Namespace(**args)

    pst_path = args.pst
    outSRx, inSRx = args.outSRx, args.inSRx

    useRch = str(getattr(args, 'useRch', 'false')).lower() in ["true", "1", "t", "yes", "y"]
    useEtach = str(getattr(args, 'useEtach', 'false')).lower() in ["true", "1", "t", "yes", "y"]

    inPstPath_raw = getattr(args, 'inPstPath', 'none')
    outPstPath_raw = getattr(args, 'outPstPath', 'none')
    inPstPath = None if str(inPstPath_raw).lower() in ["none", "n", "na"] else inPstPath_raw
    outPstPath = None if str(outPstPath_raw).lower() in ["none", "n", "na"] else outPstPath_raw

    n_itrs = args.n_itrs
    patience = getattr(args, 'patience', 5)
    trainFrac = getattr(args, 'trainFrac', 0.8)
    nwork = getattr(args, 'nwork', 1)
    init_method = getattr(args, 'init_method', 'kaiming_normal')

    n_trials = args.n_trials
    study_name = getattr(args, 'study_name', 'cnnsr_optimization')
    n_startup_trials = getattr(args, 'n_startup_trials', 5)
    n_warmup_steps = getattr(args, 'n_warmup_steps', 10)

    save_results = str(getattr(args, 'save_results', 'true')).lower() in ["true", "1", "t", "yes"]
    output_base_dir = getattr(args, 'output_base_dir', 'optuna_results')

    if torch.cuda.is_available():
        torch_devs = torch.device("cuda")
    elif torch.backends.mps.is_available():
        torch_devs = torch.device("mps")
    else:
        torch_devs = torch.device("cpu")

    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = f"{output_base_dir}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    all_trials_data = []

    print("=" * 60, flush=True)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION", flush=True)
    print("=" * 60, flush=True)
    print(f"Device: {torch_devs} | Trials: {n_trials} | Epochs/trial: {n_itrs}", flush=True)
    print("=" * 60, flush=True)

    def define_model(trial):
        n_layers = trial.suggest_int("n_layers",
                                     MODEL_HYPERPARAMS["n_layers"]["min"],
                                     MODEL_HYPERPARAMS["n_layers"]["max"])
        layers = []
        in_channels = 1
        model_architecture = {"n_layers": n_layers, "layers": []}

        for i in range(n_layers):
            kernel_size = trial.suggest_categorical(f"kernel_size_{i}", MODEL_HYPERPARAMS["kernel_size"]["choices"])
            if i < n_layers - 1:
                out_channels = trial.suggest_int(f"out_channels_{i}",
                                                 MODEL_HYPERPARAMS["out_channels"]["min"],
                                                 MODEL_HYPERPARAMS["out_channels"]["max"])
                layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
                layers.append(torch.nn.ReLU())
                activation = "ReLU"
            else:
                out_channels = 1
                layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2))
                layers.append(torch.nn.Sigmoid())
                activation = "Sigmoid"

            model_architecture["layers"].append({
                "layer_index": i, "in_channels": in_channels, "out_channels": out_channels,
                "kernel_size": kernel_size, "padding": kernel_size // 2, "activation": activation
            })
            in_channels = out_channels

        return torch.nn.Sequential(*layers), model_architecture

    def get_data(batch_size):
        ds_train = trainData_CNNSR(pst_path=pst_path, srfac_out=outSRx, srfac_in=inSRx,
                                   use_R_channel=useRch, use_Eta_channel=useEtach,
                                   normalize_R_channel=True, normalize_Eta_channel=True,
                                   use="train", train_frac=trainFrac,
                                   pst_path_X=inPstPath, pst_path_Y=outPstPath)

        ds_valid = trainData_CNNSR(pst_path=pst_path, srfac_out=outSRx, srfac_in=inSRx,
                                   use_R_channel=useRch, use_Eta_channel=useEtach,
                                   normalize_R_channel=True, normalize_Eta_channel=True,
                                   use="test", train_frac=trainFrac,
                                   pst_path_X=inPstPath, pst_path_Y=outPstPath)

        dl_train = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True,
                              num_workers=nwork, prefetch_factor=batch_size,
                              drop_last=True, pin_memory=True)

        dl_valid = DataLoader(dataset=ds_valid, batch_size=batch_size, shuffle=False,
                              num_workers=nwork, prefetch_factor=batch_size,
                              drop_last=False, pin_memory=True)

        return dl_train, dl_valid

    def save_trial_results(trial_number, all_hyperparams, model_architecture, training_history,
                           final_loss, trial_state, elapsed_time, best_epoch=None):
        if not save_results:
            return

        trial_dir = os.path.join(output_dir, f"trial_{trial_number:03d}")
        os.makedirs(trial_dir, exist_ok=True)

        with open(os.path.join(trial_dir, "hyperparameters.json"), 'w') as f:
            json.dump(all_hyperparams, f, indent=2)

        with open(os.path.join(trial_dir, "model_architecture.json"), 'w') as f:
            json.dump(model_architecture, f, indent=2)

        pd.DataFrame(training_history).to_csv(os.path.join(trial_dir, "training_history.csv"), index=False)

        best_val_loss = min(training_history['val_loss']) if training_history['val_loss'] else None
        best_val_epoch = training_history['val_loss'].index(best_val_loss) if best_val_loss is not None else None

        summary = {
            "trial_number": trial_number, "final_loss": final_loss,
            "best_val_loss": best_val_loss, "best_val_epoch": best_val_epoch,
            "trial_state": trial_state, "elapsed_time_seconds": elapsed_time,
            "total_epochs": len(training_history['epoch']),
            "hyperparameters": all_hyperparams, "model_architecture": model_architecture
        }
        with open(os.path.join(trial_dir, "trial_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        all_trials_data.append({**summary, **all_hyperparams})
        print(f"Trial {trial_number} results saved to {trial_dir}", flush=True)

    def objective(trial):
        trial_start_time = time.time()
        trial_number = trial.number
        training_history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'learning_rate': []}

        try:
            model, model_architecture = define_model(trial)

            optimizer_name = trial.suggest_categorical("optimizer", TRAINING_HYPERPARAMS["optimizer"]["choices"])
            lr = trial.suggest_float("lr", TRAINING_HYPERPARAMS["lr"]["min"], TRAINING_HYPERPARAMS["lr"]["max"],
                                     log=TRAINING_HYPERPARAMS["lr"]["log"])
            batch_size = trial.suggest_categorical("batch_size", TRAINING_HYPERPARAMS["batch_size"]["choices"])
            weight_decay = trial.suggest_float("weight_decay", TRAINING_HYPERPARAMS["weight_decay"]["min"],
                                               TRAINING_HYPERPARAMS["weight_decay"]["max"],
                                               log=TRAINING_HYPERPARAMS["weight_decay"]["log"])

            all_hyperparams = trial.params.copy()

            model = nn.DataParallel(model)
            model = model.to(torch_devs)

            def init_weights(m):
                if isinstance(m, nn.Conv2d):
                    if init_method == "kaiming_normal":
                        torch.nn.init.kaiming_normal_(m.weight)
                    elif init_method == "xavier_normal":
                        torch.nn.init.xavier_normal_(m.weight)
                    elif init_method == "he_normal":
                        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            model.apply(init_weights)

            if optimizer_name == "SGD":
                optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                                      momentum=TRAINING_HYPERPARAMS["momentum"])
            else:
                optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)

            criterion = torch.nn.MSELoss()
            dl_train, dl_valid = get_data(batch_size)

            best_val_loss = float('inf')
            best_epoch = -1
            patience_counter = 0

            print(f"Starting Trial {trial_number} | Optimizer: {optimizer_name}, lr={lr:.2e}, batch={batch_size}", flush=True)

            for epoch in range(n_itrs):
                model.train()
                train_losses = []
                for data, target in dl_train:
                    data, target = data.to(torch_devs), target.to(torch_devs)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())

                model.eval()
                val_losses = []
                with torch.no_grad():
                    for data, target in dl_valid:
                        data, target = data.to(torch_devs), target.to(torch_devs)
                        output = model(data)
                        val_losses.append(criterion(output, target).item())

                train_loss = np.mean(train_losses)
                val_loss = np.mean(val_losses)

                training_history['epoch'].append(epoch)
                training_history['train_loss'].append(train_loss)
                training_history['val_loss'].append(val_loss)
                training_history['learning_rate'].append(lr)

                print(f"Trial {trial_number}, Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}", flush=True)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"Trial {trial_number}: Early stopping at epoch {epoch}", flush=True)
                    break

                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            elapsed_time = time.time() - trial_start_time
            save_trial_results(trial_number, all_hyperparams, model_architecture,
                               training_history, best_val_loss, "COMPLETE", elapsed_time, best_epoch)
            return best_val_loss

        except optuna.exceptions.TrialPruned:
            elapsed_time = time.time() - trial_start_time
            save_trial_results(trial_number, trial.params.copy(),
                               model_architecture if 'model_architecture' in locals() else {},
                               training_history, None, "PRUNED", elapsed_time)
            raise

        except Exception as e:
            elapsed_time = time.time() - trial_start_time
            save_trial_results(trial_number, trial.params.copy(),
                               model_architecture if 'model_architecture' in locals() else {},
                               training_history, None, "FAIL", elapsed_time)
            print(f"Trial {trial_number} failed: {e}", flush=True)
            raise

    def save_study_summary(study):
        if not save_results:
            return

        study_summary = {
            "study_name": study.study_name,
            "n_trials": len(study.trials),
            "best_value": study.best_value if study.best_trial else None,
            "best_params": study.best_params if study.best_trial else None,
            "best_trial_number": study.best_trial.number if study.best_trial else None,
        }
        with open(os.path.join(output_dir, "study_summary.json"), 'w') as f:
            json.dump(study_summary, f, indent=2)

        pd.DataFrame(all_trials_data).to_csv(os.path.join(output_dir, "all_trials_summary.csv"), index=False)

        import pickle
        with open(os.path.join(output_dir, "optuna_study.pkl"), 'wb') as f:
            pickle.dump(study, f)

        print(f"Study summary saved to {output_dir}", flush=True)

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials,
                                           n_warmup_steps=n_warmup_steps)
    )
    print("Study created. Initiating optimization...", flush=True)

    try:
        study.optimize(objective, n_trials=n_trials)
    except KeyboardInterrupt:
        print("Optimization interrupted by user", flush=True)

    save_study_summary(study)

    print("Study finished.", flush=True)
    print("=" * 60, flush=True)

    if study.best_trial:
        print("Best trial:", flush=True)
        print(f"  Value: {study.best_trial.value}", flush=True)
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}", flush=True)
    else:
        print("No successful trials completed", flush=True)

    if save_results:
        print(f"\nAll results saved to: {output_dir}", flush=True)
