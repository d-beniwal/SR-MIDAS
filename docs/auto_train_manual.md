# SR-MIDAS Automated Training Manual

## Overview

`sr-midas-auto-train` is a single-command workflow that builds custom CNNSR
super-resolution models from your own MIDAS diffraction data. Point it at any
MIDAS FF-HEDM reconstruction folder and it will either **train new models from
scratch** or **fine-tune the bundled pretrained models** on that dataset's
fitted peaks. It automates the entire pipeline that would otherwise require
running 7 separate commands:

```
MIDAS recon dir  -->  peakbank  -->  patchstore  -->  train/finetune x2
    -->  pred-pst x2  -->  train/finetune x4  -->  pred-pst x4
    -->  train/finetune x8  -->  sr_config.json
```

The final output is an `sr_config.json` file that you pass directly to
`sr-midas-process` to use your newly built models instead of the default
bundled ones.

### What's new

- **`mode`: `scratch` (default) or `finetune`.** Fine-tune warm-starts each
  cascade stage from an existing checkpoint (the bundled pretrained models by
  default, or any previous run via `base_models`) and needs far fewer epochs.
- **Reads modern MIDAS output.** Recent MIDAS no longer writes one
  `Temp/<stem>_<frame>_PS.csv` per frame — it writes a single consolidated
  `Temp/AllPeaks_PS.bin`. The workflow reads **either** layout automatically
  (`peak_source: auto`).
- **Adaptive quality gate (`err_percentile`).** The peak reconstruction error
  is on a dataset-dependent scale, so a fixed absolute cutoff (the old `err_cut`
  = 0.25) can reject essentially every peak on a new detector. By default the
  workflow now keeps the cleanest `err_percentile`% (default 60) of peaks for
  **this** dataset instead. Set `err_percentile: null` for the legacy absolute
  cutoff.

---

## Prerequisites

- SR-MIDAS package installed (`pip install -e .` from the SR-MIDAS root)
- A completed MIDAS analysis directory containing:
  - A `.MIDAS.zip` file (zarr archive with detector data)
  - A `Temp/` folder with per-frame peak CSV files (from MIDAS fitting)
- A GPU is strongly recommended for training (CUDA). CPU training works but is
  significantly slower.

---

## Quick Start

### 1. Create a config file

Create a JSON file (e.g., `my_train_config.json`) with at minimum:

```json
{
    "midas_dir": ["/path/to/your/midas_data_directory"],
    "output_dir": "/path/to/training_output"
}
```

`midas_dir` is a list of one or more directories, each containing a
`.MIDAS.zip` file and a `Temp/` folder with peak CSVs.

### 2. Run the training

```bash
sr-midas-auto-train -config my_train_config.json
# equivalent, if the console script isn't on PATH yet (e.g. editable install
# from before this entry point was registered — no reinstall needed):
python -m sr_midas._cli.auto_train_cli -config my_train_config.json
```

This will take a while (minutes to hours depending on your GPU and dataset
size). Progress is printed to the terminal and logged to
`<output_dir>/auto_train.log`.

### 3. Use the trained models

Once complete, the tool prints the path to the generated config. Use it with:

```bash
sr-midas-process -midasZarrDir /path/to/data -SRconfig /path/to/training_output/sr_config.json
```

That's it. The SR pipeline will use your custom-trained models instead of the
bundled defaults.

---

## Configuration Reference

All parameters below have sensible defaults that match the hyperparameters used
to train the bundled pretrained models. You only need to override them if you
want different behavior.

### Required Parameters

| Key | Type | Description |
|-----|------|-------------|
| `midas_dir` | list of str | Paths to MIDAS data directories. Each must contain a `.MIDAS.zip` file and `Temp/` folder with peak CSVs. Can also be a single string. |
| `output_dir` | str | Directory where all output artifacts will be saved. Created if it doesn't exist. |

### Mode & fine-tuning parameters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | str | `"scratch"` | `"scratch"` trains from random init; `"finetune"` warm-starts from existing checkpoints. |
| `base_models` | null / str / dict | null | Fine-tune base. `null` → bundled pretrained models. A path to an `sr_config.json` → warm-start from a previous run. Or an explicit dict `{"SRx2": {"mod_dir","mod_itr"}, "SRx4": {...}, "SRx8": {...}}`. Relative `mod_dir`s resolve against the bundled `pretrained/` dir. |

In `finetune` mode, if you do not override them, these gentler defaults apply:
`lr` = 2e-5, `max_itr_x2` = `max_itr_x4` = `max_itr_x8` = 300. The per-stage CNN
architecture is forced to match each base checkpoint (you cannot fine-tune into
a different architecture).

### Peakbank Parameters

Control how peaks are extracted and filtered from your MIDAS data.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `peak_source` | str | `"auto"` | Where to read MIDAS-fitted peaks: `"auto"` (per-frame `Temp/*_PS.csv` if present, else consolidated `Temp/AllPeaks_PS.bin`), `"csv"`, or `"consolidated"`. |
| `err_percentile` | float / null | 60 | Adaptive quality gate: keep the cleanest N% of peaks by reconstruction error (computed from this dataset). Set `null` to use the absolute `err_cut` / `peak_recon_err_threshold` instead. |
| `max_frames` | int / null | null | Cap frames read per dataset when building the peakbank (null = all). Useful for quick smoke tests. |
| `maxModInit` | int | 3 | Max model re-initialisations in the training convergence gate before proceeding with current weights (prevents infinite loops on quick configs). |
| `cvsz` | int | 50 | Canvas size (pixels) for peak reconstruction during quality assessment. |
| `srfac` | int | 16 | Super-resolution factor used internally during peak reconstruction. |
| `I_thresh` | float | 30 | Intensity threshold. Detector pixels below this value are zeroed out. |
| `peak_recon_err_threshold` | float | 0.25 | Maximum allowed reconstruction error. Peaks with higher error are discarded. Lower values produce a cleaner peakbank but fewer peaks. |

### Patchstore Parameters

Control synthetic training data generation.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_patches` | int | 60000 | Number of synthetic patches to generate. More patches = better training data but longer generation and training time. |
| `patch_size` | int | 20 | Patch size in low-resolution pixels. Must match `lrsz` in sr_config (default 20). |
| `srfac_list` | str | "1-2-4-8" | Dash-separated list of SR factors to generate patches for. |
| `n_peaks_per_patch` | str | "1-2-3-4-5" | Dash-separated list of possible peak counts per patch. Each patch randomly gets one of these counts. |
| `peak_sep_min` | float | 3.0 | Minimum separation (pixels) between peaks in a patch. |
| `peak_sep_max` | float | 15.0 | Maximum separation (pixels) between peaks in a patch. |
| `var_R` | float | 3.0 | Maximum radial variation from the ring radius when placing peaks. |
| `err_cut` | float | 0.25 | Reconstruction error cutoff for filtering the peakbank before patchstore creation. |
| `integ_int_cut` | float | 0.0 | Minimum integrated intensity for peakbank filtering. |
| `midas_I_thresh` | float | 30.0 | MIDAS intensity threshold used during patchstore generation. |
| `peak_I_min` | float | 10.0 | Minimum peak intensity for synthetic patches. |
| `peak_I_max` | float | 100000.0 | Maximum peak intensity for synthetic patches. |
| `sr_I_thresh_fac` | float | 0.001 | Source patch intensity threshold factor. |

### Training Parameters

Control CNN model training hyperparameters.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `arch` | str | "256-5-r_128-5-r_64-5-r_32-5-r_16-5-r_8-5-r_1-5-s" | CNN architecture string. Format: `channels-kernel-activation` separated by `_`. Activations: `r`=ReLU, `s`=Sigmoid, `lr`=LeakyReLU, `t`=Tanh. |
| `lr` | float | 0.0001 | Learning rate for the Adam optimizer. |
| `loss_fn` | str | "mse" | Loss function: `"mse"` (mean squared error) or `"mae"` (mean absolute error). |
| `batch_size` | int | 512 | Default mini-batch size for training. Reduce if you run out of GPU memory. |
| `batch_size_x2` | int / null | null | Per-stage batch size for the x1->x2 model. Falls back to `batch_size` when null. |
| `batch_size_x4` | int / null | null | Per-stage batch size for the x2->x4 model. Falls back to `batch_size` when null. |
| `batch_size_x8` | int | 128 | Per-stage batch size for the x4->x8 model. **x8 operates on 160x160 patches through a 256-channel CNN, so `batch_size`=512 needs ~50 GB and OOMs typical GPUs — hence a smaller default.** Falls back to `batch_size` only if set to null. |
| `max_itr_x2` | int | 5000 | Maximum training epochs for the x1->x2 model (fine-tune default: 300). |
| `max_itr_x4` | int | 1000 | Maximum training epochs for the x2->x4 model (fine-tune default: 300). |
| `max_itr_x8` | int | 1000 | Maximum training epochs for the x4->x8 model (fine-tune default: 300). |
| `train_frac` | float | 0.8 | Fraction of patches used for training (rest for validation). |
| `ec_val` | float | 0.02 | Error convergence value. Training checks if loss drops below this. |
| `ec_itr` | int | 10 | Number of initial epochs to check for convergence before continuing. |
| `maxModInit` | int | 3 | Max model re-initialisations in the convergence gate before proceeding with current weights (prevents infinite loops on quick/hard configs). |
| `n_workers` | int | 1 | Number of DataLoader workers. Prefetch is bounded internally (was an OOM source at large `batch_size`). |
| `pred_batch_size` | int | 500 | Batch size for creating predicted patchstores between training stages. |

---

## Output Directory Structure

After a successful run, your output directory will contain:

```
output_dir/
|-- sr_config.json                    <-- Use this with sr-midas-process
|-- auto_train.log                    <-- Full pipeline log
|-- auto_train_config_resolved.json   <-- Config with all defaults filled in
|
|-- peakbank/
|   +-- peakbank.csv                  <-- Extracted and filtered MIDAS peaks
|
|-- patchstore/
|   +-- patchstore.h5                 <-- Synthetic training data (all SR levels)
|
|-- models/
|   |-- x1_x2-itrOut/                <-- Trained x1->x2 model
|   |   |-- _train_args.json
|   |   |-- _train_log.log
|   |   +-- mod-it{N}.pth            <-- Model checkpoints (one per epoch)
|   |
|   |-- x2pred_x4-itrOut/            <-- Trained x2->x4 model
|   |   +-- ...
|   |
|   +-- x4pred_x8-itrOut/            <-- Trained x4->x8 model
|       +-- ...
|
+-- pred_patchstores/
    |-- x2pred.h5                     <-- x2 model predictions (input for x4 training)
    +-- x4pred.h5                     <-- x4 model predictions (input for x8 training)
```

### Key files

- **`sr_config.json`**: The main output. A drop-in configuration file for
  `sr-midas-process` that points to your trained models with absolute paths.

- **`auto_train.log`**: Complete log with timestamps for every step. Useful for
  debugging or tracking training time.

- **`auto_train_config_resolved.json`**: The full configuration with all
  defaults filled in. Useful for reproducibility.

---

## How It Works: The Cascaded Training Pipeline

The tool trains three separate CNN models in cascade:

### Stage 1: x1 -> x2 (native to 2x resolution)

- **Input**: SRx1 patches from the patchstore (native resolution)
- **Target**: SRx2 patches from the patchstore (2x ground truth)
- **Training**: The model learns to predict what a 2x super-resolved patch
  looks like given the native-resolution input.

### Stage 2: x2 -> x4 (2x to 4x resolution)

Before training the x4 model, the tool runs the trained x2 model on all SRx1
patches to create a "predicted patchstore" (`x2pred.h5`). This is crucial:

- **Input**: x2-predicted patches (from the x2 model, NOT ground truth x2)
- **Target**: SRx4 patches from the patchstore (4x ground truth)
- **Why predicted input?** During inference, the x4 model receives output from
  the x2 model, which has imperfections. By training on predicted (imperfect)
  input, the x4 model learns to handle these artifacts.

### Stage 3: x4 -> x8 (4x to 8x resolution)

Same pattern: the x4 model's predictions become `x4pred.h5`, and the x8 model
trains on those predicted inputs against SRx8 ground truth.

---

## Resumability

The workflow is **resumable**. If it gets interrupted (e.g., job timeout, crash),
simply re-run the same command:

```bash
sr-midas-auto-train -config my_train_config.json
```

Each step checks whether its output already exists:
- If `peakbank.csv` exists -> peakbank creation is skipped
- If `patchstore.h5` exists -> patchstore creation is skipped
- If `x1_x2-itrOut/` has `.pth` files -> x2 training is skipped
- And so on for each step

This means you can also re-run after adjusting config parameters for later
steps without redoing earlier ones.

---

## Using Trained Models with sr-midas-process

### Method 1: Using the generated sr_config.json (recommended)

```bash
sr-midas-process -midasZarrDir /path/to/midas_data -SRconfig /path/to/output/sr_config.json
```

The generated `sr_config.json` contains absolute paths to your trained model
directories. It overrides only the model paths; all other SR processing
parameters (batch size, peak finding thresholds, etc.) use the bundled defaults.

### Method 2: Manual config with custom model paths

If you want to customize other SR processing parameters alongside your custom
models, create your own JSON config:

```json
{
    "mods_to_use": {
        "SRx2": {
            "mod_dir": "/absolute/path/to/output/models/x1_x2-itrOut",
            "mod_itr": 4975
        },
        "SRx4": {
            "mod_dir": "/absolute/path/to/output/models/x2pred_x4-itrOut",
            "mod_itr": 999
        },
        "SRx8": {
            "mod_dir": "/absolute/path/to/output/models/x4pred_x8-itrOut",
            "mod_itr": 999
        }
    },
    "batch_size": 400
}
```

Find the correct `mod_itr` values in the generated `sr_config.json` or by
checking the `mod-it*.pth` files in each model directory (the highest number
is used).

---

## Example Configs

### Minimal config (uses all defaults)

```json
{
    "midas_dir": ["/data/experiment_2024/sample_A"],
    "output_dir": "/results/custom_models/sample_A"
}
```

### Fine-tune the bundled models on a new dataset (recommended for most users)

```json
{
    "mode": "finetune",
    "midas_dir": ["/data/experiment_2024/sample_A"],
    "output_dir": "/results/finetuned/sample_A",
    "peak_source": "auto"
}
```

This warm-starts each cascade stage from the bundled pretrained models and
adapts them to `sample_A`'s peaks with a few hundred epochs. Much faster than
training from scratch and usually the best choice when your data is broadly
similar to standard FF-HEDM diffraction.

### Fine-tune from a *previous* run (chained refinement)

```json
{
    "mode": "finetune",
    "midas_dir": ["/data/experiment_2024/sample_B"],
    "output_dir": "/results/finetuned/sample_B",
    "base_models": "/results/finetuned/sample_A/sr_config.json"
}
```

### Multiple MIDAS directories (more training data)

```json
{
    "midas_dir": [
        "/data/experiment_2024/sample_A",
        "/data/experiment_2024/sample_B",
        "/data/experiment_2024/sample_C"
    ],
    "output_dir": "/results/custom_models/multi_sample"
}
```

### Faster training (fewer patches, fewer epochs)

```json
{
    "midas_dir": ["/data/experiment_2024/sample_A"],
    "output_dir": "/results/quick_test",
    "n_patches": 10000,
    "max_itr_x2": 1000,
    "max_itr_x4": 500,
    "max_itr_x8": 500
}
```

### Higher quality training (more patches, stricter filtering)

```json
{
    "midas_dir": ["/data/experiment_2024/sample_A"],
    "output_dir": "/results/high_quality",
    "n_patches": 100000,
    "peak_recon_err_threshold": 0.15,
    "err_cut": 0.15,
    "max_itr_x2": 10000,
    "max_itr_x4": 3000,
    "max_itr_x8": 3000
}
```

### Custom architecture (smaller/faster model)

```json
{
    "midas_dir": ["/data/experiment_2024/sample_A"],
    "output_dir": "/results/small_model",
    "arch": "64-5-r_32-5-r_16-5-r_1-5-s"
}
```

The architecture string format is `channels-kernel_size-activation` per layer,
separated by `_`. The last layer must output 1 channel with sigmoid activation
(`1-K-s`).

---

## Troubleshooting

### "No .MIDAS.zip file found"

The directory in `midas_dir` must contain a file ending in `.MIDAS.zip`.
This is the zarr archive produced by MIDAS processing.

### "No peaks found" or empty peakbank

- Check that `Temp/` folder exists in your MIDAS directory with `_PS.csv` files
- Try lowering `I_thresh` if your data has weak peaks
- Try increasing `peak_recon_err_threshold` if too many peaks are filtered out

### Out of GPU memory during training

- Reduce `batch_size` (e.g., 256 or 128)
- Reduce `pred_batch_size` (e.g., 200)
- Use a smaller architecture via the `arch` parameter

### Training takes too long

- Reduce `n_patches` (e.g., 10000-20000 for quick experiments)
- Reduce `max_itr_x2`, `max_itr_x4`, `max_itr_x8`
- Use a smaller architecture

### Pipeline interrupted

Just re-run the same command. Completed steps are automatically skipped.

### Poor SR results with custom models

- Ensure your MIDAS data has sufficient peaks for training (check the peakbank
  CSV row count)
- Try using multiple MIDAS directories to increase training data diversity
- Compare training logs (`_train_log.log`) — final validation loss should be
  below `ec_val` (default 0.02)
- Try increasing `n_patches` or `max_itr_*` for longer training
