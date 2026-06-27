<p align="center">
  <img src="docs/assets/tremorvit-overview.svg" alt="TremorViT overview" width="560"/>
</p>

<h1 align="center">TremorViT</h1>

<p align="center">
  <em>Single-station vision-transformer hypocenter localization for tectonic tremor.</em>
</p>

<p align="center">
  <b>Hi-net SAC</b> -> <b>CNN tremor detector</b> -> <b>ViT locator</b> -> <b>fused hypocenter catalog</b>
</p>

---

## Overview

TremorViT detects tectonic tremor from continuous Hi-net SAC waveforms and
estimates hypocenters with pretrained deep learning models.

The pipeline has three stages:

| Stage | Role |
| --- | --- |
| **Tremor detection** | Three-component SAC waveforms are filtered, split into one-minute windows, converted to spectrograms, and classified by the CNN detector. |
| **Single-station localization** | Each detected station-minute is converted into an `EW/NS/UD` waveform tensor and passed to a ViT locator that predicts relative east, north, and depth offsets. |
| **Hypocenter fusion** | Station-wise predictions are grouped with k-core logic and fused with covariance intersection into final hypocenter estimates. |

## Repository Layout

<details open>
<summary>Project tree</summary>

```text
TremorViT/
|-- config/
|   |-- project_config.json              # Waveform, spectrogram, and dataset settings
|   |-- detector_configs.json            # Detector model path
|   `-- experiments/
|       `-- vit_locator_v11.json         # ViT locator architecture and components
|-- data/
|   `-- version1.0/station/
|       `-- hinet_used.txt               # Default Hi-net station metadata
|-- docs/
|   `-- assets/
|       `-- tremorvit-overview.svg       # Project logo and overview image
|-- models/
|   |-- tremor_detector_v1.keras         # Pretrained CNN detector
|   `-- version1.0/vit_locator_v11/
|       `-- best_weight.pth              # Pretrained ViT locator
|-- reports/
|   `-- sample/                          # Example fused hypocenter catalog
|-- scripts/
|   |-- predict_hypocenter_turbo.py      # Main inference pipeline
|   |-- hypocenter_determination.py      # Prediction fusion
|   |-- make_waveform_dataset.py         # CSV/HDF5 dataset builder
|   |-- finetune_locator.py              # Locator fine-tuning
|   `-- evaluate.py                      # Locator evaluation
|-- src/my_module/
|   |-- sac/                             # SAC reading, trimming, caching, filtering
|   |-- process/                         # Waveform and spectrogram processing
|   |-- models/                          # CNN detector and ViT locator modules
|   |-- prediction/                      # K-core and covariance-intersection logic
|   `-- train/                           # Dataset and training utilities
|-- pyproject.toml
|-- uv.lock
`-- README.md
```

</details>

## SAC Input Layout

Waveform SAC files are not included because they are large and
machine-specific. `SacHandler` reads hourly files named:

```text
{station}.{component}.SAC
```

where `{component}` is configured by `sac.component_channels`. For example,
the model-facing keys can map to SAC channel suffixes like this:

| Model-facing key | SAC channel suffix |
| --- | --- |
| `EW` | `E` channel |
| `NS` | `N` channel |
| `UD` | `U` channel |

Both `sac.component_channels` and `sac.year_to_path` must be set in
`config/project_config.json`. Each `sac.year_to_path` value must point to a year
directory arranged like this:

```text
YEAR_SAC_DIR/
`-- {YYYYMMDDHH}/
    |-- N.URSH.E.SAC
    |-- N.URSH.N.SAC
    `-- N.URSH.U.SAC
```

Each value should point to the directory that contains hourly folders for that
year:

```json
{
  "component_channels": {
    "EW": "E",
    "NS": "N",
    "UD": "U"
  },
  "year_to_path": {
    "2016": "/path/to/sac/2016"
  }
}
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management. Python 3.12 is recommended because TensorFlow 2.17 does
not provide Python 3.13 wheels.

```bash
uv sync
```

Runtime dependencies are declared in `pyproject.toml` and locked in `uv.lock`.
When dependencies change, update both files through uv:

```bash
uv add PACKAGE
uv remove PACKAGE
uv lock
uv sync
```

## Running Inference

Run the full detection and hypocenter-localization pipeline from the repository
root:

```bash
uv run python scripts/predict_hypocenter_turbo.py \
  --start_time 2016-01-01-00:00:00.000 \
  --end_time 2016-01-31-23:59:59.000 \
  --n_producers 8 \
  --device auto
```

| Argument | Description |
| --- | --- |
| `--start_time`, `--end_time` | Processing range in `YYYY-mm-dd-HH:MM:SS.ffffff` format. |
| `--station_file` | Station metadata file. The default is `data/version1.0/station/hinet_used.txt`. |
| `--output_dir` | Per-station prediction directory. The default is `reports/version1.0/hypocenter/org`. |
| `--device` | `auto`, `cpu`, `cuda`, or `cuda:<index>`. |
| `--n_producers` | Number of parallel waveform producer threads. |

Per-station predictions are written to:

```text
reports/version1.0/hypocenter/org/{start_time}_{end_time}.csv
```

Then fuse the station-wise predictions:

```bash
uv run python scripts/hypocenter_determination.py \
  --start 2016-01-01-00:00:00.000 \
  --end 2016-01-31-23:59:59.000 \
  --org_dir reports/version1.0/hypocenter/org \
  --out_dir reports/version1.0/hypocenter
```

Fusion outputs are written as:

```text
reports/version1.0/hypocenter/fused_{start}_{end}.csv
reports/version1.0/hypocenter/fused_isolated_removed_{start}_{end}.csv
```

## Building Locator Datasets

`scripts/make_waveform_dataset.py` creates the CSV/HDF5 layout used by
fine-tuning and evaluation. Prepare split catalogs like this:

<details open>
<summary>Input catalog layout</summary>

```text
data/version1.0/catalog/
|-- tremor_catalog_train.csv
|-- tremor_catalog_val.csv
`-- tremor_catalog_test.csv
```

</details>

Each input catalog must contain:

| Column | Description |
| --- | --- |
| `start_time_for_trainlocator` | Waveform window start time. |
| `lat`, `lon`, `dep` | Event latitude, longitude, and depth. |
| `station` | Station code used to read SAC waveforms. |

Build datasets:

```bash
uv run python scripts/make_waveform_dataset.py \
  --catalog_dir data/version1.0/catalog \
  --dataset_dir dataset_ssd/version1.0 \
  --station_file data/version1.0/station/hinet_used.txt \
  --splits train val test
```

Output layout:

<details open>
<summary>Dataset output layout</summary>

```text
dataset_ssd/version1.0/
|-- train.csv
|-- val.csv
|-- test.csv
`-- hdf/
    |-- train.h5
    |-- val.h5
    `-- test.h5
```

</details>

Each HDF5 file contains:

| Dataset | Description |
| --- | --- |
| `waveforms` | `(N, 3, win_sec * fs + 2 * jitter_sec * fs)`. |
| `east_km`, `north_km`, `depth_km` | Station-relative target offsets. |
| `sta_lat`, `sta_lon` | Station coordinates. |

## Fine-Tuning

Fine-tune all layers from the included pretrained locator:

```bash
uv run python scripts/finetune_locator.py \
  --exp vit_locator_v11 \
  --dataset_dir dataset_ssd/version1.0 \
  --pretrained_weight models/version1.0/vit_locator_v11/best_weight.pth \
  --save_dir models/version1.0/vit_locator_v11_finetuned \
  --epochs 50 \
  --learning_rate 1e-5 \
  --batch_size 32 \
  --freeze all
```

For small datasets, start by training only the prediction head:

```bash
uv run python scripts/finetune_locator.py \
  --dataset_dir dataset_ssd/version1.0 \
  --save_dir models/version1.0/vit_locator_v11_head_finetuned \
  --epochs 30 \
  --learning_rate 5e-5 \
  --freeze head
```

Fine-tuning outputs are written to `--save_dir`:

```text
best_weight.pth
finetune_history.csv
finetune.log
```

## Evaluation

Evaluate a locator checkpoint on a split:

```bash
uv run python scripts/evaluate.py \
  --exp vit_locator_v11 \
  --dataset_dir dataset_ssd/version1.0 \
  --weight models/version1.0/vit_locator_v11_finetuned/best_weight.pth \
  --target test \
  --output_dir reports/version1.0/evaluate
```

For a quick smoke test, add `--max_samples 100`.

Evaluation outputs:

```text
reports/version1.0/evaluate/locator_pred_{exp}_{target}.csv
reports/version1.0/evaluate/locator_metrics_{exp}_{target}.json
```

## Output Files

The fused hypocenter catalog contains:

| Column | Description |
| --- | --- |
| `origin_time` | Estimated tremor origin time. |
| `lat`, `lon`, `dep` | Fused hypocenter latitude, longitude, and depth. |
| `stations` | Semicolon-separated station codes used for the fused source. |
| `n_in_comp` | Number of station predictions in the selected k-core component. |
| `sigma11` ... `sigma33` | 3 x 3 fused covariance matrix in local east, north, and depth coordinates. |
| `volume_km3` | 95% confidence ellipsoid volume. |
| `major_length_km` | Full length of the 95% confidence ellipsoid major axis. |
| `source_id` | Zero-based source index when multiple sources are retained for one origin time. |

A ready-made sample is included:

```text
reports/sample/fused_isolated_removed_2016-01-01-00:00:00.000_2016-09-30-23:59:59.000.csv
```

Use it to inspect the final output format without running the full SAC waveform
pipeline.

## Models

### CNN Tremor Detector

`models/tremor_detector_v1.keras` classifies normalized three-component
spectrograms and provides tremor probabilities for one-minute station windows.

### ViT Hypocenter Locator

`models/version1.0/vit_locator_v11/best_weight.pth` predicts a hypocenter
offset from a single station waveform. The default input components are defined
in `config/experiments/vit_locator_v11.json`:

```json
["EW", "NS", "UD"]
```

## Reference

If you use this code, model weights, or sample catalog in a publication, please
cite the TremorViT paper.

Sugii, A., and Hiramatsu, Y. Single station seismic observations
enable high resolution localization of tectonic tremor sources using a vision
transformer. *Scientific Reports* (2026).
[https://doi.org/10.1038/s41598-026-58641-5](https://doi.org/10.1038/s41598-026-58641-5)

```bibtex
@article{tremorvit,
  title   = {Single station seismic observations enable high resolution localization of tectonic tremor sources using a vision transformer},
  author  = {Author, Amane Sugii, Yoshihiro Hiramatsu},
  journal = {Scientific Reports},
  year    = {2026},
  doi     = {https://doi.org/10.1038/s41598-026-58641-5}
}
```
