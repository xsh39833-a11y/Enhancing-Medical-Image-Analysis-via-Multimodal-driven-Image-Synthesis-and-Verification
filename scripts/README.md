# Scripts

This directory contains lightweight shell entry points for the main stages of the released pipeline. Each script forwards a set of environment variables to the corresponding Python module in `src/`.

## Available Scripts

| Script | Stage | Backing Python entry point | Purpose |
|---|---|---|---|
| `run_generation.sh` | Generation | `src/generation/generate_synthetic_images.py` | Generate multimodal synthetic images using SDXL, LoRA, and IP-Adapter |
| `filter_classifier_data.sh` | Filtering | `src/classification/filter_classifier_data.py` | Filter synthetic SkinCon samples with the diagnosis verifier |
| `filter_segmentation_data.sh` | Filtering | `src/segmentation/sanet/filter_synthetic_segmentation_data.py` | Filter synthetic polyp samples with segmentation verifiers |
| `train_classifier.sh` | Training | `src/classification/train_classifier.py` | Train the multimodal diagnosis model |
| `train_sanet.sh` | Training | `src/segmentation/sanet/train.py` | Train SANet for the segmentation experiments |
| `train_polyp_pvt.sh` | Training | `src/segmentation/polyp_pvt/train.py` | Train Polyp-PVT for the segmentation experiments |
| `evaluate_segmentation.sh` | Evaluation | `src/evaluation/evaluate_segmentation.py` | Evaluate segmentation predictions and report metrics |

## How to use these scripts

All scripts are intentionally minimal wrappers. Before running a script, export the required environment variables in your shell or define them inline.

Example:

```bash
export SDXL_PATH=/path/to/sdxl
export LORA_PATH=/path/to/lora
export IP_ADAPTER_PATH=/path/to/ip-adapter
export IMAGE_ENCODER_PATH=/path/to/image-encoder
export INPUT_JSONL=/path/to/train.jsonl
export CONDITION_ROOT=/path/to/conditions
export OUTPUT_DIR=/path/to/output
bash scripts/run_generation.sh
```

## Required variables by script

### `run_generation.sh`

- `SDXL_PATH`
- `LORA_PATH`
- `IP_ADAPTER_PATH`
- `IMAGE_ENCODER_PATH`
- `INPUT_JSONL`
- `CONDITION_ROOT`
- `OUTPUT_DIR`

### `filter_classifier_data.sh`

- `MODEL_PATH`
- `SYN_JSONL`
- `SYN_IMAGE_DIR`
- `TRAIN_JSONL`
- `BERT_PATH`
- `OUTPUT_DIR`

### `filter_segmentation_data.sh`

- `SYN_JSONL`
- `REAL_JSONL`
- `SYN_IMAGE_DIR`
- `REAL_IMAGE_DIR`
- `GT_MASK_DIR`
- `SNAPSHOT`
- `OUTPUT_DIR`

### `train_classifier.sh`

- `TRAIN_JSONL`
- `VAL_JSONL`
- `IMAGE_DIR`
- `BERT_PATH`
- `SAVE_DIR`

### `train_sanet.sh`

- `TRAIN_JSONL`
- `TRAIN_IMAGE_ROOT`
- `TRAIN_MASK_ROOT`
- `SAVE_DIR`
- `SNAPSHOT`

### `train_polyp_pvt.sh`

- `TRAIN_JSONL`
- `TRAIN_IMAGE_ROOT`
- `TRAIN_MASK_ROOT`
- `TEST_ROOT`
- `PVT_BACKBONE`
- `SAVE_DIR`

### `evaluate_segmentation.sh`

- `CONFIG_PATH`

## Notes

- These scripts do not hard-code local paths; you should provide dataset paths, checkpoint paths, and output paths in your own environment.
- Released checkpoints can be downloaded from Hugging Face and organized under `checkpoints/` as described in `../checkpoints/README.md`.
- If you prefer, you can call the Python entry points directly instead of using the shell wrappers.
