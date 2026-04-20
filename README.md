# Enhancing Medical Image Analysis via Multimodal-driven Image Synthesis and Verification

Official PyTorch implementation for multimodal-driven medical image synthesis with task-aware verification for diagnosis and segmentation.

**Authors**  
Xinran Niu<sup>1</sup>, Shihang Xia<sup>1</sup>, Dixin Luo<sup>1</sup>

**Affiliation**  
<sup>1</sup>Beijing Institute of Technology, China

**Contact**  
Please open a GitHub issue for questions about the public release.

## Quick Links

- **Paper:** to be added after publication
- **arXiv:** to be added after public posting
- **Project Page:** not available
- **Weights:** `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights`
- **Model Card:** `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights`

## Overview

This repository releases the codebase used for our multimodal-driven medical image synthesis and verification framework. The project studies how synthetic medical images, generated with multimodal conditioning, can be filtered with task-aware verification models and then used to improve downstream medical image analysis systems.

The current release covers two representative tasks:

- **Polyp segmentation**
- **Skin disease diagnosis / classification**

The repository includes:

- generation code based on SDXL, LoRA, and IP-Adapter
- task-aware filtering pipelines for diagnosis and segmentation
- downstream classifier training and testing
- downstream segmentation training and evaluation
- released checkpoints and Hugging Face links for the main experiments

## Highlights

- Multimodal-conditioned medical image synthesis based on SDXL, LoRA, and IP-Adapter.
- Task-aware quality control for both diagnosis and segmentation tasks.
- Released verifier checkpoints and final downstream checkpoints for reproducible evaluation.
- Clear separation of generation, filtering, downstream training, and evaluation code.

## Quick Start

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/xsh39833-a11y/Enhancing-Medical-Image-Analysis-via-Multimodal-driven-Image-Synthesis-and-Verification.git
cd multimodal-medical-image-synthesis
conda env create -f environment.yml
conda activate janusflow
```

Or, with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download released checkpoints

Download the released weights from Hugging Face:

- `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights`

Then place them under `checkpoints/` following the layout documented in `checkpoints/README.md`.

### 3. Prepare datasets

Prepare the two datasets used in this release:

- `data/polyp/` for segmentation
- `data/skincon/` for diagnosis

The expected dataset structure is provided below in the `Data Preparation` section.

### 4. Run the main pipeline

A typical end-to-end workflow is:

1. generate synthetic images
2. filter synthetic data with verifier models
3. train the downstream model
4. evaluate the final predictions

The provided shell scripts under `scripts/` are the recommended entry points for these stages.

## Reproduce Main Results

### Reproduce Table I: polyp segmentation

#### SANet + Ours

1. Generate multimodal synthetic polyp images.
2. Filter the synthetic segmentation data with the SANet verifier.
3. Train SANet with the selected synthetic data.
4. Evaluate the final segmentation outputs.

Recommended script flow:

```bash
bash scripts/run_generation.sh
bash scripts/filter_segmentation_data.sh
bash scripts/train_sanet.sh
bash scripts/evaluate_segmentation.sh
```

Released checkpoints corresponding to this setting are hosted under:

- `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/tree/main/segmentation/sanet`

#### Polyp-PVT + Ours (M)

1. Generate multimodal synthetic polyp images.
2. Filter the synthetic segmentation data.
3. Train Polyp-PVT with the selected synthetic data.
4. Evaluate the final segmentation outputs.

Recommended script flow:

```bash
bash scripts/run_generation.sh
bash scripts/filter_segmentation_data.sh
bash scripts/train_polyp_pvt.sh
bash scripts/evaluate_segmentation.sh
```

Released checkpoints corresponding to this setting are hosted under:

- `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/tree/main/segmentation/polyp_pvt`

### Reproduce Table II: SkinCon diagnosis

1. Generate multimodal SkinCon images.
2. Filter the synthetic diagnosis data with the classification verifier.
3. Train the multimodal classifier.
4. Test the final diagnosis model.

Recommended script flow:

```bash
bash scripts/run_generation.sh
bash scripts/filter_classifier_data.sh
bash scripts/train_classifier.sh
```

Released final diagnosis checkpoint:

- `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/classification/skincon/task_aware_iteration_1/multimodal_model.pth`

### Reproduce verification models used in the filtering stage

The released verifier checkpoints are:

- SkinCon diagnosis verifier: `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/verification/classification/skincon/original/multimodal_model.pth`
- SANet segmentation verifier: `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/verification/segmentation/sanet/original/model-128`
- Polyp-PVT segmentation verifier: `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/verification/segmentation/polyp_pvt/original/PolypPVT.pth`

## Repository Structure

```text
multimodal-medical-image-synthesis/
├── checkpoints/                 # Place released checkpoints here
├── outputs/                     # Generated images, logs, predictions, and results
├── scripts/                     # Example shell launchers
├── src/
│   ├── classification/
│   │   ├── train_classifier.py
│   │   ├── test_classifier.py
│   │   └── filter_classifier_data.py
│   ├── evaluation/
│   │   ├── evaluate_segmentation.py
│   │   └── uacanet_utils/
│   ├── generation/
│   │   └── generate_synthetic_images.py
│   └── segmentation/
│       ├── polyp_pvt/
│       │   ├── train.py
│       │   ├── test.py
│       │   ├── lib/
│       │   └── utils/
│       └── sanet/
│           ├── train.py
│           ├── test.py
│           ├── filter_synthetic_segmentation_data.py
│           ├── model.py
│           └── res2net.py
├── .gitignore
├── environment.yml
├── requirements.txt
└── README.md
```

## Environment

You can reproduce the environment with either Conda or pip.

### Conda

```bash
conda env create -f environment.yml
conda activate med-image-synthesis
```

### Pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

This release assumes the following local organization:

- `data/` stores the raw datasets and jsonl metadata files
- `checkpoints/` stores downloaded released weights and backbone checkpoints
- `outputs/` stores generated images, filtered subsets, logs, predictions, and final metrics

### 1. Polyp segmentation

Expected structure:

```text
data/polyp/
├── TrainDataset/
│   ├── image/
│   └── masks/
├── TestDataset/
│   ├── CVC-300/
│   ├── CVC-ClinicDB/
│   ├── CVC-ColonDB/
│   ├── ETIS-LaribPolypDB/
│   └── Kvasir/
├── _train.jsonl
└── train.jsonl
```

### 2. Skin disease classification

Expected structure:

```text
data/skincon/
├── merged_unique/
├── _train.jsonl
├── _val.jsonl
└── _test.jsonl
```

## Model Assets

Please place released checkpoints and pretrained weights under `checkpoints/`.

Examples:

```text
checkpoints/
├── generation/
│   ├── sdxl/
│   ├── image_encoder/
│   ├── adapter_polyp_base/
│   └── adapter_skincon_base/
├── classification/
│   └── multimodal_model.pth
└── segmentation/
    ├── polyp_pvt/
    │   ├── pvt_v2_b2.pth
    │   └── PolypPVT.pth
    └── sanet/
        └── model-200
```

For a complete recommended checkpoint layout, see `checkpoints/README.md`.

## Provided Scripts

The shell scripts under `scripts/` are the recommended entry points for the main stages of the pipeline.

| Stage | Script | Purpose |
|---|---|---|
| Generation | `scripts/run_generation.sh` | Generate multimodal synthetic images for the target task |
| Filtering | `scripts/filter_classifier_data.sh` | Filter synthetic SkinCon samples with the diagnosis verifier |
| Filtering | `scripts/filter_segmentation_data.sh` | Filter synthetic polyp samples with segmentation verifiers |
| Training | `scripts/train_classifier.sh` | Train the multimodal diagnosis model |
| Training | `scripts/train_sanet.sh` | Train SANet on the selected segmentation setting |
| Training | `scripts/train_polyp_pvt.sh` | Train Polyp-PVT on the selected segmentation setting |
| Evaluation | `scripts/evaluate_segmentation.sh` | Evaluate segmentation predictions and report metrics |

## Model Zoo and Hugging Face Weights

The official released checkpoints are hosted on Hugging Face:

- **Weights repository:** `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights`

This repository hosts:

- multimodal generation weights for `polyp` and `skincon`
- verification models used in the task-aware filtering stage
- final downstream classification and segmentation checkpoints

### Hugging Face repository layout

```text
medical-image-analysis-multimodal-synthesis-weights/
├── generation/
│   ├── polyp/
│   │   └── checkpoint-20000/
│   │       ├── ip_adapter.safetensors
│   │       └── pytorch_lora_weights.safetensors
│   └── skincon/
│       └── checkpoint-20000/
│           ├── ip_adapter.safetensors
│           └── pytorch_lora_weights.safetensors
├── verification/
│   ├── classification/
│   │   └── skincon/original/multimodal_model.pth
│   └── segmentation/
│       ├── sanet/original/model-128
│       └── polyp_pvt/original/PolypPVT.pth
├── classification/
│   └── skincon/task_aware_iteration_1/multimodal_model.pth
└── segmentation/
    ├── sanet/
    │   ├── 200_run1/model-200
    │   ├── 200_run2/model-200
    │   ├── 200_run3/model-200
    │   ├── 400_run1/model-200
    │   ├── 400_run2/model-200
    │   ├── 400_run3/model-200
    │   ├── 800_run1/model-200
    │   ├── 800_run2/model-200
    │   └── 800_run3/model-200
    └── polyp_pvt/
        ├── 200_run1/PolypPVT.pth
        ├── 200_run2/PolypPVT.pth
        ├── 200_run3/PolypPVT.pth
        ├── 400_run1/PolypPVT.pth
        ├── 400_run2/PolypPVT.pth
        ├── 400_run3/PolypPVT.pth
        ├── 800_run1/PolypPVT.pth
        ├── 800_run2/PolypPVT.pth
        └── 800_run3/PolypPVT.pth
```

### Formal Model Zoo

| Group | Task | Model | Setting | Paper role | Hugging Face link |
|---|---|---|---|---|---|
| Generation | Polyp synthesis | SDXL + LoRA | checkpoint-20000 | generator for the segmentation pipeline | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/tree/main/generation/polyp/checkpoint-20000` |
| Generation | Polyp synthesis | SDXL + IP-Adapter | checkpoint-20000 | generator for the segmentation pipeline | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/tree/main/generation/polyp/checkpoint-20000` |
| Generation | SkinCon synthesis | SDXL + LoRA | checkpoint-20000 | generator for the diagnosis pipeline | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/tree/main/generation/skincon/checkpoint-20000` |
| Generation | SkinCon synthesis | SDXL + IP-Adapter | checkpoint-20000 | generator for the diagnosis pipeline | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/tree/main/generation/skincon/checkpoint-20000` |
| Verification | SkinCon diagnosis | Multimodal classifier | original | verifier used in Algorithm 2 / high-confidence filtering | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/verification/classification/skincon/original/multimodal_model.pth` |
| Verification | Polyp segmentation | SANet | original | verifier used in Eq. (8)-(10) and Algorithm 1 | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/verification/segmentation/sanet/original/model-128` |
| Verification | Polyp segmentation | Polyp-PVT | original | verifier / original baseline checkpoint | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/verification/segmentation/polyp_pvt/original/PolypPVT.pth` |
| Classification | SkinCon diagnosis | Multimodal classifier | task_aware_iteration_1 | final iterative diagnosis model, corresponding to `Ours-iter` in Table II | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/classification/skincon/task_aware_iteration_1/multimodal_model.pth` |
| Segmentation | Polyp segmentation | SANet | 200_run1 | SANet + Ours, 200 synthetic images, trial 1, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/sanet/200_run1/model-200` |
| Segmentation | Polyp segmentation | SANet | 200_run2 | SANet + Ours, 200 synthetic images, trial 2, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/sanet/200_run2/model-200` |
| Segmentation | Polyp segmentation | SANet | 200_run3 | SANet + Ours, 200 synthetic images, trial 3, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/sanet/200_run3/model-200` |
| Segmentation | Polyp segmentation | SANet | 400_run1 | SANet + Ours, 400 synthetic images, trial 1, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/sanet/400_run1/model-200` |
| Segmentation | Polyp segmentation | SANet | 400_run2 | SANet + Ours, 400 synthetic images, trial 2, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/sanet/400_run2/model-200` |
| Segmentation | Polyp segmentation | SANet | 400_run3 | SANet + Ours, 400 synthetic images, trial 3, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/sanet/400_run3/model-200` |
| Segmentation | Polyp segmentation | SANet | 800_run1 | SANet + Ours, 800 synthetic images, trial 1, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/sanet/800_run1/model-200` |
| Segmentation | Polyp segmentation | SANet | 800_run2 | SANet + Ours, 800 synthetic images, trial 2, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/sanet/800_run2/model-200` |
| Segmentation | Polyp segmentation | SANet | 800_run3 | SANet + Ours, 800 synthetic images, trial 3, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/sanet/800_run3/model-200` |
| Segmentation | Polyp segmentation | Polyp-PVT | 200_run1 | Polyp-PVT + Ours, 200 synthetic images, trial 1, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/polyp_pvt/200_run1/PolypPVT.pth` |
| Segmentation | Polyp segmentation | Polyp-PVT | 200_run2 | Polyp-PVT + Ours, 200 synthetic images, trial 2, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/polyp_pvt/200_run2/PolypPVT.pth` |
| Segmentation | Polyp segmentation | Polyp-PVT | 200_run3 | Polyp-PVT + Ours, 200 synthetic images, trial 3, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/polyp_pvt/200_run3/PolypPVT.pth` |
| Segmentation | Polyp segmentation | Polyp-PVT | 400_run1 | Polyp-PVT + Ours, 400 synthetic images, trial 1, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/polyp_pvt/400_run1/PolypPVT.pth` |
| Segmentation | Polyp segmentation | Polyp-PVT | 400_run2 | Polyp-PVT + Ours, 400 synthetic images, trial 2, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/polyp_pvt/400_run2/PolypPVT.pth` |
| Segmentation | Polyp segmentation | Polyp-PVT | 400_run3 | Polyp-PVT + Ours, 400 synthetic images, trial 3, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/polyp_pvt/400_run3/PolypPVT.pth` |
| Segmentation | Polyp segmentation | Polyp-PVT | 800_run1 | Polyp-PVT + Ours, 800 synthetic images, trial 1, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/polyp_pvt/800_run1/PolypPVT.pth` |
| Segmentation | Polyp segmentation | Polyp-PVT | 800_run2 | Polyp-PVT + Ours, 800 synthetic images, trial 2, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/polyp_pvt/800_run2/PolypPVT.pth` |
| Segmentation | Polyp segmentation | Polyp-PVT | 800_run3 | Polyp-PVT + Ours, 800 synthetic images, trial 3, Table I | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/blob/main/segmentation/polyp_pvt/800_run3/PolypPVT.pth` |

### Practical download summary

| Category | Link |
|---|---|
| All released weights | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights` |
| Generation weights | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/tree/main/generation` |
| Verification weights | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/tree/main/verification` |
| Classification weights | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/tree/main/classification` |
| Segmentation weights | `https://huggingface.co/franku123/medical-image-analysis-multimodal-synthesis-weights/tree/main/segmentation` |

### Mapping released checkpoints to paper results

- **Table I, SANet + Ours**: corresponds to `segmentation/sanet/{200,400,800}_run{1,2,3}/model-200`.
- **Table I, Polyp-PVT + Ours (M)**: corresponds to `segmentation/polyp_pvt/{200,400,800}_run{1,2,3}/PolypPVT.pth`.
- **Table II, Ours-iter**: corresponds to `classification/skincon/task_aware_iteration_1/multimodal_model.pth`.
- **Algorithm 1 / Eq. (8)-(10) segmentation verifier**: corresponds to `verification/segmentation/sanet/original/model-128` and `verification/segmentation/polyp_pvt/original/PolypPVT.pth`.
- **Algorithm 2 diagnosis verifier**: corresponds to `verification/classification/skincon/original/multimodal_model.pth`.
- **Generation backbones for both tasks**: corresponds to the LoRA / IP-Adapter files under `generation/polyp/checkpoint-20000/` and `generation/skincon/checkpoint-20000/`.


## Usage

### A. Synthetic image generation

```bash
python src/generation/generate_synthetic_images.py \
  --pretrained_model_name_or_path checkpoints/generation/sdxl \
  --pretrained_lora_path checkpoints/generation/adapter_polyp_base/checkpoint-20000 \
  --ip_adapter_path checkpoints/generation/adapter_polyp_base/checkpoint-20000 \
  --image_encoder_path checkpoints/generation/image_encoder \
  --data_path data/polyp/train.jsonl \
  --condition_image_root outputs/polyp_masks \
  --save_dir outputs/polyp_synthesis/images
```

### B. Filter synthetic classification data

```bash
python src/classification/filter_classifier_data.py \
  --model_path checkpoints/classification/multimodal_model.pth \
  --synthetic_jsonl outputs/skincon_synthesis/_data.jsonl \
  --synthetic_image_dir outputs/skincon_synthesis/images \
  --original_train_jsonl data/skincon/_train.jsonl \
  --text_encoder_path /path/to/bert-base-uncased \
  --output_dir outputs/skincon_filter
```

### C. Train classification model

```bash
python src/classification/train_classifier.py \
  --train_jsonl data/skincon/_train.jsonl \
  --val_jsonl data/skincon/_val.jsonl \
  --image_dir data/skincon/merged_unique \
  --text_encoder_path /path/to/bert-base-uncased \
  --save_dir outputs/classifier_skincon
```

### D. Test classification model

```bash
python src/classification/test_classifier.py \
  --train_jsonl data/skincon/_train.jsonl \
  --test_jsonl data/skincon/_test.jsonl \
  --image_dir data/skincon/merged_unique \
  --text_encoder_path /path/to/bert-base-uncased \
  --model_path outputs/classifier_skincon/multimodal_model.pth
```

### E. Filter synthetic segmentation data

```bash
python src/segmentation/sanet/filter_synthetic_segmentation_data.py \
  --synthetic_jsonl outputs/polyp_synthesis/_data.jsonl \
  --real_jsonl data/polyp/train.jsonl \
  --synthetic_image_dir outputs/polyp_synthesis/images \
  --real_image_dir data/polyp/TrainDataset/image \
  --gt_mask_dir data/polyp/TrainDataset/masks \
  --snapshot checkpoints/segmentation/sanet/model-200 \
  --output_dir outputs/polyp_filter
```

### F. Train Polyp-PVT

```bash
python src/segmentation/polyp_pvt/train.py \
  --train_jsonl data/polyp/_train.jsonl \
  --train_image_root data/polyp/TrainDataset/image \
  --train_mask_root data/polyp/TrainDataset/masks \
  --test_path data/polyp/TestDataset \
  --pretrained_backbone checkpoints/segmentation/polyp_pvt/pvt_v2_b2.pth \
  --train_save outputs/polyp_pvt
```

### G. Train SANet

```bash
python src/segmentation/sanet/train.py \
  --train_jsonl data/polyp/_train.jsonl \
  --image_root data/polyp/TrainDataset/image \
  --mask_root data/polyp/TrainDataset/masks \
  --savepath outputs/sanet \
  --snapshot checkpoints/segmentation/sanet/model-200
```

### H. Evaluate segmentation predictions

```bash
python src/evaluation/evaluate_segmentation.py --config configs/UACANet-L.yaml --verbose
```

## Notes on Provenance

This open-source release consolidates the implementations used in our experiments into a cleaner and more reproducible repository structure. The following source files were the main implementation basis during code organization:

- synthesis: `LDM_for_medical_images/inference_multi.py`
- classification filter/train/test:
  - `LDM_for_medical_images/classifier_filter.py`
  - `LDM_for_medical_images/classifier_train.py`
  - `LDM_for_medical_images/classifier_test.py`
- segmentation filter/train/test:
  - `SANet/src/eval.py`
  - `SANet/src/train_xsh.py`
  - `SANet/src/test.py`
  - `Polyp-PVT/train_xsh.py`
  - `Polyp-PVT/Test.py`
- metric evaluation:
  - `UACANet-main/run/Eval.py`

## Reproducibility Notes

- Please update dataset paths, checkpoint paths, and output paths to match your local environment.
- Released weights are hosted on Hugging Face and should be placed under `checkpoints/` following `checkpoints/README.md`.
- The shell scripts in `scripts/` are intentionally lightweight wrappers around the Python entry points in `src/`; you may either edit the scripts or call the Python modules directly.
- Some original research code used environment-specific local paths. In this release, the main workflows have been refactored into reusable command-line interfaces for easier reproduction.
- Reproducing one table in the paper usually requires only a subset of the full checkpoints and datasets; you do not need to download every released artifact unless you want the complete pipeline.

## Citation

If you find this repository useful in your research, please cite the paper and software release.

You can also use the citation metadata provided in `CITATION.cff`.

```bibtex
@misc{niu2026enhancing,
  title={Enhancing Medical Image Analysis via Multimodal-driven Image Synthesis and Verification},
  author={Xinran Niu and Shihang Xia and Dixin Luo},
  year={2026},
  note={Code release and model weights},
  howpublished={\url{https://github.com/xsh39833-a11y/Enhancing-Medical-Image-Analysis-via-Multimodal-driven-Image-Synthesis-and-Verification}}
}
```

## License

This project is released under the MIT License. See the `LICENSE` file for details.
