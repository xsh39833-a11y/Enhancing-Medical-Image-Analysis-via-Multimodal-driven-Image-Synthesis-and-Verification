#!/usr/bin/env bash
set -e

python src/segmentation/polyp_pvt/train.py \
  --train_jsonl "$TRAIN_JSONL" \
  --train_image_root "$TRAIN_IMAGE_ROOT" \
  --train_mask_root "$TRAIN_MASK_ROOT" \
  --test_path "$TEST_ROOT" \
  --pretrained_backbone "$PVT_BACKBONE" \
  --train_save "$SAVE_DIR"
