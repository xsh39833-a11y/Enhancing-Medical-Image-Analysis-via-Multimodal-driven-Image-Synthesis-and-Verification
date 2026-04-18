#!/usr/bin/env bash
set -e

python src/segmentation/sanet/train.py \
  --train_jsonl "$TRAIN_JSONL" \
  --image_root "$TRAIN_IMAGE_ROOT" \
  --mask_root "$TRAIN_MASK_ROOT" \
  --savepath "$SAVE_DIR" \
  --snapshot "$SNAPSHOT"
