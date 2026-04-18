#!/usr/bin/env bash
set -e

python src/classification/train_classifier.py \
  --train_jsonl "$TRAIN_JSONL" \
  --val_jsonl "$VAL_JSONL" \
  --image_dir "$IMAGE_DIR" \
  --text_encoder_path "$BERT_PATH" \
  --save_dir "$SAVE_DIR"
