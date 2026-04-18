#!/usr/bin/env bash
set -e

python src/classification/filter_classifier_data.py \
  --model_path "$MODEL_PATH" \
  --synthetic_jsonl "$SYN_JSONL" \
  --synthetic_image_dir "$SYN_IMAGE_DIR" \
  --original_train_jsonl "$TRAIN_JSONL" \
  --text_encoder_path "$BERT_PATH" \
  --output_dir "$OUTPUT_DIR"
