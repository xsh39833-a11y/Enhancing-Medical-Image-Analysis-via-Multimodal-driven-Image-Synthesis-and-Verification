#!/usr/bin/env bash
set -e

python src/segmentation/sanet/filter_synthetic_segmentation_data.py \
  --synthetic_jsonl "$SYN_JSONL" \
  --real_jsonl "$REAL_JSONL" \
  --synthetic_image_dir "$SYN_IMAGE_DIR" \
  --real_image_dir "$REAL_IMAGE_DIR" \
  --gt_mask_dir "$GT_MASK_DIR" \
  --snapshot "$SNAPSHOT" \
  --output_dir "$OUTPUT_DIR"
