#!/usr/bin/env bash
set -e

python src/generation/generate_synthetic_images.py \
  --pretrained_model_name_or_path "$SDXL_PATH" \
  --pretrained_lora_path "$LORA_PATH" \
  --ip_adapter_path "$IP_ADAPTER_PATH" \
  --image_encoder_path "$IMAGE_ENCODER_PATH" \
  --data_path "$INPUT_JSONL" \
  --condition_image_root "$CONDITION_ROOT" \
  --save_dir "$OUTPUT_DIR"
