#!/usr/bin/env bash
set -e

python src/evaluation/evaluate_segmentation.py --config "$CONFIG_PATH" --verbose
