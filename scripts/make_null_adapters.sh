#!/bin/bash
# Builds the zero-initialised "null" LoRA adapters that the VQA service attaches at
# startup (vqa/artifacts/*_null_adapter). Only their adapter_config.json files are
# tracked in git; this script regenerates the weights from those configs.
#
#   BASE_MODEL=Qwen/Qwen3-VL-8B-Instruct DEVICE=cpu bash scripts/make_null_adapters.sh
set -e
cd "$(dirname "$0")/.."

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
DEVICE="${DEVICE:-cpu}"

for name in sar_bbox_null_adapter normal_caption_null_adapter sar_caption_null_adapter; do
    python finetune/scripts/create_null_adapter.py \
        --base-model "$BASE_MODEL" \
        --template-adapter "vqa/artifacts/$name" \
        --output "vqa/artifacts/$name" \
        --dtype bf16 --device "$DEVICE"
done
