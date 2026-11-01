#!/bin/bash
# Installs the Python dependencies for the VQA and SAM-3 services.
# Run inside an activated Python 3.10 environment (e.g. `conda activate geonli`).
set -e
cd "$(dirname "$0")/.."

pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
pip install -r requirements.txt
pip install unsloth==2025.11.6 unsloth_zoo==2025.11.6

# sam3/api.py imports Sam3Model / Sam3Processor; we ran transformers built from source for these
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install '.[torch]'
cd ..
rm -rf transformers
