# Fine-tuning

LoRA training and evaluation scripts for the three Qwen3-VL-8B domain adapters used by TerraQuery. Training runs on [Unsloth](https://github.com/unslothai/unsloth) with the base model frozen.

| Adapter | Script | Data | LoRA |
|---|---|---|---|
| SAR grounding (`sar_bbox`) | `finetune_qwen_bbox.py` | MMRS SARV2 ship detection (HBB) | r=16, α=32 |
| Optical captioning (`normal_caption`) | `finetune_qwen_caption.py` | VRSBench / SpaceNet6 RGB captions | r=8, α=16 |
| SAR captioning (`sar_caption`) | `finetune_qwen_caption.py` | Paired RGB+SAR captions (EarthMind, DFC23/25, SpaceNet6, M4-SIR) | r=8, α=16 |

## Setup

```bash
conda activate geonli          # created by scripts/install.sh
pip install -r finetune/requirements.txt
```

## Train

**SAR grounding.** The grounding script adds a composite loss on top of LM cross-entropy: IoU loss on predicted boxes, plus false-positive and false-negative penalties for unmatched boxes.

```bash
python finetune/finetune_qwen_bbox.py \
  --train-json path/to/MMRS/json/detection/SARV2/SARv2_trainnval_hbb.json \
  --img-root   path/to/MMRS/detection/SARV2/images \
  --output-dir finetune/checkpoints/sar_bbox \
  --batch-size 4 --epochs 3 --lora-r 16 --lora-alpha 32 \
  --grounding-loss-every-n-steps 5
```

**Captioning (optical or SAR).**

```bash
python finetune/finetune_qwen_caption.py \
  --use-lora \
  --train-json path/to/captions.json \
  --image-dir  path/to/images \
  --output-dir finetune/checkpoints/normal_caption
```

## Evaluate

| Script | What it measures |
|---|---|
| `evaluate.py` | Draw grounded boxes for an object class with a base or LoRA checkpoint |
| `bbox_eval.py` | Fine-tuned vs. base captions scored with BERT-BLEU (batched, PEFT adapter or full checkpoint) |
| `bbscore_all.py` | Fine-tuned vs. base captions with BERT-BLEU, single-image or batch mode, with box overlays |
| `bert_bleu.py` | Reference implementation of the BERT-BLEU metric |
| `infer.py` | Caption a single image with a base, LoRA, or merged checkpoint |

```bash
python finetune/infer.py --checkpoint finetune/checkpoints/normal_caption/final --image sample.png
```

## Deploying a trained adapter

Copy (or symlink) each adapter folder into `vqa/checkpoints/` with the names `normal_caption`, `sar_caption`, and `sar_bbox`. The VQA service picks them up automatically, or you can point `QWEN_ADAPTER_ROOT` elsewhere.

`scripts/create_null_adapter.py` builds a zero-initialised adapter that mirrors a trained one's config. The VQA service swaps back to it after each request, so the base model is restored without reloading weights.

## Data sources

- **MMRS SARV2**: see the [EarthGPT](https://github.com/wivizhang/EarthGPT) project for download and licensing.
- **EarthMind / EarthMind-Bench**: [huggingface.co/datasets/sy1998/EarthMind-data](https://huggingface.co/datasets/sy1998/EarthMind-data)
- **VRSBench**: [github.com/lx709/VRSBench](https://github.com/lx709/VRSBench)
