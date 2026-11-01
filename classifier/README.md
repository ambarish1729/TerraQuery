# Modality classifier

A ResNet50 that labels each incoming image as **SAR**, **RGB** or **false-colour (NIR-R-G)**. The VQA service uses the label to choose the right LoRA adapter and to prepend a sensor-specific prompt, for example "bright = rough or metallic surface" for SAR, or "red = vegetation" for NIR composites.

```
ResNet50 (ImageNet init, all layers fine-tuned)
  └─ fc: Linear(2048→512) → BatchNorm → ReLU → Dropout(0.3) → Linear(512→3)
```

| File | Purpose |
|---|---|
| `train_band_classifier.ipynb` | Training notebook (Colab). Builds a balanced 3-class set from SpaceNet6, SSL4EO-S12 band composites and SAR tiles (MMRS SARV2, EarthMind). Includes quadrant tiling and black-pixel filtering. |
| `predict.py` | Standalone CLI. Loads the checkpoint on CPU first, then moves it to the first usable GPU. |

```bash
python classifier/predict.py --checkpoint best_model.pt --image tile.png
```

The VQA service loads the same architecture through [`vqa/image_type_classifier.py`](../vqa/image_type_classifier.py). Point `BAND_CLASSIFIER_CHECKPOINT` at your `.pt` file. Without a checkpoint, routing falls back to the RGB path.

> The notebook paths point to the team's Google Drive. Update `CONFIG["dataset_path"]` and the cache paths before running it.
