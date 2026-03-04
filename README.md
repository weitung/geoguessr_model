# GeoGuessr Model

Country-level geographic prediction from street-view images, built on [GeoCLIP](https://arxiv.org/abs/2309.16020) (NeurIPS 2023).

Given a single image, the model predicts which of 55 countries it was taken in.

## Architecture

The model finetunes [GeoCLIP](https://github.com/VicenteVivan/geo-clip)'s pretrained image encoder with a country classification head:

1. **CLIP ViT-L/14** (frozen) — extracts 768-dim image features
2. **GeoCLIP MLP** (finetuned) — projects to 512-dim geo-aware embeddings
3. **Classification head** (trained) — LayerNorm → Linear(512, 512) → GELU → Linear(512, 55)

Only the MLP and classification head are trained (~1.3M parameters). The CLIP backbone stays frozen.

## Training Results

Best checkpoint at epoch 11 (early stopping with patience 7):

| Metric | Value |
|---|---|
| Country accuracy | **80.3%** |
| Accuracy < 750 km | 84.8% |
| Accuracy < 2500 km | 91.5% |
| Mean distance | 849 km |
| Median distance | 0 km |
| Validation loss | 0.743 |

Trained with AdamW (lr=1e-4, weight_decay=1e-4), cosine schedule with 2 epoch warmup, batch size 32.

## Dataset

[marcelomoreno26/geoguessr](https://huggingface.co/datasets/marcelomoreno26/geoguessr) — ~36k street-view images across 55 countries.

| Split | Samples |
|---|---|
| Train | 25,160 |
| Validation | 5,372 |
| Test | 5,445 |

## Quick Start

```bash
pip install -r requirements.txt

# Download dataset
python data/download.py

# Train
python train.py --config configs/default.yaml

# Evaluate
python evaluate.py --checkpoint checkpoints/best.pt

# Predict on a single image
python predict.py --image test.jpg
```

## Project Structure

```
configs/default.yaml          — Training hyperparameters
data/download.py              — Download dataset from HuggingFace
data/dataset.py               — PyTorch Dataset, CountryMapper, train/val/test splits
models/geoclip_finetune.py    — GeoGuessrModel (GeoCLIP + classification head)
train.py                      — Training loop with early stopping
evaluate.py                   — Distance-based evaluation metrics
predict.py                    — Single-image inference
notebooks/explore_data.ipynb  — Data exploration notebook
```

## Pretrained Model

The pretrained base is [GeoCLIP](https://github.com/VicenteVivan/geo-clip), which pairs CLIP ViT-L/14 with a GPS-aware location encoder trained on 1M geotagged images. We use only the image encoder side — the CLIP backbone plus GeoCLIP's learned MLP projection — and replace the contrastive location matching with a 55-class country classifier. The GeoCLIP weights are loaded automatically via the `geoclip` pip package.
