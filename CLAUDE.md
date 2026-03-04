# GeoGuessr Model

Region-level geographic prediction from images, built on GeoCLIP (NeurIPS 2023).

## Architecture
- **Image encoder**: Frozen CLIP ViT-L/14 → MLP (768→512) from GeoCLIP
- **Classification head**: LayerNorm → Linear(512→512) → GELU → Linear(512→num_regions)
- **Region labels**: KMeans clustering on GPS coordinates (default 128 clusters)
- Loss: CrossEntropy on region IDs (classification mode) or InfoNCE contrastive loss

## Quick Start
```bash
pip install -r requirements.txt
python data/download.py                          # Download dataset
python train.py --config configs/default.yaml    # Train
python evaluate.py --checkpoint checkpoints/best.pt  # Evaluate
python predict.py --image test.jpg               # Predict
```

## Project Structure
- `configs/default.yaml` — Training hyperparams, data paths
- `data/download.py` — Download marcelomoreno26/geoguessr from HuggingFace
- `data/dataset.py` — PyTorch Dataset, RegionClusterer, train/val/test splits
- `models/geoclip_finetune.py` — GeoGuessrModel wrapping GeoCLIP with classification head
- `train.py` — Training loop with validation, early stopping, checkpointing
- `evaluate.py` — Distance metrics (25km/200km/750km accuracy, GeoScore)
- `predict.py` — Single-image inference

## Key Metrics
- **acc_25km**: City-level accuracy (within 25km)
- **acc_200km**: Region-level accuracy (within 200km)
- **acc_750km**: Country-level accuracy (within 750km)
- **GeoScore**: Exponential decay score (5000 max, like GeoGuessr game)

## Data
- Primary: `marcelomoreno26/geoguessr` (~36k images, 55 countries)
- Optional: OSV-5M subset for scaling

## Config
Edit `configs/default.yaml` for hyperparams. Key settings:
- `data.num_regions`: Number of KMeans clusters (default 128)
- `model.freeze_clip`: Keep CLIP backbone frozen (default true)
- `training.learning_rate`: Default 1e-4
- `training.patience`: Early stopping patience (default 7)
