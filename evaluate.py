"""Evaluation script for GeoGuessr model with distance-based metrics."""

import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import CountryMapper, create_datasets, haversine_np
from models.geoclip_finetune import GeoGuessrModel


def geoscore(distance_km: float) -> float:
    """GeoGuessr-style score: 5000 at 0km, exponential decay."""
    return 5000 * np.exp(-distance_km / 1492.7)


def evaluate_checkpoint(checkpoint_path: str, config_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config")
    if cfg is None:
        assert config_path is not None, "Config not in checkpoint, provide --config"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

    # Load mapper
    mapper_path = os.path.join(cfg["checkpoint"]["save_dir"], "country_mapper.json")
    mapper = CountryMapper.load(mapper_path)

    # Create test dataset
    data_cfg = cfg["data"]
    _, _, test_ds, _ = create_datasets(
        data_dir=data_cfg["data_dir"] + "/geoguessr",
        image_size=data_cfg["image_size"],
        augment_train=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )

    # Load model
    model = GeoGuessrModel(
        num_classes=mapper.num_classes,
        freeze_clip=cfg["model"]["freeze_clip"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")

    # Evaluate
    all_preds = []
    all_labels = []
    all_pred_gps = []
    all_true_gps = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            labels = batch["label"]

            output = model(images)
            preds = output["logits"].argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

            for pl in preds:
                all_pred_gps.append(mapper.get_centroid(int(pl)))
            for tl in labels.numpy():
                all_true_gps.append(mapper.get_centroid(int(tl)))

    pred_gps = np.array(all_pred_gps)
    true_gps = np.array(all_true_gps)
    preds = np.array(all_preds)
    labels = np.array(all_labels)

    # When prediction is correct, distance = 0. Otherwise use centroid distance.
    correct_mask = preds == labels
    distances = haversine_np(
        pred_gps[:, 0], pred_gps[:, 1], true_gps[:, 0], true_gps[:, 1]
    )
    distances[correct_mask] = 0.0  # Correct predictions are 0 distance

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Samples: {len(distances)}")
    print(f"\nCountry Classification Accuracy: {np.mean(correct_mask):.4f}")

    print(f"\nDistance Metrics (centroid-based, 0 if country correct):")
    print(f"  Mean distance:   {np.mean(distances):.1f} km")
    print(f"  Median distance: {np.median(distances):.1f} km")

    print(f"\nAccuracy at Distance Thresholds:")
    for thresh in [200, 750, 2500]:
        acc = np.mean(distances < thresh)
        print(f"  < {thresh:>5d} km: {acc:.4f}")

    # GeoScore
    scores = np.array([geoscore(d) for d in distances])
    print(f"\nGeoScore: {np.mean(scores):.1f} / 5000")

    # Per-country breakdown
    print(f"\nPer-Country Accuracy (top 15 by sample count):")
    unique, counts = np.unique(labels, return_counts=True)
    top_countries = unique[np.argsort(-counts)][:15]
    for c in top_countries:
        mask = labels == c
        c_acc = np.mean(preds[mask] == c)
        name = mapper.decode(int(c))
        print(f"  {name:20s} (n={mask.sum():>4d}): acc={c_acc:.3f}")

    return {
        "accuracy": float(np.mean(correct_mask)),
        "mean_km": float(np.mean(distances)),
        "acc_750km": float(np.mean(distances < 750)),
        "geoscore_mean": float(np.mean(scores)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GeoGuessr model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default=None, help="Config path (if not in checkpoint)")
    args = parser.parse_args()

    evaluate_checkpoint(args.checkpoint, args.config)


if __name__ == "__main__":
    main()
