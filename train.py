"""Training script for GeoGuessr country classification model."""

import argparse
import math
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import CountryMapper, create_datasets, haversine_np
from models.geoclip_finetune import GeoGuessrModel


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def create_scheduler(optimizer, cfg, steps_per_epoch: int):
    total_steps = cfg["training"]["num_epochs"] * steps_per_epoch
    warmup_steps = cfg["training"]["warmup_epochs"] * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, dataloader, mapper, device):
    """Run evaluation and return metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_pred_gps = []
    all_true_gps = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            output = model(images)
            loss = model.compute_loss(output, labels)
            total_loss += loss.item() * len(images)

            preds = output["logits"].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(images)

            # Map predictions to GPS centroids for distance metrics
            for pl in preds.cpu().numpy():
                all_pred_gps.append(mapper.get_centroid(int(pl)))
            for tl in labels.cpu().numpy():
                all_true_gps.append(mapper.get_centroid(int(tl)))

    pred_gps = np.array(all_pred_gps)
    true_gps = np.array(all_true_gps)
    distances = haversine_np(
        pred_gps[:, 0], pred_gps[:, 1], true_gps[:, 0], true_gps[:, 1]
    )

    metrics = {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "acc_750km": float(np.mean(distances < 750)),
        "acc_2500km": float(np.mean(distances < 2500)),
        "median_km": float(np.median(distances)),
        "mean_km": float(np.mean(distances)),
    }
    return metrics


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    data_cfg = cfg["data"]
    print("Loading dataset...")
    train_ds, val_ds, test_ds, mapper = create_datasets(
        data_dir=data_cfg["data_dir"] + "/geoguessr",
        image_size=data_cfg["image_size"],
        augment_train=True,
    )
    num_classes = mapper.num_classes
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    print(f"Countries: {num_classes}")

    # Save mapper
    os.makedirs(cfg["checkpoint"]["save_dir"], exist_ok=True)
    mapper.save(os.path.join(cfg["checkpoint"]["save_dir"], "country_mapper.json"))

    # DataLoaders
    train_cfg = cfg["training"]
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=train_cfg["pin_memory"],
    )

    # Model
    model_cfg = cfg["model"]
    model = GeoGuessrModel(
        num_classes=num_classes,
        freeze_clip=model_cfg["freeze_clip"],
        hidden_dim=model_cfg["hidden_dim"],
        dropout=model_cfg["dropout"],
    ).to(device)
    print(
        f"Trainable params: {model.num_trainable_params():,} / {model.num_total_params():,}"
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = create_scheduler(optimizer, cfg, len(train_loader))

    # Training loop
    best_metric = 0.0
    patience_counter = 0
    metric_name = cfg["checkpoint"]["metric"]
    log_interval = cfg["logging"]["log_interval"]

    for epoch in range(train_cfg["num_epochs"]):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for i, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            output = model(images)
            loss = model.compute_loss(output, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() * len(images)
            preds = output["logits"].argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += len(images)

            if (i + 1) % log_interval == 0:
                avg_loss = epoch_loss / epoch_total
                acc = epoch_correct / epoch_total
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  [{epoch+1}][{i+1}/{len(train_loader)}] "
                    f"loss={avg_loss:.4f} acc={acc:.3f} lr={lr:.2e}"
                )

        # Epoch summary
        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        elapsed = time.time() - t0

        # Validation
        val_metrics = evaluate(model, val_loader, mapper, device)
        print(
            f"Epoch {epoch+1}/{train_cfg['num_epochs']} ({elapsed:.0f}s) | "
            f"Train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"Val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.3f} | "
            f"750km={val_metrics['acc_750km']:.3f} "
            f"2500km={val_metrics['acc_2500km']:.3f} | "
            f"median={val_metrics['median_km']:.0f}km"
        )

        # Checkpoint best model
        current_metric = val_metrics.get(metric_name, val_metrics["accuracy"])
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            if cfg["checkpoint"]["save_best"]:
                save_path = os.path.join(cfg["checkpoint"]["save_dir"], "best.pt")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                        "config": cfg,
                        "num_classes": num_classes,
                    },
                    save_path,
                )
                print(f"  Saved best model ({metric_name}={best_metric:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg["patience"]:
                print(f"Early stopping after {epoch+1} epochs")
                break

    # Final evaluation on test set
    print("\n--- Test Set Evaluation ---")
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    best_path = os.path.join(cfg["checkpoint"]["save_dir"], "best.pt")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")

    test_metrics = evaluate(model, test_loader, mapper, device)
    print("Test Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    return model, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train GeoGuessr model")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to config YAML"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
