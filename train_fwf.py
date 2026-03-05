"""Fine-tune on Fun with Flags map, starting from marcelomoreno26 checkpoint.

- Loads best.pt checkpoint (55-class model)
- 80:20 train:test split on fun_with_flags images
- Test split filtered to countries present in marcelomoreno26 dataset
- Evaluates accuracy at end
"""

import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from data.dataset import CountryMapper, haversine_np
from models.geoclip_finetune import GeoGuessrModel

# Country code -> marcelomoreno26 country name (52 overlapping countries)
CODE_TO_COUNTRY = {
    "ar": "Argentina", "au": "Australia", "at": "Austria", "bd": "Bangladesh",
    "be": "Belgium", "bo": "Bolivia", "bw": "Botswana", "br": "Brazil",
    "bg": "Bulgaria", "kh": "Cambodia", "ca": "Canada", "cl": "Chile",
    "co": "Colombia", "hr": "Croatia", "dk": "Denmark", "fi": "Finland",
    "fr": "France", "de": "Germany", "gh": "Ghana", "gr": "Greece",
    "hu": "Hungary", "in": "India", "id": "Indonesia", "ie": "Ireland",
    "il": "Israel", "it": "Italy", "jp": "Japan", "ke": "Kenya",
    "kr": "South Korea", "lv": "Latvia", "lt": "Lithuania", "my": "Malaysia",
    "mx": "Mexico", "nl": "Netherlands", "nz": "New Zealand", "ng": "Nigeria",
    "no": "Norway", "pe": "Peru", "ph": "Philippines", "pl": "Poland",
    "pt": "Portugal", "ro": "Romania", "ru": "Russia", "sg": "Singapore",
    "sk": "Slovakia", "za": "South Africa", "es": "Spain", "se": "Sweden",
    "ch": "Switzerland", "tw": "Taiwan", "th": "Thailand", "tr": "Turkey",
    "ua": "Ukraine", "gb": "United Kingdom",
}


class FunWithFlagsDataset(Dataset):
    """Dataset for fun_with_flags images using the original 55-class mapper."""

    def __init__(self, samples: list[dict], mapper: CountryMapper, augment: bool = False):
        self.samples = samples
        self.mapper = mapper

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        image = self.transform(img)
        label = self.mapper.encode(s["country_name"])
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_samples(metadata_csv: str) -> list[dict]:
    """Load samples, mapping country codes to country names."""
    samples = []
    with open(metadata_csv) as f:
        for row in csv.DictReader(f):
            cc = row.get("country_code", "").strip()
            country_name = CODE_TO_COUNTRY.get(cc)
            if country_name is None:
                continue
            samples.append({
                "image_path": row["image_path"],
                "lat": float(row["latitude"]),
                "lng": float(row["longitude"]),
                "country_code": cc,
                "country_name": country_name,
            })
    return samples


def evaluate(model, dataloader, mapper, device):
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

            for pl in preds.cpu().numpy():
                all_pred_gps.append(mapper.get_centroid(int(pl)))
            for tl in labels.cpu().numpy():
                all_true_gps.append(mapper.get_centroid(int(tl)))

    pred_gps = np.array(all_pred_gps)
    true_gps = np.array(all_true_gps)
    distances = haversine_np(
        pred_gps[:, 0], pred_gps[:, 1], true_gps[:, 0], true_gps[:, 1]
    )

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "acc_750km": float(np.mean(distances < 750)),
        "acc_2500km": float(np.mean(distances < 2500)),
        "median_km": float(np.median(distances)),
        "mean_km": float(np.mean(distances)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load original mapper and checkpoint
    mapper = CountryMapper.load("checkpoints/country_mapper.json")
    print(f"Original mapper: {mapper.num_classes} classes")

    checkpoint = torch.load("checkpoints/best.pt", map_location=device, weights_only=False)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, "
          f"val_acc={checkpoint['val_metrics']['accuracy']:.4f}")

    # Load fun_with_flags samples (only those matching original 55 countries)
    metadata_csv = "data/raw/fun_with_flags/images/metadata.csv"
    all_samples = load_samples(metadata_csv)
    print(f"\nFun with Flags samples (overlapping countries): {len(all_samples)}")

    # Count per country
    from collections import Counter
    country_counts = Counter(s["country_name"] for s in all_samples)
    print(f"Countries represented: {len(country_counts)}")

    # 80:20 train:test split (stratified shuffle)
    rng = np.random.RandomState(42)
    indices = list(range(len(all_samples)))
    rng.shuffle(indices)
    n_train = int(len(all_samples) * 0.8)

    train_samples = [all_samples[i] for i in indices[:n_train]]
    test_samples = [all_samples[i] for i in indices[n_train:]]
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")

    # Create datasets
    train_ds = FunWithFlagsDataset(train_samples, mapper, augment=True)
    test_ds = FunWithFlagsDataset(test_samples, mapper, augment=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Build model from checkpoint
    model = GeoGuessrModel(
        num_classes=mapper.num_classes,
        freeze_clip=True,
        hidden_dim=512,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded, trainable params: {model.num_trainable_params():,}")

    # Evaluate before fine-tuning
    print("\n--- Before Fine-tuning ---")
    pre_metrics = evaluate(model, test_loader, mapper, device)
    for k, v in pre_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Fine-tune
    num_epochs = 30
    lr = 3e-5
    patience = 7

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=1e-4)

    # Cosine schedule with warmup
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = 2 * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_acc = 0.0
    patience_counter = 0
    save_dir = "checkpoints"

    print(f"\n--- Fine-tuning ({num_epochs} epochs, lr={lr}) ---")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()

        for batch in train_loader:
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

        train_loss = epoch_loss / epoch_total
        train_acc = epoch_correct / epoch_total
        elapsed = time.time() - t0

        # Evaluate on test set
        test_metrics = evaluate(model, test_loader, mapper, device)

        print(
            f"Epoch {epoch+1:2d}/{num_epochs} ({elapsed:.0f}s) | "
            f"Train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"Test loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.3f} | "
            f"750km={test_metrics['acc_750km']:.3f} 2500km={test_metrics['acc_2500km']:.3f} | "
            f"median={test_metrics['median_km']:.0f}km"
        )

        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "test_metrics": test_metrics,
                "num_classes": mapper.num_classes,
            }, os.path.join(save_dir, "best_fwf.pt"))
            print(f"  -> Saved best (acc={best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

    # Final evaluation with best model
    print("\n--- Final Test Evaluation ---")
    best_ckpt = torch.load(os.path.join(save_dir, "best_fwf.pt"), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final_metrics = evaluate(model, test_loader, mapper, device)
    print(f"Best epoch: {best_ckpt['epoch']}")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
