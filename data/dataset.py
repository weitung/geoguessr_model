"""PyTorch Dataset for GeoGuessr image-country pairs."""

import json

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# Approximate GPS centroids for each country (lat, lon)
COUNTRY_CENTROIDS = {
    "Argentina": (-34.6, -58.4),
    "Australia": (-25.3, 133.8),
    "Austria": (47.5, 14.6),
    "Bangladesh": (23.7, 90.4),
    "Belgium": (50.5, 4.5),
    "Bolivia": (-16.3, -63.6),
    "Botswana": (-22.3, 24.7),
    "Brazil": (-14.2, -51.9),
    "Bulgaria": (42.7, 25.5),
    "Cambodia": (12.6, 105.0),
    "Canada": (56.1, -106.3),
    "Chile": (-35.7, -71.5),
    "Colombia": (4.6, -74.1),
    "Croatia": (45.1, 15.2),
    "Czechia": (49.8, 15.5),
    "Denmark": (56.3, 9.5),
    "Finland": (61.9, 25.7),
    "France": (46.2, 2.2),
    "Germany": (51.2, 10.4),
    "Ghana": (7.9, -1.0),
    "Greece": (39.1, 21.8),
    "Hungary": (47.2, 19.5),
    "India": (20.6, 79.0),
    "Indonesia": (-0.8, 113.9),
    "Ireland": (53.4, -8.2),
    "Israel": (31.0, 34.9),
    "Italy": (41.9, 12.6),
    "Japan": (36.2, 138.3),
    "Kenya": (-0.02, 37.9),
    "Latvia": (56.9, 24.1),
    "Lithuania": (55.2, 23.9),
    "Malaysia": (4.2, 101.0),
    "Mexico": (23.6, -102.6),
    "Netherlands": (52.1, 5.3),
    "New Zealand": (-40.9, 174.9),
    "Nigeria": (9.1, 8.7),
    "Norway": (60.5, 8.5),
    "Peru": (-9.2, -75.0),
    "Philippines": (12.9, 121.8),
    "Poland": (51.9, 19.1),
    "Portugal": (39.4, -8.2),
    "Romania": (45.9, 24.97),
    "Russia": (61.5, 105.3),
    "Singapore": (1.35, 103.8),
    "Slovakia": (48.7, 19.7),
    "South Africa": (-30.6, 22.9),
    "South Korea": (35.9, 127.8),
    "Spain": (40.5, -3.7),
    "Sweden": (60.1, 18.6),
    "Switzerland": (46.8, 8.2),
    "Taiwan": (23.7, 121.0),
    "Thailand": (15.9, 100.0),
    "Turkey": (39.0, 35.2),
    "Ukraine": (48.4, 31.2),
    "United Kingdom": (55.4, -3.4),
}


def haversine_np(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


class CountryMapper:
    """Maps country names to integer labels and provides GPS centroids."""

    def __init__(self, countries: list[str] = None):
        if countries is None:
            countries = sorted(COUNTRY_CENTROIDS.keys())
        self.countries = countries
        self.country_to_idx = {c: i for i, c in enumerate(countries)}
        self.idx_to_country = {i: c for c, i in self.country_to_idx.items()}
        self.num_classes = len(countries)

    def encode(self, country: str) -> int:
        return self.country_to_idx[country]

    def decode(self, idx: int) -> str:
        return self.idx_to_country[idx]

    def get_centroid(self, idx: int) -> tuple[float, float]:
        """Return (lat, lon) for a country index."""
        country = self.decode(idx)
        return COUNTRY_CENTROIDS[country]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"countries": self.countries}, f)

    @classmethod
    def load(cls, path: str) -> "CountryMapper":
        with open(path) as f:
            data = json.load(f)
        return cls(countries=data["countries"])


class GeoGuessrDataset(Dataset):
    """PyTorch Dataset for GeoGuessr image-country pairs.

    Each sample returns:
    - image: preprocessed image tensor (3, H, W)
    - label: integer country label
    - country: country name string
    """

    def __init__(
        self,
        data_dir: str = "data/raw/geoguessr",
        split: str = "train",
        image_size: int = 224,
        country_mapper: CountryMapper = None,
        augment: bool = False,
    ):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size

        # Load HuggingFace dataset from disk
        ds = load_from_disk(data_dir)
        if hasattr(ds, "keys") and split in ds.keys():
            self.ds = ds[split]
        elif hasattr(ds, "keys") and "train" in ds.keys():
            self.ds = ds["train"]
        else:
            self.ds = ds

        # Build or use provided country mapper
        if country_mapper is None:
            all_labels = sorted(set(self.ds["label"]))
            self.country_mapper = CountryMapper(all_labels)
        else:
            self.country_mapper = country_mapper

        # CLIP-style preprocessing
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        image_size, scale=(0.8, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        image_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        item = self.ds[idx]

        # Handle image
        img = item["image"]
        if isinstance(img, str):
            img = Image.open(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        image = self.transform(img)

        # Country label
        country = item["label"]
        label = self.country_mapper.encode(country)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }


class GeoGuessrMapDataset(Dataset):
    """Dataset from a scraped GeoGuessr map (metadata.csv + image files).

    Each sample returns:
    - image: preprocessed image tensor (3, H, W)
    - label: integer country label
    - coords: (lat, lng) tuple
    """

    def __init__(
        self,
        metadata_csv: str,
        image_size: int = 224,
        country_mapper: CountryMapper = None,
        augment: bool = False,
    ):
        import csv

        self.image_size = image_size

        # Load metadata
        self.samples = []
        with open(metadata_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                cc = row.get("country_code", "").strip()
                if not cc:
                    continue
                self.samples.append({
                    "image_path": row["image_path"],
                    "lat": float(row["latitude"]),
                    "lng": float(row["longitude"]),
                    "heading": float(row.get("heading", 0)),
                    "pitch": float(row.get("pitch", 0)),
                    "country_code": cc,
                })

        # Build country mapper from data if not provided
        if country_mapper is None:
            all_codes = sorted(set(s["country_code"] for s in self.samples))
            self.country_mapper = CountryMapper(all_codes)
        else:
            self.country_mapper = country_mapper
            # Filter samples to only include known countries
            self.samples = [
                s for s in self.samples
                if s["country_code"] in self.country_mapper.country_to_idx
            ]

        # CLIP-style preprocessing
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size, scale=(0.8, 1.0),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
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
                transforms.Resize(
                    image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]

        img = Image.open(s["image_path"])
        if img.mode != "RGB":
            img = img.convert("RGB")
        image = self.transform(img)

        label = self.country_mapper.encode(s["country_code"])

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "coords": torch.tensor([s["lat"], s["lng"]], dtype=torch.float32),
        }


def create_map_datasets(
    metadata_csv: str,
    image_size: int = 224,
    augment_train: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple["GeoGuessrMapDataset", "GeoGuessrMapDataset", "GeoGuessrMapDataset", CountryMapper]:
    """Create train/val/test splits from a scraped GeoGuessr map dataset.

    Returns (train_ds, val_ds, test_ds, country_mapper).
    """
    from torch.utils.data import Subset

    full_ds = GeoGuessrMapDataset(metadata_csv, image_size, augment=False)
    mapper = full_ds.country_mapper

    n = len(full_ds)
    indices = list(range(n))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_ds = GeoGuessrMapDataset(metadata_csv, image_size, country_mapper=mapper, augment=augment_train)
    val_ds = GeoGuessrMapDataset(metadata_csv, image_size, country_mapper=mapper, augment=False)
    test_ds = GeoGuessrMapDataset(metadata_csv, image_size, country_mapper=mapper, augment=False)

    return (
        Subset(train_ds, train_idx),
        Subset(val_ds, val_idx),
        Subset(test_ds, test_idx),
        mapper,
    )


def create_datasets(
    data_dir: str = "data/raw/geoguessr",
    image_size: int = 224,
    augment_train: bool = True,
) -> tuple[GeoGuessrDataset, GeoGuessrDataset, GeoGuessrDataset, CountryMapper]:
    """Create train/val/test datasets using the dataset's built-in splits.

    Returns (train_ds, val_ds, test_ds, country_mapper).
    """
    # Build mapper from train split
    train_ds = GeoGuessrDataset(data_dir, "train", image_size, augment=augment_train)
    mapper = train_ds.country_mapper

    val_ds = GeoGuessrDataset(data_dir, "validation", image_size, country_mapper=mapper)
    test_ds = GeoGuessrDataset(data_dir, "test", image_size, country_mapper=mapper)

    return train_ds, val_ds, test_ds, mapper
