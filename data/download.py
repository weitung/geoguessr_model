"""Download GeoGuessr datasets from HuggingFace."""

import argparse
import os

from datasets import load_dataset


def download_geoguessr(data_dir: str = "data/raw") -> None:
    """Download marcelomoreno26/geoguessr dataset (~36k images, 55 countries)."""
    print("Downloading marcelomoreno26/geoguessr dataset...")
    ds = load_dataset("marcelomoreno26/geoguessr")
    os.makedirs(data_dir, exist_ok=True)
    ds.save_to_disk(os.path.join(data_dir, "geoguessr"))
    print(f"Saved dataset to {os.path.join(data_dir, 'geoguessr')}")

    # Print summary
    for split_name, split_ds in ds.items():
        print(f"  {split_name}: {len(split_ds)} examples")
        print(f"  Columns: {split_ds.column_names}")


def download_osv5m_subset(data_dir: str = "data/raw", max_samples: int = 50000) -> None:
    """Download a subset of OSV-5M for additional training data."""
    print(f"Downloading OSV-5M subset (max {max_samples} samples)...")
    ds = load_dataset("osv5m/osv5m", split=f"train[:{max_samples}]", streaming=False)
    os.makedirs(data_dir, exist_ok=True)
    ds.save_to_disk(os.path.join(data_dir, "osv5m_subset"))
    print(f"Saved OSV-5M subset to {os.path.join(data_dir, 'osv5m_subset')}")
    print(f"  {len(ds)} examples")
    print(f"  Columns: {ds.column_names}")


def main():
    parser = argparse.ArgumentParser(description="Download GeoGuessr datasets")
    parser.add_argument("--data-dir", default="data/raw", help="Directory to save data")
    parser.add_argument(
        "--include-osv5m", action="store_true", help="Also download OSV-5M subset"
    )
    parser.add_argument(
        "--osv5m-samples", type=int, default=50000, help="Number of OSV-5M samples"
    )
    args = parser.parse_args()

    download_geoguessr(args.data_dir)

    if args.include_osv5m:
        download_osv5m_subset(args.data_dir, args.osv5m_samples)

    print("\nDone! Dataset ready for training.")


if __name__ == "__main__":
    main()
