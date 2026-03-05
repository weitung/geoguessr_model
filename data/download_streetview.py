"""Download Google Street View images for collected GeoGuessr locations.

Usage:
    python data/download_streetview.py \
        --locations data/raw/map_locations.json \
        --api-key YOUR_GOOGLE_API_KEY \
        --output-dir data/raw/streetview_images

Requires a Google Cloud API key with Street View Static API enabled.
Free tier: 25,000 requests/month.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import requests


STREETVIEW_URL = "https://maps.googleapis.com/maps/api/streetview"
METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"


def check_pano_availability(
    session: requests.Session, pano_id: str, api_key: str
) -> bool:
    """Check if a Street View panorama exists."""
    resp = session.get(
        METADATA_URL,
        params={"pano": pano_id, "key": api_key},
    )
    if resp.status_code == 200:
        data = resp.json()
        return data.get("status") == "OK"
    return False


def download_image(
    session: requests.Session,
    output_path: str,
    api_key: str,
    pano_id: str | None = None,
    lat: float | None = None,
    lng: float | None = None,
    heading: float = 0,
    pitch: float = 0,
    fov: int = 90,
    size: str = "640x640",
) -> bool:
    """Download a single Street View image."""
    params = {
        "size": size,
        "heading": heading,
        "pitch": pitch,
        "fov": fov,
        "key": api_key,
        "location": f"{lat},{lng}",
    }

    resp = session.get(STREETVIEW_URL, params=params)
    if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(resp.content)
        return True
    return False


def download_streetview_images(
    locations_path: str,
    api_key: str,
    output_dir: str = "data/raw/streetview_images",
    num_headings: int = 1,
    size: str = "640x640",
    delay: float = 0.1,
):
    """Download Street View images for all locations.

    Args:
        locations_path: Path to JSON file from collect_map.py
        api_key: Google Cloud API key
        output_dir: Directory to save images
        num_headings: Number of heading angles per location (1=original orientation, 2, or 4)
        size: Image size (max 640x640 for free tier)
        delay: Delay between requests in seconds
    """
    with open(locations_path) as f:
        data = json.load(f)

    locations = data["locations"]
    print(f"Loaded {len(locations)} locations from {locations_path}")

    if num_headings == 1:
        heading_offsets = [0]
    elif num_headings == 2:
        heading_offsets = [0, 180]
    else:
        heading_offsets = [0, 90, 180, 270]

    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.csv")

    # Load existing progress
    existing = set()
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row["image_path"])
        print(f"Resuming: {len(existing)} images already downloaded")

    write_header = not os.path.exists(metadata_path)
    csv_file = open(metadata_path, "a", newline="")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "location_id", "latitude", "longitude", "heading", "pitch",
            "zoom", "pano_id", "country_code", "image_path",
        ],
    )
    if write_header:
        writer.writeheader()

    session = requests.Session()
    downloaded = 0
    skipped = 0
    failed = 0

    total_images = len(locations) * len(heading_offsets)
    print(f"Downloading {total_images} images ({len(locations)} locations × {len(heading_offsets)} headings)")

    for loc_idx, loc in enumerate(locations):
        base_heading = loc.get("heading", 0)

        for offset in heading_offsets:
            heading = (base_heading + offset) % 360
            filename = f"loc_{loc_idx:04d}_h{int(heading):03d}.jpg"
            image_path = os.path.join(output_dir, filename)

            if image_path in existing:
                skipped += 1
                continue

            success = download_image(
                session,
                image_path,
                api_key,
                pano_id=loc.get("panoId"),
                lat=loc["lat"],
                lng=loc["lng"],
                heading=heading,
                pitch=max(loc.get("pitch", 0), -10),  # Clamp pitch
                size=size,
            )

            if success:
                writer.writerow({
                    "location_id": loc_idx,
                    "latitude": loc["lat"],
                    "longitude": loc["lng"],
                    "heading": heading,
                    "pitch": loc.get("pitch", 0),
                    "zoom": loc.get("zoom", 0),
                    "pano_id": loc.get("panoId", ""),
                    "country_code": loc.get("countryCode", ""),
                    "image_path": image_path,
                })
                downloaded += 1
            else:
                failed += 1

            if (downloaded + failed) % 50 == 0:
                csv_file.flush()
                print(
                    f"  Progress: {downloaded} downloaded, {failed} failed, "
                    f"{skipped} skipped ({loc_idx + 1}/{len(locations)} locations)"
                )

            time.sleep(delay)

    csv_file.close()
    print(f"\nDone! {downloaded} downloaded, {failed} failed, {skipped} skipped")
    print(f"Images saved to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Street View images for GeoGuessr locations"
    )
    parser.add_argument(
        "--locations", required=True,
        help="Path to locations JSON from collect_map.py"
    )
    parser.add_argument("--api-key", required=True, help="Google Cloud API key")
    parser.add_argument(
        "--output-dir", default="data/raw/streetview_images",
        help="Output directory for images"
    )
    parser.add_argument(
        "--headings", type=int, default=1, choices=[1, 2, 4],
        help="Number of heading angles per location (default 4)"
    )
    parser.add_argument(
        "--size", default="640x640",
        help="Image size (default 640x640)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.1,
        help="Delay between requests in seconds (default 0.1)"
    )
    args = parser.parse_args()

    download_streetview_images(
        locations_path=args.locations,
        api_key=args.api_key,
        output_dir=args.output_dir,
        num_headings=args.headings,
        size=args.size,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
