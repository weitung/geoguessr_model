"""Screenshot GeoGuessr panoramas using Playwright.

Instead of using Google Street View Static API (which fails for unofficial
coverage maps), this script creates games on GeoGuessr, loads each round in
a headless browser, and screenshots the panorama view.

Usage:
    python data/screenshot_map.py \
        --locations data/raw/fun_with_flags/locations.json \
        --cookie "YOUR_NCFA_COOKIE" \
        --output-dir data/raw/fun_with_flags/images

Uses the already-collected locations from collect_map.py. Creates one game
per 5 locations needed, screenshots each round, then advances.
"""

import argparse
import csv
import json
import os
import time

import requests
from playwright.sync_api import sync_playwright


GEOGUESSR_API = "https://www.geoguessr.com/api/v3"


def create_game(session: requests.Session, map_id: str) -> dict | None:
    """Create a new game. Returns None if account is suspended/rate-limited."""
    resp = session.post(
        f"{GEOGUESSR_API}/games",
        json={
            "map": map_id,
            "type": "standard",
            "timeLimit": 0,
            "forbidMoving": False,
            "forbidZooming": False,
            "forbidRotating": False,
            "rounds": 5,
        },
    )
    if resp.status_code != 200:
        print(f"  Game creation failed: {resp.status_code} {resp.text[:100]}")
        return None
    return resp.json()


def submit_guess_and_advance(session: requests.Session, game_token: str) -> dict | None:
    """Submit guess and GET to advance round."""
    resp = session.post(
        f"{GEOGUESSR_API}/games/{game_token}",
        json={"lat": 0, "lng": 0, "timedOut": False, "stepsCount": 0},
    )
    if resp.status_code != 200:
        return None
    resp = session.get(f"{GEOGUESSR_API}/games/{game_token}")
    if resp.status_code != 200:
        return None
    return resp.json()


def screenshot_game_rounds(
    page,
    session: requests.Session,
    game_token: str,
    round_infos: list[dict],
    output_dir: str,
    writer,
    loc_index_start: int,
    delay: float,
) -> int:
    """Screenshot each round of a game. Returns number of screenshots taken."""
    count = 0

    for round_num in range(5):
        if round_num >= len(round_infos):
            break

        loc = round_infos[round_num]

        # Load game page (navigates to current round)
        page.goto(
            f"https://www.geoguessr.com/game/{game_token}",
            wait_until="domcontentloaded",
            timeout=30000,
        )

        # Wait for panorama to load
        page.wait_for_timeout(4000)

        # Hide UI overlays to get clean panorama screenshot
        page.evaluate("""
            // Hide game UI elements
            for (const sel of [
                '[class*="game-layout__controls"]',
                '[class*="game_controls"]',
                '[class*="panorama-compass"]',
                '[class*="game-layout__status"]',
                '[class*="status_inner"]',
                '[class*="round-score"]',
                '[class*="game_map"]',
                '[class*="minimap"]',
                '[class*="guess-map"]',
                'footer', 'header',
            ]) {
                for (const el of document.querySelectorAll(sel)) {
                    el.style.display = 'none';
                }
            }
        """)
        page.wait_for_timeout(500)

        loc_idx = loc_index_start + round_num
        filename = f"loc_{loc_idx:04d}.jpg"
        filepath = os.path.join(output_dir, filename)

        page.screenshot(path=filepath, type="jpeg", quality=90)

        writer.writerow({
            "location_id": loc_idx,
            "latitude": loc["lat"],
            "longitude": loc["lng"],
            "heading": loc.get("heading", 0),
            "pitch": loc.get("pitch", 0),
            "zoom": loc.get("zoom", 0),
            "pano_id": loc.get("panoId", ""),
            "country_code": loc.get("countryCode", ""),
            "image_path": filepath,
        })
        count += 1

        # Advance to next round (except last)
        if round_num < 4:
            time.sleep(0.5)
            result = submit_guess_and_advance(session, game_token)
            if result is None:
                print(f"    Failed to advance past round {round_num + 1}")
                break
            time.sleep(delay)

    return count


def screenshot_locations(
    locations_path: str,
    cookie: str,
    output_dir: str = "data/raw/fun_with_flags/images",
    delay: float = 5.0,
):
    """Screenshot panoramas for all collected locations.

    This creates new games on the same map. The locations won't match exactly
    (GeoGuessr randomly samples), but each screenshot captures the panorama
    at the orientation specified by the game round, which is what we want.
    """
    with open(locations_path) as f:
        data = json.load(f)

    map_id = data["map_id"]
    total_locs = data["total_locations"]
    print(f"Map {map_id}: {total_locs} locations to screenshot")

    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.csv")

    # Check existing progress
    existing_count = 0
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            existing_count = sum(1 for _ in csv.DictReader(f))
        print(f"Resuming: {existing_count} screenshots already taken")

    # API session
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "Cookie": f"_ncfa={cookie}",
    })

    # Number of games needed (5 locations per game)
    num_games = (total_locs - existing_count + 4) // 5

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

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 640, "height": 640})
        ctx.add_cookies([{
            "name": "_ncfa",
            "value": cookie,
            "domain": ".geoguessr.com",
            "path": "/",
        }])
        page = ctx.new_page()

        loc_idx = existing_count
        screenshots_taken = 0

        for game_num in range(num_games):
            if loc_idx >= total_locs:
                break

            game = create_game(session, map_id)
            if game is None:
                print("Account may be suspended. Waiting 60s...")
                time.sleep(60)
                game = create_game(session, map_id)
                if game is None:
                    print("Still failing. Stopping.")
                    break

            token = game["token"]
            rounds = game.get("rounds", [])

            # Extract round location info
            round_infos = []
            for r in rounds:
                round_infos.append({
                    "lat": r["lat"],
                    "lng": r["lng"],
                    "panoId": r.get("panoId"),
                    "heading": r.get("heading", 0),
                    "pitch": r.get("pitch", 0),
                    "zoom": r.get("zoom", 0),
                    "countryCode": r.get("streakLocationCode"),
                })

            # We only have round 1 from creation, need to play through for rest
            # But we can screenshot round 1 first, then advance
            n = screenshot_game_rounds(
                page, session, token, round_infos, output_dir, writer,
                loc_idx, delay,
            )
            screenshots_taken += n
            loc_idx += n

            csv_file.flush()
            print(
                f"Game {game_num + 1}: +{n} screenshots, "
                f"{screenshots_taken} total ({loc_idx}/{total_locs})"
            )

            # Longer delay between games to avoid suspension
            time.sleep(delay)

        browser.close()

    csv_file.close()
    print(f"\nDone! {screenshots_taken} screenshots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Screenshot GeoGuessr panoramas via Playwright"
    )
    parser.add_argument(
        "--locations", required=True,
        help="Path to locations JSON from collect_map.py"
    )
    parser.add_argument("--cookie", required=True, help="GeoGuessr _ncfa cookie")
    parser.add_argument(
        "--output-dir", default="data/raw/fun_with_flags/images",
        help="Output directory for screenshots"
    )
    parser.add_argument(
        "--delay", type=float, default=5.0,
        help="Delay between rounds/games in seconds (default 5.0, be gentle)"
    )
    args = parser.parse_args()

    screenshot_locations(
        locations_path=args.locations,
        cookie=args.cookie,
        output_dir=args.output_dir,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
