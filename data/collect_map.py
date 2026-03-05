"""Collect location data from a GeoGuessr map by playing games via the API.

Usage:
    python data/collect_map.py \
        --map-id 61a1846aee665b00016680ce \
        --cookie "YOUR_NCFA_COOKIE" \
        --num-games 300 \
        --output data/raw/map_locations.json

This creates games on the specified map, plays through them (submitting dummy
guesses), and records every location's coordinates, panoId, heading, etc.
Since GeoGuessr randomly samples locations from the map, we play many games
and deduplicate by panoId to collect as many unique locations as possible.
"""

import argparse
import json
import os
import time

import requests


GEOGUESSR_API = "https://www.geoguessr.com/api/v3"


def create_game(session: requests.Session, map_id: str) -> dict:
    """Create a new single-player game on the given map."""
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
    resp.raise_for_status()
    return resp.json()


def submit_guess(session: requests.Session, game_token: str) -> dict:
    """Submit a dummy guess (0,0) and then GET game state to advance round."""
    resp = session.post(
        f"{GEOGUESSR_API}/games/{game_token}",
        json={
            "lat": 0,
            "lng": 0,
            "timedOut": False,
            "stepsCount": 0,
        },
    )
    resp.raise_for_status()
    # Must GET game state to see the next round
    resp = session.get(f"{GEOGUESSR_API}/games/{game_token}")
    resp.raise_for_status()
    return resp.json()


def extract_locations_from_game(session: requests.Session, map_id: str) -> list[dict]:
    """Play through one game and return all 5 locations."""
    game = create_game(session, map_id)
    game_token = game["token"]
    seen_panos = set()
    locations = []

    def _add_rounds(data):
        for round_data in data.get("rounds", []):
            pano = round_data.get("panoId")
            if pano and pano not in seen_panos:
                seen_panos.add(pano)
                locations.append(_extract_location(round_data))

    # Round 1 is already in the game creation response
    _add_rounds(game)

    # Submit guesses for rounds 1-5 to reveal all locations
    for _ in range(4):
        time.sleep(0.3)
        result = submit_guess(session, game_token)
        _add_rounds(result)

    return locations


def _extract_location(round_data: dict) -> dict:
    return {
        "lat": round_data["lat"],
        "lng": round_data["lng"],
        "panoId": round_data.get("panoId"),
        "heading": round_data.get("heading", 0),
        "pitch": round_data.get("pitch", 0),
        "zoom": round_data.get("zoom", 0),
        "countryCode": round_data.get("streakLocationCode"),
    }


def collect_locations(
    cookie: str,
    map_id: str,
    num_games: int = 300,
    output_path: str = "data/raw/map_locations.json",
    delay: float = 1.5,
) -> dict:
    """Collect unique locations from a GeoGuessr map by playing multiple games.

    Returns dict with metadata and list of unique locations.
    """
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
        "Cookie": f"_ncfa={cookie}",
    })

    # Load existing progress if resuming
    seen_pano_ids = set()
    all_locations = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
            all_locations = existing.get("locations", [])
            seen_pano_ids = {loc["panoId"] for loc in all_locations if loc.get("panoId")}
            print(f"Resuming: {len(all_locations)} locations already collected")

    games_played = 0
    consecutive_dupes = 0

    for game_idx in range(num_games):
        try:
            locs = extract_locations_from_game(session, map_id)
            games_played += 1
            new_count = 0

            for loc in locs:
                pano_id = loc.get("panoId")
                if pano_id and pano_id not in seen_pano_ids:
                    seen_pano_ids.add(pano_id)
                    all_locations.append(loc)
                    new_count += 1

            if new_count == 0:
                consecutive_dupes += 1
            else:
                consecutive_dupes = 0

            print(
                f"Game {game_idx + 1}/{num_games}: "
                f"+{new_count} new, {len(all_locations)} total unique"
            )

            # Early stop if we keep getting duplicates (likely collected most locations)
            if consecutive_dupes >= 30:
                print(
                    f"Stopping early: {consecutive_dupes} consecutive games "
                    f"with no new locations"
                )
                break

            # Save progress periodically
            if (game_idx + 1) % 10 == 0:
                _save(output_path, map_id, all_locations, games_played)

            time.sleep(delay)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print("Rate limited, waiting 30s...")
                time.sleep(30)
            else:
                print(f"HTTP error: {e}")
                time.sleep(5)
        except Exception as e:
            print(f"Error in game {game_idx + 1}: {e}")
            time.sleep(5)

    _save(output_path, map_id, all_locations, games_played)
    print(f"\nDone! Collected {len(all_locations)} unique locations in {games_played} games")
    return {"map_id": map_id, "locations": all_locations}


def _save(path: str, map_id: str, locations: list, games_played: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "map_id": map_id,
        "total_locations": len(locations),
        "games_played": games_played,
        "locations": locations,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect locations from a GeoGuessr map"
    )
    parser.add_argument("--map-id", required=True, help="GeoGuessr map ID")
    parser.add_argument("--cookie", required=True, help="GeoGuessr _ncfa cookie value")
    parser.add_argument(
        "--num-games", type=int, default=300,
        help="Max games to play (default 300, ~1500 location samples)"
    )
    parser.add_argument(
        "--output", default="data/raw/map_locations.json",
        help="Output JSON path"
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="Delay between games in seconds (default 1.5)"
    )
    args = parser.parse_args()

    collect_locations(
        cookie=args.cookie,
        map_id=args.map_id,
        num_games=args.num_games,
        output_path=args.output,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
