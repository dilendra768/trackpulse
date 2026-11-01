"""Generate the synthetic demonstration dataset used by TrackPulse.

The popularity target comes from a deliberately simple formula plus noise, so the
workflow runs end to end. It says nothing about real-world music popularity.
Usage: python generate_sample_data.py [--rows 600] [--seed 42]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "sample_tracks.csv"
ARTISTS = ["Juniper Sky", "Velvet Circuit", "Nova Lane", "The Paper Moons", "Glass Avenue",
           "Echo Harbor", "Northbound", "Lunar Arcade", "Mira Sol", "Static Bloom"]


def generate(rows: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "track_id": [f"demo_{i:04d}" for i in range(1, rows + 1)],
        "track_name": [f"Demo Track {i:03d}" for i in range(1, rows + 1)],
        "artist_name": rng.choice(ARTISTS, rows),
        "danceability": rng.beta(4, 3, rows).clip(0.05, 0.98),
        "energy": rng.beta(4, 3, rows).clip(0.03, 0.99),
        "loudness": rng.normal(-10, 5, rows).clip(-30, -1.5),
        "speechiness": rng.beta(1.2, 8, rows).clip(0.0004, 0.7),
        "acousticness": rng.beta(1.5, 2.5, rows).clip(0.01, 0.99),
        "instrumentalness": rng.beta(0.5, 5, rows).clip(0.0008, 0.9),
        "liveness": rng.beta(1.5, 7, rows).clip(0.0015, 0.8),
        "valence": rng.beta(3, 3, rows).clip(0.02, 0.99),
        "tempo": rng.normal(120, 28, rows).clip(55, 200),
        "duration_ms": rng.normal(210000, 45000, rows).clip(90000, 360000),
        "key": rng.integers(0, 12, rows),
        "mode": rng.integers(0, 2, rows),
        "time_signature": rng.choice([3, 4, 5, 7], rows, p=[0.1, 0.8, 0.07, 0.03]),
        "release_year": rng.integers(2000, 2026, rows),
    })
    signal = (28 * df.danceability + 18 * df.energy + 10 * df.valence
              + 0.6 * df.loudness - 12 * df.acousticness - 10 * df.instrumentalness
              + 0.25 * (df.release_year - 2000))
    df["popularity"] = (signal + 18 + rng.normal(0, 9, rows)).round().clip(0, 100).astype(int)
    cols = ["track_id", "track_name", "artist_name", "popularity", "danceability", "energy",
            "loudness", "speechiness", "acousticness", "instrumentalness", "liveness",
            "valence", "tempo", "duration_ms", "key", "mode", "time_signature", "release_year"]
    df = df[cols]
    for c, n in [("danceability", 4), ("energy", 4), ("loudness", 2), ("speechiness", 4),
                 ("acousticness", 4), ("instrumentalness", 4), ("liveness", 4),
                 ("valence", 4), ("tempo", 1)]:
        df[c] = df[c].round(n)
    df["duration_ms"] = df["duration_ms"].round().astype(int)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    generate(a.rows, a.seed).to_csv(a.out, index=False)
    print(f"Wrote {a.rows} synthetic tracks to {a.out}")


if __name__ == "__main__":
    main()
