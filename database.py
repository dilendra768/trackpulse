"""Load track records into SQLite and run reproducible SQL summaries."""
from pathlib import Path
import argparse
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "music_analytics.db"
DEFAULT_CSV = ROOT / "data" / "sample_tracks.csv"


def load_database(csv_path=DEFAULT_CSV, db_path=DEFAULT_DB):
    df = pd.read_csv(csv_path)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("tracks", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tracks_popularity ON tracks(popularity)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist_name)")
    return len(df)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args()
    count = load_database(args.csv, args.db)
    print(f"Loaded {count} tracks into {args.db}")


if __name__ == "__main__":
    main()
