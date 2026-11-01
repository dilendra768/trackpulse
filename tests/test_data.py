from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def test_sample_dataset_has_expected_columns_and_range():
    df = pd.read_csv(ROOT / "data" / "sample_tracks.csv")
    expected = {
        "track_id", "track_name", "artist_name", "popularity",
        "danceability", "energy", "loudness", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence",
        "tempo", "duration_ms", "key", "mode", "time_signature", "release_year"
    }
    assert expected.issubset(df.columns)
    assert len(df) >= 100
    assert df["popularity"].between(0, 100).all()
    assert df["danceability"].between(0, 1).all()
