"""Streamlit dashboard for exploring and predicting track popularity."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "popularity_model.joblib"
DATA_PATH = ROOT / "data" / "sample_tracks.csv"
FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
    "key", "mode", "time_signature", "release_year"
]

st.set_page_config(page_title="TrackPulse | Music Popularity Lab", page_icon="🎧", layout="wide")
st.title("🎧 TrackPulse")
st.caption("Music popularity prediction · Python · SQL · Machine Learning")

if not MODEL_PATH.exists():
    st.warning("Model not found. Run `python train.py` in the project folder first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["pipeline"]
df = pd.read_csv(DATA_PATH)

with st.sidebar:
    st.header("Explore the dataset")
    min_pop = st.slider("Minimum popularity", 0, 100, 0)
    st.caption("Demo dataset is synthetic. Replace it with a properly licensed real dataset for research.")

filtered = df[df["popularity"] >= min_pop]
col1, col2, col3 = st.columns(3)
col1.metric("Tracks in view", f"{len(filtered):,}")
col2.metric("Average popularity", f"{filtered['popularity'].mean():.1f}" if len(filtered) else "—")
col3.metric("Average tempo", f"{filtered['tempo'].mean():.1f} BPM" if len(filtered) else "—")

left, right = st.columns(2)
with left:
    st.subheader("Popularity distribution")
    st.bar_chart(filtered["popularity"].value_counts(bins=10, sort=False).sort_index())
with right:
    st.subheader("Audio features and popularity")
    st.scatter_chart(filtered, x="energy", y="popularity", color="valence")

st.subheader("Try a prediction")
st.write("Adjust audio features to estimate a popularity score from 0 to 100.")
defaults = {
    "danceability": 0.65, "energy": 0.70, "loudness": -7.0,
    "speechiness": 0.05, "acousticness": 0.20, "instrumentalness": 0.01,
    "liveness": 0.12, "valence": 0.60, "tempo": 120.0,
    "duration_ms": 210000, "key": 5, "mode": 1,
    "time_signature": 4, "release_year": 2024
}
input_values = {}
sliders = st.columns(2)
for i, feature in enumerate(FEATURES):
    with sliders[i % 2]:
        if feature in ["key", "mode", "time_signature", "release_year", "duration_ms"]:
            bounds = {
                "key": (0, 11), "mode": (0, 1), "time_signature": (3, 7),
                "release_year": (1950, 2026), "duration_ms": (60000, 600000)
            }
            lo, hi = bounds[feature]
            input_values[feature] = st.number_input(feature.replace("_", " ").title(), min_value=lo, max_value=hi, value=defaults[feature], step=1)
        elif feature == "loudness":
            input_values[feature] = st.slider("Loudness (dB)", -60.0, 0.0, defaults[feature], 0.5)
        elif feature == "tempo":
            input_values[feature] = st.slider("Tempo (BPM)", 40.0, 220.0, defaults[feature], 1.0)
        else:
            input_values[feature] = st.slider(feature.title(), 0.0, 1.0, defaults[feature], 0.01)

if st.button("Estimate popularity", type="primary"):
    sample = pd.DataFrame([input_values], columns=FEATURES)
    estimate = float(model.predict(sample)[0])
    st.success(f"Estimated popularity: {max(0, min(100, estimate)):.1f} / 100")
    st.caption("This is a model estimate, not a guarantee. Results depend heavily on the dataset and its biases.")

st.subheader("Sample tracks")
st.dataframe(filtered.head(25), use_container_width=True, hide_index=True)
