"""Train and evaluate a track-popularity regression model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "sample_tracks.csv"
MODEL_DIR = ROOT / "models"

NUMERIC_FEATURES = [
    "danceability", "energy", "loudness", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
    "key", "mode", "time_signature", "release_year"
]
CATEGORICAL_FEATURES = []  # Add categorical features only after checking leakage and cardinality.
TARGET = "popularity"


def build_pipeline():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    preprocess = ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES)], remainder="drop")
    model = RandomForestRegressor(
        n_estimators=350, max_depth=12, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="CSV file containing Spotify-like audio features")
    parser.add_argument("--target", default=TARGET, help="Target column (default: popularity)")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    missing = [c for c in NUMERIC_FEATURES + [args.target] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")
    df = df[NUMERIC_FEATURES + [args.target]].copy()
    for col in NUMERIC_FEATURES + [args.target]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[args.target])
    if len(df) < 30:
        raise SystemExit("Please provide at least 30 rows with a valid target value.")

    X = df[NUMERIC_FEATURES]
    y = df[args.target].clip(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    predictions = np.clip(pipeline.predict(X_test), 0, 100)

    metrics = {
        "rows_used": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "baseline_mae": round(float(mean_absolute_error(y_test, baseline_pred)), 3),
        "model_mae": round(float(mean_absolute_error(y_test, predictions)), 3),
        "model_rmse": round(float(np.sqrt(mean_squared_error(y_test, predictions))), 3),
        "model_r2": round(float(r2_score(y_test, predictions)), 3),
        "target": args.target,
        "features": NUMERIC_FEATURES,
        "data_file": str(args.data.name)
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": NUMERIC_FEATURES}, MODEL_DIR / "popularity_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    importances = pipeline.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame({"feature": NUMERIC_FEATURES, "importance": importances})
    importance_df.sort_values("importance", ascending=False).to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model and reports to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
