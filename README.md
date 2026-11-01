# TrackPulse 🎧 — Music Popularity Prediction

A portfolio project using **Python, SQL and machine learning** to explore how measurable audio characteristics relate to track popularity and to build a baseline prediction model.

## What it does

- Explores audio features such as danceability, energy, loudness, acousticness, valence and tempo.
- Trains a Random Forest regression model to estimate a track popularity score.
- Compares the model with a simple mean-prediction baseline.
- Reports MAE, RMSE and R² on a held-out test set.
- Exports feature-importance results.
- Loads the dataset into SQLite and includes practical analytical SQL queries.
- Provides an interactive Streamlit dashboard for exploration and what-if predictions.

## Important dataset note

The `data/sample_tracks.csv` file (created by `python generate_sample_data.py`) contains **600 synthetic demonstration records**, not real Spotify listening data. The popularity values were generated from a deliberately simple formula with noise so the project runs immediately and can demonstrate the workflow. Metrics from this synthetic dataset are not evidence about real-world music popularity.

For a credible research or hiring portfolio, replace the sample with a legally obtained, well-documented real dataset. Check its license, attribution requirements, collection date, sampling method and whether its popularity field is defined consistently. Do not scrape or redistribute data in violation of a platform's terms.

## Project structure

```text
trackpulse/
├── data/
│   └── sample_tracks.csv    # created by generate_sample_data.py
├── models/                  # Created by training
├── sql/
│   └── analysis_queries.sql
├── tests/
│   └── test_data.py
├── generate_sample_data.py  # creates the synthetic demo dataset
├── train.py                 # ML training + evaluation
├── database.py              # SQLite ingestion
├── dashboard.py             # Streamlit application
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

Python 3.10 or newer recommended.

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the demo dataset

```bash
python generate_sample_data.py
```

### 4. Train the model

```bash
python train.py
```

The script writes:
- `models/popularity_model.joblib`
- `models/metrics.json`
- `models/feature_importance.csv`

### 5. Load data into SQLite

```bash
python database.py
```

This creates `data/music_analytics.db` with a `tracks` table.

### 6. Launch the interactive dashboard

```bash
streamlit run dashboard.py
```

### 7. Run tests

```bash
pytest -q
```

## Use a real dataset

The training script accepts a CSV path:

```bash
python train.py --data path/to/your_tracks.csv
```

The CSV must contain these columns:

`popularity, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms, key, mode, time_signature, release_year`

The dataset can contain additional columns; the trainer selects the required numeric fields. If the source has a different schema, write and document a cleaning/mapping step rather than silently renaming columns.

## Methodology

1. Validate required columns and convert feature fields to numeric.
2. Drop rows without the target value.
3. Split into train and test sets with a fixed random seed.
4. Impute missing feature values with training-set medians inside a scikit-learn pipeline.
5. Compare a `DummyRegressor` baseline against a `RandomForestRegressor`.
6. Evaluate with:
   - **MAE:** average absolute prediction error, in popularity points.
   - **RMSE:** penalizes larger errors more strongly.
   - **R²:** variance explained relative to a mean baseline; can be negative.
7. Export feature importance as a model-specific diagnostic, not a causal explanation.

## Research questions worth investigating

- How strongly do audio features associate with a popularity score?
- Does model performance differ for older and newer releases?
- Do audio features alone provide useful predictive power?
- How much does performance change when release year is excluded?
- Does the model generalize to a later time period?

For a stronger study, use a **time-based split** (train on older releases, test on newer releases) rather than only a random split. Avoid data leakage, document how popularity is measured, and compare with simple baselines.

## Known limitations

- Popularity is influenced by promotion, artist recognition, playlist placement, social trends, release timing and many factors absent from audio features.
- A Random Forest feature-importance score does not establish causation.
- The sample data is synthetic.
- The current evaluation uses a random holdout. Use a chronological split for a more realistic future-release experiment.
- This is a learning prototype, not a production recommendation engine.

## Suggested next upgrades

1. Add cross-validation and a time-based holdout.
2. Add permutation importance and SHAP explanations.
3. Compare Random Forest with Gradient Boosting and linear regression.
4. Add confidence intervals or prediction uncertainty.
5. Build a recommendation feature based on audio-feature similarity.
6. Add a documented real-world dataset and a reproducible data-cleaning notebook.
7. Add GitHub Actions for tests and linting.

## Suggested GitHub description

`End-to-end music popularity prediction project using Python, scikit-learn, SQLite, SQL analytics and Streamlit. Includes reproducible training, baseline comparison and an interactive dashboard.`

## License

MIT for the original project code. The synthetic dataset generated by this project is also provided for demonstration under MIT. Any replacement dataset retains its own license and terms.
