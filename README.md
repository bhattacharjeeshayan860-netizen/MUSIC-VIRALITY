# Music Virality System

**Production-grade ML system to detect currently-viral music videos and forecast short-horizon virality from YouTube snapshot data.**

Built like a small MLOps pipeline rather than a notebook demo: modular feature engineering, group-aware validation, threshold tuning on validation, SHAP interpretability, and real serving surfaces (Streamlit dashboards + FastAPI).

---

## Business Problem

Music labels, distributors, and marketing teams need to decide which videos to promote *before* they peak. Two related but distinct decisions:

| Task | Question | Value |
|---|---|---|
| **Virality Detection** | Is this video already viral right now? | Inventory triage, priority routing, reporting. |
| **Virality Prediction** | Will this video cross the virality threshold later? | Early promotion, playlist placement, ad spend allocation. |

Naive approaches collapse these two problems. A detection model trained on `view_count` perfectly separates classes but learns nothing useful. A prediction model that includes the same video in train and test will report fantasy numbers. This system treats them as separate targets with separate labels, features, and evaluation protocols.

---

## System Overview

```
Raw YouTube snapshots
        |
        v
Data cleaning  -->  Feature engineering  -->  Dual labels
        |                                    (is_viral / future_is_viral)
        v
Group-aware stratified split  -->  Train/val/test
        |
        v
Model selection + threshold tuning on validation
        |
        v
Honest test evaluation + pre-viral slice + SHAP
        |
        v
Serving: Streamlit dashboards + FastAPI predict endpoints
```

The repo separates concerns cleanly:

- `src/features/` — feature engineering + labeling
- `src/training/` — model training / evaluation
- `src/inference/` — single-source-of-truth feature builder for serving
- `src/api/` — FastAPI service + YouTube Data API client
- `src/pipelines/` — standalone data-collection pipeline
- `tests/` — deterministic pytest suite on synthetic data

---

## The Hardest Part: Leakage Control

On this dataset, the easiest way to get a high score is to leak information across time or across videos. The pipeline is designed to make that impossible.

### 1. Group-aware splits
Splits are done at the **video level** with `video_id` as the group key. The same video never appears in train and test. A naïve row-level split leaks ~77% of test rows because a video has multiple snapshots.

### 2. Stratified splits
Positive-class rates are preserved across train/val/test by stratifying on the per-video label. This matters because the dataset is heavily skewed (~11% detection positives, ~19% prediction positives).

| Split | Detection positive rate | Prediction positive rate |
|---|---|---|
| Train | 10.8% | 18.9% |
| Validation | 10.6% | 19.8% |
| Test | 11.3% | 19.4% |

### 3. Different feature sets for different targets
- **Detection** excludes `view_count_log` and `views_ratio_to_first`. The label is `view_count >= 10M`; including a monotonic transform of `view_count` would make the task a lookup table, not a model.
- **Prediction** includes current view scale because the label is a **future** snapshot, not the same row.

### 4. Validation-driven model selection and threshold tuning
The test set is touched exactly once. Model choice, hyperparameters, and classification thresholds are all selected on the validation split. This removes optimistic bias from repeated test peeking.

### 5. No SMOTE
Class imbalance is handled with `scale_pos_weight` / `class_weight="balanced"`. SMOTE would synthesize rows and break the group structure, re-introducing leakage.

---

## Feature Engineering

Instead of throwing raw counts at a model, the pipeline builds **causally available signals** grouped into six families:

| Family | Examples | Why it matters |
|---|---|---|
| Engagement ratios | `like_rate`, `comment_rate`, `engagement_rate`, `comment_like_ratio` | Scale-invariant; a 10K-sub channel with 5% like rate is more interesting than a 100M-sub channel with 0.1%. |
| Velocity | `likes_per_day`, `comments_per_day`, `engagement_per_day` | Cumulative views lie; daily velocity tells you whether momentum is building. |
| Channel authority | `subscriber_count_log`, `views_to_subs_ratio`, `channel_tier` | Virality is outperformance relative to the channel’s baseline. |
| Content format | `duration_log`, `is_short_video` | Shorts and full music videos have incompatible viral mechanics. |
| Upload-time signals | `publish_hour`, `publish_is_prime_time`, `publish_month` | Release strategy correlates with initial reach; safe because fixed at upload. |
| Title signals | `title_has_viral_keyword`, `title_has_hashtag`, `title_caps_ratio` | Marketing intent visible before any views accumulate; strong signal for the pre-viral slice. |
| Momentum / trajectory | `views_growth_rate`, `views_acceleration`, `days_since_first_snapshot`, `views_ratio_to_first` | Uses only *prior* snapshots of the same video; defaults safely for single-snapshot rows. |

Total engineered signals: **41 for detection, 43 for prediction**.

---

## Modeling

Both tasks are framed as binary classification with heavy class imbalance. Candidate models are trained on the training split and ranked by validation PR-AUC:

- Dummy baseline
- Random Forest
- Logistic Regression
- XGBoost

The selected model is XGBoost for both tasks. Thresholds are then optimized on validation F1, saved as artifacts, and used at inference instead of the default 0.5.

---

## Results

All numbers are from the untouched test set after leakage-aware, stratified splitting.

### Detection model

| Metric | Value |
|---|---|
| Validation PR-AUC | 0.9948 |
| Test F1 @ tuned threshold (0.764) | **0.9602** |
| Test PR-AUC | **0.9945** |
| Test ROC-AUC | **0.9992** |
| Precision / Recall | 95.4% / 96.7% |
| Held-out test rows | 7,161 |
| Train/Val F1 gap | 0.038 |
| 5-fold CV F1 | 0.975 ± 0.002 |

### Prediction model

| Metric | Value |
|---|---|
| Validation PR-AUC | 0.9986 |
| Test F1 @ tuned threshold (0.711) | **0.9857** |
| Test PR-AUC | **0.9988** |
| Test ROC-AUC | **0.9996** |
| Precision / Recall | 99.8% / 97.4% |
| Held-out test rows | 2,745 |

### Pre-viral honest slice

The headline prediction score is misleading. In this data, **95.9%** of future-viral rows were already above the 10M threshold at the training snapshot. The honest early-warning task is predicting future virality *only for videos currently below the threshold*:

| Metric | Before upload-time features | After adding title + publish-time features |
|---|---|---|
| Rows in slice | 2,231 | 2,231 |
| Future-viral among them | 18 (0.8%) | 18 (0.8%) |
| F1 | 0.2564 | **0.4167** |
| PR-AUC | 0.2381 | **0.7226** |
| ROC-AUC | 0.9705 | **0.9888** |

This slice is the real early-warning metric. The improvement shows that upload-time signals genuinely help the model when current view count is no longer the answer.

---

## Interpretability

SHAP values are produced for both models. Top drivers from the current run:

- **Detection:** `subscriber_count_log`, `views_to_subs_ratio`, `age_bucket`, `likes_per_day_log`, `engagement_per_day`.
- **Prediction:** `view_count_log` (expected, since the label is future views), `likes_per_day_log`, `age_bucket`, `title_caps_ratio`, `publish_month`.

Bar plots, beeswarm plots, and importance CSVs are saved to `models/explainability/`.

---

## Serving & MLOps

The trained artifacts are consumed by two interfaces, both using the same `src/inference.feature_builder` so the feature schema can never drift:

1. **Streamlit dashboards** (`dashboard.py`, `dashboard_prediction.py`) — interactive scoring with optional title and publish-date metadata.
2. **FastAPI service** (`src/api/app.py`) — `/predict/detection` and `/predict/prediction` endpoints returning probability, threshold-based decision, and the list of features used.

Model and threshold artifacts are versioned as files (pkl) and loaded at serving time; there is no manual feature list hardcoded in the UI layer.

---

## Testing & Quality

The test suite runs on small synthetic datasets — fast, deterministic, and isolated from the live YouTube API and the raw dataset. Coverage is focused on the failure modes that would invalidate the results:

| Test file | What it protects |
|---|---|
| `tests/test_splits.py` | No `video_id` leakage; positive-rate stratification. |
| `tests/test_labeling.py` | Detection labels are same-row; future labels exclude the last snapshot. |
| `tests/test_feature_engineering.py` | Zero-safe ratios, bin edges, NaN handling for first snapshot, new title/publish features. |
| `tests/test_inference_feature_builder.py` | Inference columns match training columns exactly per model. |
| `tests/test_evaluation.py` | Threshold optimization behaves correctly. |
| `tests/test_youtube_client.py` | API client batches, retries, throttles — all mocked. |
| `tests/test_api.py` | FastAPI endpoints return correct decisions. |

```bash
make test
# or
pytest tests/ -v --cov=src --cov-report=term-missing
```

**Current status: 46 tests passing.**

---

## Limitations & What This System Is Not

A senior ML writeup should be clear about scope:

- **Short horizon only:** the future label spans roughly 6 days. This is not long-range virality forecasting.
- **Small, skewed dataset:** classes are imbalanced; the pre-viral slice has only 18 positives.
- **Snapshot system:** performance assumes regular snapshots of the same videos. Dashboards default single-snapshot momentum features to 0.
- **YouTube-specific:** features like duration thresholds and title keywords are tuned for music-video content.

Next steps in a production setting: longer collection window, channel-level historical baselines, A/B test on promotion decisions, and drift monitoring for `view_count` distributions.

---

## Docker (Recommended for Reproducibility)

The image uses `python:3.10-slim`, runs as a non-root `appuser`, and copies only the files needed at runtime — no `.venv`, `.git`, notebooks, or raw data.

```bash
# Build
docker build -t music-virality .

# Detection dashboard  -> http://localhost:8501
docker run -p 8501:8501 music-virality

# Prediction dashboard -> http://localhost:8502
docker run -p 8502:8501 music-virality \
  python -m streamlit run dashboard_prediction.py \
  --server.port=8501 --server.address=0.0.0.0 --server.headless=true

# FastAPI service      -> http://localhost:8000
docker run -p 8000:8000 music-virality \
  python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Docker Compose

```bash
# Detection dashboard
docker compose up dashboard

# Prediction dashboard
docker compose up dashboard-pred

# FastAPI
docker compose up api

# Run the full training pipeline (one-off)
docker compose run --rm pipeline
```

| Service | Port | Compose command |
|---|---|---|
| Detection dashboard | 8501 | `docker compose up dashboard` |
| Prediction dashboard | 8502 | `docker compose up dashboard-pred` |
| FastAPI | 8000 | `docker compose up api` |

---

## Setup

```bash
# Runtime dependencies only
make install
# or: python -m pip install -r requirements.txt

# Runtime + development dependencies (tests, linting, formatting)
make install-dev
# or: python -m pip install -r requirements.txt -r requirements-dev.txt

# Configure API key (only needed for data collection)
cp .env.example .env
# edit .env with your YOUTUBE_API_KEY
```

## Usage

```bash
# Full pipeline
python main.py

# Detection dashboard
python -m streamlit run dashboard.py

# Prediction dashboard
python -m streamlit run dashboard_prediction.py

# FastAPI server
python -m uvicorn src.api.app:app --reload

# Collect new YouTube data
python -m src.pipelines.data_collection
```

---

## License

MIT License
