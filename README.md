# Music Virality System

An end-to-end machine learning project for detecting current music virality and forecasting short-horizon virality from YouTube snapshot data.

This repository is intentionally built like a small production ML system, not a notebook demo. It covers data cleaning, feature engineering, label construction, grouped model training, honest evaluation, SHAP-based explanation, and Streamlit-based review surfaces.

## Executive Summary

The project answers two separate questions:

1. Is this video already viral at the current snapshot?
2. Will this video become viral at a later snapshot?

That separation matters. Detection and prediction are different targets, so the repository keeps separate labels, feature sets, training pipelines, and evaluation logic for each task.

The dataset is sensitive in the ML sense: it is small, skewed, and only spans a short collection window. That makes leakage control, honest validation, and careful wording more important than raw headline metrics.

## Why This Project Is Credible

The design choices reflect the constraints of a real ML problem rather than a benchmark exercise:

- Group-aware splitting by `video_id` prevents the same video from appearing in multiple splits.
- Stratified splitting preserves the positive rate across train, validation, and test, which matters on a skewed dataset.
- Detection features exclude obvious shortcuts such as raw `view_count_log`.
- Model selection and threshold tuning happen on validation, not on the final test set.
- The prediction task includes a pre-viral honest slice, which is the only part that reflects real early-warning difficulty.
- SHAP is used for interpretability, so the model can be inspected instead of treated as a black box.
- The README documents the dataset limitations instead of hiding them behind a single number.

## Results

The current leak-fixed run uses group-aware, stratified splits. Every split is disjoint by `video_id`, and the positive rate stays stable across train, validation, and test.

### Detection model

- Model: XGBoost
- Selected on validation PR-AUC: 0.9947
- Threshold tuned on validation: 0.841
- Test F1 at tuned threshold: 0.9591
- Test PR-AUC: 0.9948
- Test ROC-AUC: 0.9993
- Precision: 96.4%
- Recall: 95.4%
- Held-out test rows: 7,161
- Stability check: train/validation F1 gap of 0.030, with 5-fold CV of 0.974 ± 0.003
- Stratified split positive rates: train 10.8% | val 10.6% | test 11.3%

### Prediction model

- Model: XGBoost
- Selected on validation PR-AUC: 0.9945
- Threshold tuned on validation: 0.715
- Test F1 at tuned threshold: 0.9532
- Test PR-AUC: 0.9925
- Test ROC-AUC: 0.9979
- Precision: 96.9%
- Recall: 93.8%
- Held-out test rows: 2,745
- Stratified split positive rates: train 18.9% | val 19.8% | test 19.4%

### Pre-viral honest slice

The full prediction score is not the right headline for an early-warning problem. In this dataset, most future-viral rows are already large by the training snapshot, so the model is partly learning whether an already-prominent video stays prominent.

To measure the harder problem honestly, the prediction model is also evaluated only on test rows whose current `view_count` is still below the virality threshold:

- Rows in slice: 2,231
- Future-viral among them: 18 (0.8%)
- F1: 0.2564
- PR-AUC: 0.2381
- ROC-AUC: 0.9705

That slice is the real early-warning signal. It shows the project is not pretending a short-horizon, small-data setup is the same thing as true long-range virality forecasting.

## What Makes the Metrics High

The scores are strong for two reasons, and the README should be honest about both.

First, earlier leakage issues were removed. The current pipeline splits by `video_id`, tunes on validation, and evaluates the test set once at the end. That makes the reported numbers materially more trustworthy than older versions.

Second, the dataset is naturally favorable to short-horizon prediction. The target is derived from a short collection window of roughly 6 days, so many videos that end up positive are already trending toward the threshold by the time the model sees them. In the current run, 95.9% of the videos labeled as future viral were already above the virality threshold at the training snapshot.

The result is a useful system, but not a magical one. It is a short-horizon classifier/forecaster with strong signal and explicit limits.

## Data And Modeling Constraints

This project should be described carefully because the data is limited and the use case is sensitive:

- The label horizon is short.
- The class distribution is skewed.
- Group leakage would invalidate the evaluation.
- The model should be read as a snapshot system, not a long-range universal virality predictor.

That framing is deliberate. In a small-data setting, a senior ML writeup should emphasize what the model can support, what it cannot support, and which numbers are trustworthy.

## Pipeline

1. Clean raw YouTube observations.
2. Engineer engagement, momentum, channel, age, and format features.
3. Build current-virality and future-virality labels.
4. Train grouped models with leakage-aware, stratified splits.
5. Select the model and threshold on validation only.
6. Evaluate on the untouched test set and on the pre-viral honest slice.
7. Save artifacts, inspect SHAP explanations, and surface the results in dashboards.

## Repository Layout

- `data/`: raw and processed datasets
- `models/`: trained artifacts and explainability outputs
- `src/features/`: feature engineering and labeling logic
- `src/training/`: model training and evaluation pipelines
- `src/data_processing/`: dataset preparation and future-label generation
- `src/inference/`: shared feature builder used by dashboards and API code
- `src/utils/`: evaluation, stratified splits, logging, and SHAP helpers
- `src/api/`: FastAPI scaffold and YouTube data client
- `dashboard.py`: detection dashboard
- `dashboard_prediction.py`: prediction dashboard
- `main.py`: pipeline orchestrator

## Testing

The repo includes a real pytest suite that runs on synthetic data, so tests are fast, deterministic, and isolated from the live YouTube API and the raw dataset.

Coverage focuses on failure modes that would invalidate the reported results:

- `tests/test_splits.py` checks that train/validation/test splits are group-aware, stratified, and leakage-free.
- `tests/test_labeling.py` checks that detection labels are same-row labels and that future labels do not use the last snapshot as its own target.
- `tests/test_feature_engineering.py` checks zero-safe ratio and velocity features, bin edges, momentum features, and prediction features such as `view_count_log`.
- `tests/test_evaluation.py` checks the evaluation utilities, threshold search, and business-impact summary.
- `tests/test_inference_feature_builder.py` checks that inference-time features match the training-time schema exactly.

Run the test suite:

```bash
make test
# or:
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Installation

```bash
make install
```

On Windows, `make` may require Git Bash or WSL. If that is not available, install the dependencies directly:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run the full pipeline:

```bash
python main.py
```

Run the detection dashboard:

```bash
python -m streamlit run dashboard.py
```

Run the prediction dashboard:

```bash
python -m streamlit run dashboard_prediction.py
```

## Configuration

Create a `.env` file for API keys and environment settings:

```env
YOUTUBE_API_KEY=your_key_here
```

## Notes For Reviewers

- Final test scores are reported after leakage fixes.
- Validation is used for model selection and threshold tuning.
- The prediction model is evaluated both on the full test set and on the pre-viral honest slice.
- Class skew is handled with `class_weight="balanced"` / `scale_pos_weight`, not SMOTE, to avoid grouped leakage issues.
- SHAP explanations are part of the workflow, not an afterthought.
- Serving is centered on Streamlit dashboards; the FastAPI scaffold is present for extension.

## License

MIT License
