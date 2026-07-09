# Music Virality System

An end-to-end machine learning system for detecting and predicting music virality from YouTube snapshot data.

The project is built as a practical analytics stack, not a toy notebook:

- clean raw video observations into model-ready snapshots,
- engineer engagement, momentum, channel, and timing signals,
- train separate models for current virality detection and short-horizon future virality prediction,
- explain predictions with SHAP,
- and ship the result through Streamlit dashboards.

## What the project does

This system answers two different questions:

1. Is this video already viral right now?
2. Will this video become viral at a later snapshot?

That split matters. Detection and prediction are separate tasks, so the repository keeps separate labeling logic, feature sets, training pipelines, and evaluation paths for each one.

## Why this project is strong

The work is designed to demonstrate the kind of reasoning expected in production ML roles:

- group-aware splitting by `video_id` so the same video never leaks across train, validation, and test sets,
- stratified group splitting that preserves the positive-class rate across all three splits (the dataset is skewed, so a plain group split can starve a split of positives),
- feature selection that removes obvious target shortcuts like raw `view_count_log` from detection,
- holdout-aware evaluation so model selection and threshold tuning happen on validation data, not the final test set,
- a pre-viral honest evaluation slice that scores the prediction model only on videos not yet above threshold,
- explainability with SHAP so the model can be inspected rather than treated as a black box,
- and an explicit audit trail for the dataset limitations.

That is the level of discipline employers usually look for in a portfolio project for stronger ML roles.

## Results

The current leak-fixed, stratified run produced the following honest held-out test performance. Every split is group-aware (no video appears in more than one split) and stratified (the positive rate is preserved across train/val/test).

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

The full prediction score above is inflated: ~96% of "future viral" videos were already above the 10M threshold at the training snapshot, so the model mostly learns whether an already-big video stays big. The real question for an early-warning system is whether it can flag a video that is not yet viral.

To measure that honestly, the pipeline additionally evaluates the prediction model only on test rows whose current `view_count` is below the threshold:

- Rows in slice: 2,231
- Future-viral among them: 18 (0.8%)
- F1: 0.2564
- PR-AUC: 0.2381
- ROC-AUC: 0.9705

This is the true early-warning difficulty. The full-set 0.99 PR-AUC is not a generic virality-forecasting breakthrough — it is largely the model recognizing already-large videos. The pre-viral slice is where future model work has room to improve.

## Why the scores are so high

These numbers are strong, but they are not magic. There are two reasons the metrics look exceptionally high:

### 1. The leakage issues were fixed

Earlier versions of the project had evaluation bias from using the test set too early. That is now addressed by:

- splitting train, validation, and test sets by `video_id`,
- tuning hyperparameters and thresholds on validation only,
- and touching the final test set once, at the very end.

That is why the current scores are more trustworthy than the older ones.

### 2. The dataset is intrinsically favorable to prediction

The future-virality target is based on a short collection window of roughly 6 days. In practice, that means many of the rows that eventually become positive are already trending toward the threshold by the time the model sees them.

In the current run, 95.9% of the videos labeled as future viral were already above the virality threshold at the training snapshot.

That does not mean the model is wrong. It means the dataset is asking a somewhat easier question than true long-range early virality forecasting. The model is learning momentum, stability, and channel context over a short horizon, not predicting a song months before it breaks out.

### 3. The pre-viral slice proves where the difficulty really is

Because of (2), the full-set prediction score (PR-AUC 0.99) is not a fair measure of early-warning ability. When the model is evaluated only on videos **not yet** above threshold, PR-AUC drops to ~0.24 and F1 to ~0.26. That gap between the full score and the pre-viral slice is the single most important number in the project: it separates "the data was easy" from "the model is genuinely good at forecasting".

## Interpreting the project honestly

This is the right way to describe the system in interviews and on a resume:

- It is a real ML pipeline with clean separation between cleaning, feature engineering, labeling, training, evaluation, and explainability.
- The detection model is a strong snapshot classifier over current virality.
- The prediction model is a short-horizon forecast under a limited observation window.
- The unusually high scores are partly due to a genuinely strong signal and partly due to the dataset design, especially the short collection horizon.

That honesty is important. A strong candidate does not oversell leakage-fixed scores as if they were a generic breakthrough. They explain the data, the constraints, and the tradeoffs clearly.

## Project structure

- `data/`: raw and processed datasets
- `models/`: trained artifacts and explainability outputs
- `src/features/`: feature engineering and labeling logic
- `src/training/`: model training and evaluation pipelines
- `src/data_processing/`: dataset preparation and future-label generation
- `src/inference/`: single-source-of-truth feature builder shared by dashboards and the API
- `src/utils/`: evaluation, stratified splits, logging, and SHAP helpers
- `src/api/`: FastAPI backend scaffold and YouTube data client
- `dashboard.py`: detection dashboard
- `dashboard_prediction.py`: prediction dashboard
- `main.py`: pipeline orchestrator

## Key pipeline stages

1. Clean raw YouTube data.
2. Engineer engagement, momentum, channel authority, age, and format features.
3. Build two label sets: current virality and future virality.
4. Train grouped models with leakage-aware, stratified splits.
5. Evaluate on held-out data once, plus a pre-viral honest slice for the prediction model.
6. Save artifacts and inspect explanations.
7. Serve the results in Streamlit dashboards, with a FastAPI scaffold available for extension.

## Monitoring & retraining strategy

A model trained on a 6-day snapshot window is not a deploy-and-forget system. It
will drift, and the project is explicit about the three drift modes that matter:

- **Concept drift from the virality threshold itself.** The 10M-view cutoff is a
  moving target. Inflation in view counts, platform growth, and regional audience
  expansion mean "viral" in 2022 is not the same as "viral" in 2026. A fixed
  threshold slowly mislabels the tail of the distribution. The mitigation is to
  treat `VIRALITY_THRESHOLD` as a configurable parameter (it already is — see
  `src/features/labeling.py`) and re-derive it periodically from a fresh
  percentile of the active video distribution rather than hardcoding it.

- **Covariate drift from YouTube's product changes.** Algorithm changes (e.g.
  Shorts monetization, recommendation reshuffles, the comment-count display
  change) shift the distribution of `like_rate`, `comment_rate`, and velocity
  features even when the underlying "is this song popular" concept is stable.
  This is the most insidious drift because the model keeps scoring confidently on
  inputs that no longer mean what they meant at training time. The detection is
  feature-distribution monitoring: track PSI (population stability index) or KL
  divergence on the inference feature set against a training baseline, and alert
  when a feature crosses a threshold rather than waiting for labels to degrade.

- **Label lag.** Virality labels only become trustworthy weeks after a snapshot,
  so label-based monitoring (PR-AUC / F1 on fresh labeled data) always lags the
  feature drift. That is why feature-distribution monitoring is the leading
  indicator and label-based evaluation is the confirming one — never rely on
  label metrics alone for a system whose ground truth arrives late.

Operationally, the retraining cadence this implies is roughly: re-fit on a fresh
rolling window every 4–8 weeks, re-tune the decision threshold on the new
validation set (never carry the old threshold forward — the README's threshold
tuning is validation-only and per-run), and trigger an out-of-cycle retrain
whenever two or more monitored features breach their stability band. The
`pre-viral honest slice` evaluation should be re-run on every retrain because it
is the only slice that reflects genuine early-warning ability — a model can hold
its full-set PR-AUC while quietly losing the pre-viral signal that actually
matters.

## Installation

```bash
make install
```

On Windows, `make` may require Git Bash or WSL. If that is not available, run the commands directly:

```bash
python -m pip install -r requirements.txt
```

## Testing

The repo ships a real pytest suite (not a one-off notebook script). Tests run
against small synthetic datasets so they are fast, deterministic, and do not
touch the live YouTube API or the raw data files.

Coverage targets the parts of the codebase where a silent bug would invalidate
every reported metric — the leak-free split, the labeling logic, the
feature-engineering invariants, the evaluation utilities, and the
inference-time feature builder that both dashboards depend on.

```bash
make test
# or directly:
pytest tests/ -v --cov=src --cov-report=term-missing
```

What is covered:

- `tests/test_splits.py` — group-aware train/val/test split has **no video
  leakage** across splits, all rows are accounted for, positive-class rate is
  preserved (stratification), and the no-`video_id` fallback path works.
- `tests/test_labeling.py` — detection labels are a pure same-row function of
  `view_count` (no temporal leakage); future labels drop each video's last
  snapshot (no self-reference leakage) and derive the target from the final
  snapshot.
- `tests/test_feature_engineering.py` — ratio/velocity features are NaN-safe
  against zero views/likes/days; channel-tier, age-bucket, and short-video bins
  match the documented edges; momentum diffs are NaN on the first snapshot;
  `view_count_log` is produced for the prediction model.
- `tests/test_evaluation.py` — `evaluate_classifier` returns the expected
  metric keys and shape-consistent predictions; `find_best_threshold` stays in
  `[0,1]` and never underperforms the default 0.5 threshold; business-impact
  summary runs.
- `tests/test_inference_feature_builder.py` — the inference feature vector has
  exactly the columns both trained models expect, in order (no serving-time
  feature-name mismatch), and mirrors training's zero-safe arithmetic and bin
  edges.

## Usage

Run the full pipeline:

```bash
python main.py
```

Run the dashboard:

```bash
python -m streamlit run dashboard.py
```

Run the prediction dashboard:

```bash
python -m streamlit run dashboard_prediction.py
```

Run tests:

```bash
make test
```

## Configuration

Create a `.env` file for API keys and environment settings.

```env
YOUTUBE_API_KEY=your_key_here
```

## Notes for reviewers

- The final test scores are intentionally reported after leak fixes.
- Validation was used for model selection and threshold tuning; the test set is touched once.
- Splits are both group-aware (no video crosses splits) **and** stratified (positive rate preserved), which is what makes the skewed dataset's evaluation stable.
- The prediction model reports a **pre-viral honest slice** in addition to the full-test score — that slice is the real measure of early-warning ability.
- Class skew is handled via `class_weight="balanced"` / `scale_pos_weight` (not SMOTE, which would risk cross-video leakage on grouped data).
- SHAP explanations are part of the workflow, not an afterthought.
- Serving is centered on the Streamlit dashboards; the FastAPI scaffold is present for extension.
- The project is best described as a short-horizon virality modeling system with explicit dataset constraints.

## License

MIT License
