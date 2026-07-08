"""
Stratified group-aware train/val/test split.

WHY: the previous split used GroupShuffleSplit, which keeps videos disjoint but
does NOT preserve the positive-class rate across splits. With a skewed dataset
(~11-19% viral) a random group split can starve the validation/test split of
positives, making PR-AUC unstable and threshold tuning unreliable.

This helper splits at the VIDEO level (so no video leaks across splits) while
stratifying on a per-video label (1 if any snapshot of that video is positive),
so each split keeps roughly the same positive rate. That makes evaluation of a
skewed dataset far more stable and trustworthy.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def stratified_group_three_way_split(
    df, X, y, val_size=0.15, test_size=0.15, random_state=42,
):
    """
    Returns (x_train, x_val, x_test, y_train, y_val, y_test) with disjoint
    video_ids AND a preserved positive rate across all three splits.

    Falls back to a stratified random split if video_id is unavailable.
    """
    if "video_id" not in df.columns or df["video_id"].isna().all():
        print("[Warning] video_id not available. Falling back to stratified random split.")
        from sklearn.model_selection import train_test_split
        x_tr, x_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=(val_size + test_size),
            random_state=random_state, stratify=y,
        )
        x_va, x_te, y_va, y_te = train_test_split(
            x_tmp, y_tmp, test_size=test_size / (val_size + test_size),
            random_state=random_state, stratify=y_tmp,
        )
        return x_tr, x_va, x_te, y_tr, y_va, y_te

    groups = df["video_id"].astype(str).values
    y_arr = np.asarray(y)

    # One row per video; stratification label = 1 if ANY snapshot is positive.
    vid_df = pd.DataFrame({"vid": groups, "y": y_arr})
    vid_label = vid_df.groupby("vid")["y"].max()
    vids = vid_label.index.to_numpy()
    strat = vid_label.to_numpy()

    pos_rate = strat.mean()
    print(f"\n[Split] Overall positive rate (video-level): {pos_rate:.1%} "
          f"({int(strat.sum())}/{len(strat)} videos)")

    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(val_size + test_size), random_state=random_state
    )
    train_vid_idx, tmp_vid_idx = next(sss1.split(vids, strat))

    tmp_vids = vids[tmp_vid_idx]
    tmp_strat = strat[tmp_vid_idx]
    holdout_frac = test_size / (val_size + test_size)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=holdout_frac, random_state=random_state
    )
    val_vid_idx_local, test_vid_idx_local = next(sss2.split(tmp_vids, tmp_strat))

    train_vids = set(vids[train_vid_idx])
    val_vids = set(tmp_vids[val_vid_idx_local])
    test_vids = set(tmp_vids[test_vid_idx_local])

    g_series = pd.Series(groups)
    train_mask = g_series.isin(train_vids).to_numpy()
    val_mask = g_series.isin(val_vids).to_numpy()
    test_mask = g_series.isin(test_vids).to_numpy()

    def _rate(mask):
        return y_arr[mask].mean() if mask.sum() else 0.0

    print(f"[Split] Train videos: {len(train_vids):,} ({_rate(train_mask):.1%} pos) | "
          f"Val videos: {len(val_vids):,} ({_rate(val_mask):.1%} pos) | "
          f"Test videos: {len(test_vids):,} ({_rate(test_mask):.1%} pos)")
    print(f"[Split] Rows -> train: {int(train_mask.sum()):,} | "
          f"val: {int(val_mask.sum()):,} | test: {int(test_mask.sum()):,}")

    overlap = (train_vids & val_vids) | (train_vids & test_vids) | (val_vids & test_vids)
    if overlap:
        print("[WARN] Group leakage detected. Check video_id column.")
    else:
        print("[OK] No group leakage - stratified train/val/test split confirmed.")

    return (
        X.loc[train_mask], X.loc[val_mask], X.loc[test_mask],
        y.loc[train_mask], y.loc[val_mask], y.loc[test_mask],
    )
