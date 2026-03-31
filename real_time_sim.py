import os
import numpy as np
import warnings
from collections import deque
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from GSD3_test import KheirkhahanGSD
from MM_own_all_robust import load_segmented   # reuse your existing loader

import pandas as pd
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


# ── Config (mirror your original script) ─────────────────────────────────────
DATA_PATH      = r"C:\Users\orlov\intern\gait_detection\QSense_data_clinic\sub4"
FILE_NAME      = "s1_1RW.txt"
SAMPLING_RATE  = 50
WINDOW_SIZE    = 9 *SAMPLING_RATE   # 450 samples  — full buffer
STEP_SIZE      = 1 * SAMPLING_RATE   # 50 samples   — shift per tick
THRESHOLD_STILL = 0.1
DEBUG          = True
# ─────────────────────────────────────────────────────────────────────────────


def run_gsd_on_window(window_df: pd.DataFrame) -> np.ndarray:
    """
    Run KheirkhahanGSD on a 9-second window DataFrame.
    Returns a per-sample binary prediction array of length len(window_df).
    Only the last STEP_SIZE samples are used by the caller.
    """
    acc_cols = [c for c in window_df.columns if 'acc' in c.lower()]
    seg_imu  = window_df[acc_cols].copy().astype(float)
    seg_imu.columns = ['acc_is', 'acc_ml', 'acc_pa']
    seg_imu  = seg_imu.reset_index(drop=True)

    gsd = KheirkhahanGSD(threshold_still=THRESHOLD_STILL)
    bout_result = gsd.detect(seg_imu, sampling_rate_hz=SAMPLING_RATE)

    y_window = np.zeros(len(seg_imu))
    if hasattr(bout_result, 'gs_list_') and not bout_result.gs_list_.empty:
        for _, row in bout_result.gs_list_.iterrows():
            y_window[int(row['start']):int(row['end'])] = 1

    return y_window


def simulate_realtime(df: pd.DataFrame) -> np.ndarray:
    """
    Slide a 9-second window over df one second at a time.
    For each position, only the label of the LAST 1 second is committed.
    Samples before the first full window get NaN.
    """
    n          = len(df)
    y_pred     = np.full(n, np.nan)
    buffer     = deque(maxlen=WINDOW_SIZE)   # rolling sample buffer (row indices)
    tick       = 0                           # counts 1-second ticks

    # We iterate sample by sample but only process at each 1-second boundary
    for sample_idx in range(n):
        buffer.append(sample_idx)

        # Only act at each 1-second boundary
        if (sample_idx + 1) % STEP_SIZE != 0:
            continue

        tick += 1

        # Wait until we have a full 9-second window
        if len(buffer) < WINDOW_SIZE:
            if DEBUG:
                print(f"  Tick {tick:>4d} | buffering… ({len(buffer)}/{WINDOW_SIZE} samples)")
            continue

        # Build window DataFrame (preserving original column names/values)
        window_indices = list(buffer)          # exactly WINDOW_SIZE indices
        window_df      = df.iloc[window_indices].copy()

        # Run GSD on the window
        y_window = run_gsd_on_window(window_df)

        # Commit only the LAST 1-second prediction
        last_step_indices = window_indices[-STEP_SIZE:]
        label             = 1 if y_window[-STEP_SIZE:].mean() >= 0.5 else 0
        y_pred[last_step_indices] = label

        if DEBUG:
            print(f"  Tick {tick:>4d} | window [{window_indices[0]:>5d}–{window_indices[-1]:>5d}] "
                  f"→ last-1s label = {label}")

    return y_pred


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, file_name: str) -> None:
    """Compute and print classification metrics, ignoring NaN predictions."""
    valid      = ~np.isnan(y_pred)
    yt         = (y_true[valid] == 1).astype(int)
    yp         = y_pred[valid].astype(int)

    acc  = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, zero_division=0)
    rec  = recall_score(yt, yp, zero_division=0)
    f1   = f1_score(yt, yp, zero_division=0)

    n_valid   = valid.sum()
    n_skipped = (~valid).sum()

    print("\n" + "=" * 60)
    print(f"Results for: {file_name}")
    print("=" * 60)
    print(f"  Evaluated samples : {n_valid}  (skipped {n_skipped} pre-buffer samples)")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall            : {rec:.4f}")
    print(f"  F1 Score          : {f1:.4f}")
    print("-" * 60)
    print(f"  True  Positives   : {int(np.sum((yp == 1) & (yt == 1)))}")
    print(f"  False Positives   : {int(np.sum((yp == 1) & (yt == 0)))}")
    print(f"  False Negatives   : {int(np.sum((yp == 0) & (yt == 1)))}")
    print(f"  True  Negatives   : {int(np.sum((yp == 0) & (yt == 0)))}")
    print("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load data (reuse existing loader)
    print(f"Loading {FILE_NAME} …")
    df = load_segmented(DATA_PATH, FILE_NAME)

    # 2. Build acc columns (×9.8 as in original)
    acc_cols = [c for c in df.columns if 'acc' in c.lower()]
    df[acc_cols] = df[acc_cols].astype(float) * 9.8

    # 3. Ground truth
    if 'Label' in df.columns:
        y_true = df['Label'].astype(int).to_numpy()
    else:
        y_true = np.ones(len(df), dtype=int) #### change this 

    # 4. Simulate real-time processing
    print(f"\nStarting real-time simulation  "
          f"(window={WINDOW_SIZE} samples, step={STEP_SIZE} samples) …\n")
    y_pred = simulate_realtime(df)

    # 5. Print metrics
    print_metrics(y_true, y_pred, FILE_NAME)