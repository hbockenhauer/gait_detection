import os
import numpy as np
import warnings
from collections import deque
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from GSD3_test import KheirkhahanGSD
from MM_own_all_robust import load_segmented   # reuse your existing loader

import pandas as pd
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


# ── Config (mirror your original script) ─────────────────────────────────────
DATA_PATH      = r"C:\Users\orlov\intern\gait_detection\QSense_data_clinic\sub4"
FILE_NAME_R    = "s1_1RW.txt"
FILE_NAME_L    = "s2_2LW.txt"
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
 
    Each tick the GSD result is written across all 9s of the window.
    Positive detections (1) are sticky: a sample already labelled as
    walking cannot be overwritten by a later window's zero.
    Samples before the first full window get NaN.
    """
    n      = len(df)
    y_pred = np.full(n, np.nan)
    buffer = deque(maxlen=WINDOW_SIZE)   # rolling sample index buffer
    tick   = 0
 
    for sample_idx in range(n):
        buffer.append(sample_idx)
 
        # Only act at each 1-second boundary
        if (sample_idx + 1) % STEP_SIZE != 0:
            continue
 
        tick += 1
 
        # Wait until we have a full 9-second window
        if len(buffer) < WINDOW_SIZE:
            if DEBUG:
                print(f"  Tick {tick:>4d} | Waiting for 9s ({len(buffer)}/{WINDOW_SIZE} samples)")
            continue
 
        window_indices = list(buffer)             # exactly WINDOW_SIZE indices
        window_df      = df.iloc[window_indices].copy()
 
        # Run GSD on the full 9s window
        y_window = run_gsd_on_window(window_df)   # shape: (WINDOW_SIZE,)
        # print()
        # print("y_window", y_window)
        # print()
 
        # Write predictions across the whole window.
        # Rule: only overwrite a sample with 0 if it hasn't been set to 1 yet.
        for local_i, global_i in enumerate(window_indices):
            # print("local_i", local_i)
            # print("global_i", global_i)
            new_label = y_window[local_i]
            if np.isnan(y_pred[global_i]) or (y_pred[global_i] == 0):
                # First time this sample is seen — write unconditionally
                y_pred[global_i] = new_label
            # elif new_label == 1:
            #     # Walking detected — overwrite any previous 0
            #     y_pred[global_i] = 1
            # else: new_label == 0 and existing label is already 1 → keep 1
 
        if DEBUG:
            n_walking = int(y_window.sum())
            print(f"  Tick {tick:>4d} | window [{window_indices[0]:>5d}–{window_indices[-1]:>5d}] "
                  f"→ {n_walking}/{WINDOW_SIZE} walking samples detected")
 
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

def plot_results(df: pd.DataFrame,
                 y_pred, y_true, title):
    # Parse timestamps from df into timedeltas
    time_series = pd.to_timedelta(df['HH:mm:ss.fff'].str.strip())
    # Convert to total seconds (float) for plotting
    time_per_second_sec = time_series.iloc[::SAMPLING_RATE].reset_index(drop=True).dt.total_seconds()

    total_seconds = len(time_per_second_sec)

    all_segment_first_rows = df.groupby('segment').nth(0).index
    all_segment_last_rows = df.groupby('segment').nth(-1).index
    jump_row_indices = all_segment_first_rows[1:]
    jump_times_sec = [time_series.iloc[idx].total_seconds() for idx in jump_row_indices]


    time_all_sec = time_series.dt.total_seconds() # seconds from midnight, accurate to 2 decimals
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # ── figure 1: raw data ────────────────────────────────────────────────────
    ax1.fill_between(time_all_sec, -1, 2, where=(y_true == 1),
                    alpha=0.2, color='green', transform=ax1.get_xaxis_transform(),
                    label='Ground truth (walking)')
    if 2 in y_true:
        ax1.fill_between(time_all_sec, -1, 2, where=(y_true == 2),
                        alpha=0.2, color='purple', transform=ax1.get_xaxis_transform(),
                        label='Functional Arm use')
    acc_cols = [c for c in df.columns if 'acc' in c]
    for acc in acc_cols:

        ax1.plot(time_all_sec, df[acc], label=acc, alpha=0.8, marker='.', linestyle='None', markersize=3)

    for i, jt in enumerate(jump_times_sec):
        ax1.axvline(x=jt, color='orange', linewidth=1.0, linestyle='--', alpha=0.8,
                    label='Time gap' if i == 0 else None)

    ax1.set_ylabel(f'Acceleration (m/s^{2})')
    ax1.set_title(f'{title}')
    ax1.legend(loc='upper left')

    # ── figure 5: y_pred and y_true ──────────────────────────────────────────────
    ax2.fill_between(time_all_sec, -1, 2, where=(y_true == 1),
                    alpha=0.2, color='green', label='Ground truth (walking)')
    if 2 in y_true:
        ax2.fill_between(time_all_sec, -1, 2, where=(y_true == 2),
                        alpha=0.2, color='purple', label='Functional Arm use')
    ax2.plot(time_all_sec, y_pred, label='y_pred (GSD)', alpha=0.8, 
            linewidth=1, color='steelblue')

    for i, jt in enumerate(jump_times_sec):
        ax2.axvline(x=jt, color='orange', linewidth=1.0, linestyle='--', alpha=0.8,
                    label='Time gap' if i == 0 else None)

    ax2.set_ylabel('Walking (1) / Not (0)')
    ax2.legend(loc='upper left')
    ax2.set_ylim(-0.1, 1.4)
    

    # Format x-axis as HH:MM:SS
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x//3600):02d}:{int((x%3600)//60):02d}:{int(x%60):02d}"
    ))
    fig.autofmt_xdate()
    plt.tight_layout()
    

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load data (reuse existing loader)
    print(f"Scanning {DATA_PATH} …")
    df_r = load_segmented(DATA_PATH, FILE_NAME_R)
    df_l = load_segmented(DATA_PATH, FILE_NAME_L)


    # 2. Build acc columns (×9.8 as in original)
    acc_cols = [c for c in df_r.columns if 'acc' in c.lower()]
    df_r[acc_cols] = df_r[acc_cols].astype(float) * 9.8

    # acc_cols = [c for c in df_r.columns if 'acc' in c.lower()]
    df_l[acc_cols] = df_l[acc_cols].astype(float) * 9.8

    # 3. Ground truth
    if 'Label' in df_r.columns:
        y_true = df_r['Label'].astype(int).to_numpy()
    else:
        y_true = np.ones(len(df_r), dtype=int) #### change this 

    # 4. Simulate real-time processing
    print(f"\nStarting real-time simulation  "
          f"(window={WINDOW_SIZE} samples, step={STEP_SIZE} samples) …\n")
    y_pred_r = simulate_realtime(df_r)
    y_pred_l = simulate_realtime(df_l)

    # 5. Print metrics
    print_metrics(y_true, y_pred_r, DATA_PATH)

    plot_results(df=df_r, y_pred=y_pred_r, y_true=y_true, title="Real-time simulation")
    plt.show()