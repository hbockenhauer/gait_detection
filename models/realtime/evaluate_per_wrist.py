import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import sys
import os 

from models.realtime.detect_per_wrist import (
     simulate_realtime, load_segmented, 
     SAMPLING_RATE, WINDOW_SIZE)

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
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
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

def evaluate_per_wrist(datapath):
    # 1. load the data and build the acc columns 
    folder_path, filename = os.path.split(datapath)
    df = load_segmented(folder_path, filename)

    acc_cols = [c for c in df.columns if 'acc' in c.lower()]
    df[acc_cols] = df[acc_cols].astype(float) * 9.8

    # 2. Get the annotations 
    if 'Label' in df.columns:
        y_true = df['Label'].astype(int).to_numpy()
    elif 'walk' in str(folder_path).lower():
        y_true = np.ones(len(df), dtype=int) 
    else:
        y_true = np.zeros(len(df), dtype=int)

    # 3. Get the predictions 
    y_pred = np.zeros(len(df))
    print(f"\nStarting real-time simulation  "
          f"(window={WINDOW_SIZE} samples, step={SAMPLING_RATE} samples) …\n")
    for _, grp_seg in df.groupby('segment', sort=True): 
                    if len(grp_seg) < WINDOW_SIZE:
                        y_pred[grp_seg.index] = np.nan
                        continue
                    seg_pred = simulate_realtime(grp_seg.reset_index(drop=True))
                    y_pred[grp_seg.index] = seg_pred

    # 4. Print the metrics for evaluation as well as the plot 
    print_metrics(y_true, y_pred, datapath)

    plot_results(df=df, y_pred=y_pred, y_true=y_true, title="Real-time simulation")
    plt.show()

if __name__ == "__main__":
  DATA_PATH = sys.argv[1]
  evaluate_per_wrist(DATA_PATH)
  

