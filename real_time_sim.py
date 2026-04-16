import os
import numpy as np
import warnings
from collections import deque
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from Kheirkhahan.GSD3_test import KheirkhahanGSD
from Kheirkhahan.singleGSD_robust import load_segmented   

import pandas as pd
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


# ── Config (mirror your original script) ─────────────────────────────────────
DATA_PATH      = r"C:\Users\orlov\intern\gait_detection\QSense_data_clinic\sub1"
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
    activity_counts, _ = gsd.get_activity(seg_imu, sampling_rate_hz=SAMPLING_RATE)
    # print("activity_counts", activity_counts)

    y_window = np.zeros(len(seg_imu))
    if hasattr(bout_result, 'gs_list_') and not bout_result.gs_list_.empty:
        for _, row in bout_result.gs_list_.iterrows():
            y_window[int(row['start']):int(row['end'])] = 1

    return y_window

# def simulate_realtime(df: pd.DataFrame) -> np.ndarray:
#     """
#     Slide a 9-second window over df one second at a time.
 
#     Each tick the GSD result is written across all 9s of the window.
#     Positive detections (1) are sticky: a sample already labelled as
#     walking cannot be overwritten by a later window's zero.
#     Samples before the first full window get NaN.
#     """
#     n      = len(df)
#     print()
#     print("n is", n)
#     print()
#     y_pred = np.zeros(len(df))
#     buffer = deque(maxlen=WINDOW_SIZE)   # rolling sample index buffer
#     tick   = 0
#     # j = 0 
#     for sample_idx in range(n):
#         buffer.append(sample_idx) # contraints window indices 
 
#         # Only act at each 1-second boundary
#         if (sample_idx + 1) % STEP_SIZE != 0:
#             continue
 
#         # Wait until we have a full 9-second window
#         if len(buffer) < WINDOW_SIZE:
#             # if DEBUG:
#             #     print(f"  Tick {tick:>4d} | Waiting for 9s ({len(buffer)}/{WINDOW_SIZE} samples)")
#             continue
 
#         window_indices = list(buffer)             # exactly WINDOW_SIZE indices
#         window_df      = df.iloc[window_indices].copy()
 
#         # Run GSD on the full 9s window
#         y_window = run_gsd_on_window(window_df)   # shape: (WINDOW_SIZE,)
#         if  ~np.isnan(y_window).sum() == 0: 
#             print("predictions missing?")
        
#         # Write predictions across the whole window.
#         # Rule: only overwrite a sample with 0 if it hasn't been set to 1 yet.
#         for local_i, global_i in enumerate(window_indices):

#             new_label = y_window[local_i]
#             if np.isnan(y_pred[global_i]) or (y_pred[global_i] == 0):
#                 # First time this sample is seen — write unconditionally
#                 y_pred[global_i] = new_label
#                 if y_pred[global_i] == np.nan:
#                     print("global_i of", global_i)
#             # elif new_label == 1:
#             #     # Walking detected — overwrite any previous 0
#             #     y_pred[global_i] = 1
#             # else: new_label == 0 and existing label is already 1 → keep 1
#             # if j <5: 
#             #     print("local_i", local_i)
#             #     print("global_i", global_i)
#             #     j += 1 
#         # print("final global_i is", global_i)
#         n_walking = int(y_window.sum())
#         if n_walking != 0:
#             if n_walking != 450:
#                 print("n_walking", n_walking)
#         # if DEBUG:    
#         #     print(f"  Tick {tick:>4d} | window [{window_indices[0]:>5d}–{window_indices[-1]:>5d}] "
#         #           f"→ {n_walking}/{WINDOW_SIZE} walking samples detected")
#         tick += 1 
#     # valid = ~np.isnan(y_pred)
#     # print("valid:")
#     # print(valid)
#     # print(f"valid present {valid.sum()} \n")
#     # print(f"y predict is :")
#     # print(y_pred)
#     print("sample_idx", sample_idx)
#     return y_pred


BUFFER_SIZE = 13 * SAMPLING_RATE   # 2s padding each side + 9s window
TRUST_START = 2 * SAMPLING_RATE    # skip first 2s
TRUST_END   = 11 * SAMPLING_RATE   # skip last 2s

def simulate_realtime(df):
    n = len(df)
    y_pred = np.full(n, np.nan)
    buffer = deque(maxlen=BUFFER_SIZE)

    for sample_idx in range(n):
        buffer.append(sample_idx)

        if (sample_idx + 1) % STEP_SIZE != 0:
            continue
        if len(buffer) < BUFFER_SIZE:
            continue

        window_indices = list(buffer)
        window_df = df.iloc[window_indices].copy()
        y_window = run_gsd_on_window(window_df)   # length = BUFFER_SIZE

        # Only trust the middle portion — skip the 2s edges
        for local_i in range(TRUST_START, TRUST_END):
            global_i = window_indices[local_i]
            if np.isnan(y_pred[global_i]) or y_pred[global_i]==0:
                y_pred[global_i] = y_window[local_i]

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
    print(f"  Evaluated samples : {n_valid}  (skipped {n_skipped} due to the shift step)")
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

    return acc, prec, rec, f1

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
    
def process_realtime(rw_merged: pd.DataFrame, lw_merged: pd.DataFrame,
                 fl_merged: pd.DataFrame, 
                 save_results: bool = True, print_stats: bool = False) -> pd.DataFrame:
    """
    Run GSD on every (subject, wrist) segment inside the merged DataFrames.
    Prints a per-file table and condition/wrist averages, saves HickeyGSD_Results.csv.
    """
    results = []

    print(f"\n{'Subject':<35} | {'Wrist':<5} | {'Cond':<10} | "
          f"{'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 90)

    # ────────── Process the data per wrist ────────────────────────────────────── 
    for wrist_label, merged_df in [('RW', rw_merged), ('LW', lw_merged), ('',fl_merged)]:
        if merged_df.empty:
            print(f"[{wrist_label}] No data — skipping.")
            continue

        # Process each recording (subject folder) separately so the GSD sees
        # one continuous, coherent signal — not a mix of activities concatenated.
        for subject, grp_sub in merged_df.groupby('subject', sort=True):
            grp_sub = grp_sub.reset_index(drop=True)
            y_true    = grp_sub['y_true'].to_numpy()
            condition = grp_sub['condition'].iloc[0]
            label     = f"{subject}/{wrist_label}"

            y_pred = np.zeros(len(grp_sub))
            skipped_seg = 0

            for seg, grp_seg in grp_sub.groupby('segment', sort=True): 
                    if len(grp_seg) < WINDOW_SIZE:
                        y_pred[grp_seg.index] = np.nan
                        # y_true[grp_seg.index] = np.nan
                        skipped_seg += 1
                        continue
                    seg_pred = simulate_realtime(grp_seg.reset_index(drop=True))
                    y_pred[grp_seg.index] = seg_pred

            valid_mask = ~np.isnan(y_pred)
            acc  = accuracy_score(y_true[valid_mask], y_pred[valid_mask])
            prec = precision_score(y_true[valid_mask], y_pred[valid_mask], zero_division=0)
            rec  = recall_score(y_true[valid_mask], y_pred[valid_mask], zero_division=0)
            f1   = f1_score(y_true[valid_mask], y_pred[valid_mask], zero_division=0)

            results.append({
                'Subject':   label,
                'Wrist':     wrist_label,
                'Folder':    subject,
                'Condition': condition,
                'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1, 
                'TP': np.sum((y_pred == 1) & (y_true == 1)), 
                'FP': np.sum((y_pred == 1) & (y_true == 0)), 
                'FN': np.sum((y_pred == 0) & (y_true == 1)), 
                'TN': np.sum((y_pred == 0) & (y_true == 0))
            })

            if print_stats == True: 
                print(f"{label[:35]:<35} | {wrist_label:<5} | {condition:<10} | "
                      f"{acc:.2f}   | {prec:.2f}   | "
                      f"{rec:.2f}   | {f1:.2f}")    
    
    if not results:
        print("No results to summarise.")
        return pd.DataFrame()

    res_df = pd.DataFrame(results)

    VARIABLES = ['TP', 'FP', 'FN', 'TN']

    def _avg_row(row_type: str, label: str,
                 wrist: str, condition: str,
                 subset: pd.DataFrame) -> dict:
        """Build a single summary dict from a subset of res_df."""
        tp = subset['TP'].sum()
        fp = subset['FP'].sum()
        fn = subset['FN'].sum()
        tn = subset['TN'].sum()
        total = tp + fp + fn + tn

        accuracy_av  = (tp + tn) / total                          if total > 0             else 0.0
        precision_av = tp / (tp + fp)                             if (tp + fp) > 0         else 0.0
        recall_av    = tp / (tp + fn)                             if (tp + fn) > 0         else 0.0
        f1_av        = 2 * precision_av * recall_av / (precision_av + recall_av) \
                                                              if (precision_av + recall_av) > 0 else 0.0

        return {
            'row_type':  row_type,
            'Subject':   label,
            'Wrist':     wrist,
            'Folder':    '',
            'Condition': condition,
            'Accuracy':     accuracy_av,
            'Precision':    precision_av,
            'Recall':       recall_av,
            'F1':           f1_av,
            **{p: round(subset[p].sum(), 4) for p in VARIABLES}
        }

    def _print_avg(label: str, subset: pd.DataFrame):
        if subset.empty:
            return
        tp = subset['TP'].sum()
        fp = subset['FP'].sum()
        fn = subset['FN'].sum()
        tn = subset['TN'].sum()
        total = tp + fp + fn + tn

        accuracy_av  = (tp + tn) / total                          if total > 0             else 0.0
        precision_av = tp / (tp + fp)                             if (tp + fp) > 0         else 0.0
        recall_av    = tp / (tp + fn)                             if (tp + fn) > 0         else 0.0
        f1_av        = 2 * precision_av * recall_av / (precision_av + recall_av) \
                                                              if (precision_av + recall_av) > 0 else 0.0
        print(f"{label:<35} | {'':5} | {'':10} | "
              f"{accuracy_av:.5f}   | "
              f"{precision_av:.5f}   | "
              f"{recall_av:.5f}   | "
              f"{f1_av:.5f}")

    # Collect average rows for CSV
    avg_rows: list[dict] = []

    # Per-wrist averages
    for wrist in ['RW', 'LW']:
        sub = res_df[res_df['Wrist'] == wrist]
        if not sub.empty:
            avg_rows.append(_avg_row('avg_wrist',
                                     f"{wrist} average",
                                     wrist, '', sub))

    # Per-condition averages (all wrists combined)
    for condition in sorted(res_df['Condition'].unique()):
        sub = res_df[res_df['Condition'] == condition]
        avg_rows.append(_avg_row('avg_condition',
                                  f"Cond={condition} average",
                                  '', condition, sub))

    # Overall average
    avg_rows.append(_avg_row('avg_overall', 'AVERAGE (Overall)', '', '', res_df))

    # Print summary to console
    if print_stats: 
        print("-" * 100)
        _print_avg("AVERAGE (RW  Right Wrist)", res_df[res_df['Wrist'] == 'RW'])
        _print_avg("AVERAGE (LW  Left Wrist)",  res_df[res_df['Wrist'] == 'LW'])
        print()
    for condition in sorted(res_df['Condition'].unique()):
        _print_avg(f"AV(cond={condition})", res_df[res_df['Condition'] == condition])
    print("-" * 100)
    _print_avg("AVERAGE (Overall)", res_df)

    # Save the csv
    res_df.insert(0, 'row_type', 'result')

    avg_df    = pd.DataFrame(avg_rows)
    blank_row = pd.DataFrame([{c: '' for c in res_df.columns}])

    csv_df = pd.concat(
        [res_df,
         blank_row,
         avg_df],
        ignore_index=True
    )

    if save_results == True:
        csv_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved → {OUTPUT_FILE}")

    return res_df


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # 1. Load data (reuse existing loader)
    print(f"Loading {FILE_NAME} …")
    df = load_segmented(DATA_PATH, FILE_NAME, debug=DEBUG)

    # 2. Build acc columns (×9.8 as in original)
    acc_cols = [c for c in df.columns if 'acc' in c.lower()]
    df[acc_cols] = df[acc_cols].astype(float) * 9.8

    # 3. Ground truth
    if 'Label' in df.columns:
        y_true = df['Label'].astype(int).to_numpy()
    elif 'walk' in str(DATA_PATH).lower():
        y_true = np.ones(len(df), dtype=int) 
    else:
        y_true = np.zeros(len(df), dtype=int) 

    # 4. Simulate real-time processing
    y_pred = np.zeros(len(df))
    # print(f"made y pred of {len(y_pred)}")
    skipped_seg = 0 # might not to necessary 
    print(f"\nStarting real-time simulation  "
          f"(window={WINDOW_SIZE} samples, step={STEP_SIZE} samples) …\n")
    for seg, grp_seg in df.groupby('segment', sort=True): 
                    if len(grp_seg) < WINDOW_SIZE:
                        if DEBUG:
                            print(f"Segment {seg} is too short at {len(grp_seg)} samples.")
                        y_pred[grp_seg.index] = np.nan
                        # y_true[grp_seg.index] = np.nan
                        skipped_seg += 1
                        continue
                    seg_pred = simulate_realtime(grp_seg.reset_index(drop=True))
                    y_pred[grp_seg.index] = seg_pred
    
    if DEBUG: 
        print(f"Found {len(y_pred)} predictions")
        print(f"with {len(y_true)} annotations")
        valid = ~np.isnan(y_pred)
        print(f"valid present full {valid.sum()}")



    # 5. Print metrics
    print_metrics(y_true, y_pred, FILE_NAME)

    plot_results(df=df, y_pred=y_pred, y_true=y_true, title="Real-time simulation")
    plt.show()