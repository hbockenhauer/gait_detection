import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import csv
import os 
from collections import deque
from datetime import time

# from detect_per_wrist import simulate_realtime, load_segmented, SAMPLING_RATE, WINDOW_SIZE
from GSD3_fused import KheirkhahanGSD
# from singleGSD_robust import load_segmented 

DATA_PATH      = r"C:\Users\orlov\intern\gait_detection\QSense_data_clinic\sub3"
# ── Config (mirror your original script) ─────────────────────────────────────
FILE_NAME_R = 's1_1RW.txt'
FILE_NAME_L = 's2_2LW.txt'

SAMPLING_RATE  = 50
WINDOW_SIZE    = 9 *SAMPLING_RATE   # 450 samples  — full buffer
STEP_SIZE      = 1 * SAMPLING_RATE   # 50 samples   — shift per tick
THRESHOLD_STILL = 0.1
DEBUG          = True

BUFFER_SIZE = 13 * SAMPLING_RATE   # 2s padding each side + 9s window
TRUST_START = 2 * SAMPLING_RATE    # skip first 2s
TRUST_END   = 11 * SAMPLING_RATE   # skip last 2s
# ─────────────────────────────────────────────────────────────────────────────

def parse_time(t_str):
    h, m, s_ms = t_str.strip().split(':')
    s, ms = s_ms.split('.')
    return time(int(h), int(m), int(s), int(ms) * 1000)

def trim_to_multiple(group, factor=SAMPLING_RATE):
    n = len(group)
    trimmed = n - (n % factor)
    return group.iloc[:trimmed]
    
def load_segmented(DATA_PATH, file_name, debug: bool = False) -> pd.DataFrame:
    try:
        # open the file 
        filepath = os.path.join(DATA_PATH, file_name)
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)
        # print(rows)
        # clip the first 10 seconds depending on the data path 
        # rows = rows if ("mixed" or "clinic") in str(DATA_PATH) else rows[500:]
        # if debug == True:
        #     print("Data taken fully.") if ("mixed" or "clinic") in str(DATA_PATH) else print("First 10s are clipped.")

        clean_rows = []
        segments   = []
        max_time   = None
        prev_time  = None
        segment_id = 0
        dropped_rows = 0

        for row in rows:
            try:
                t = parse_time(row['HH:mm:ss.fff'])
            except Exception:
                continue

            if max_time is None or t > max_time:
                if prev_time is not None:
                    # compute gap in ms (handles minute/hour rollover simply)
                    gap_ms = (t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
                              - prev_time.hour * 3600 - prev_time.minute * 60 - prev_time.second - prev_time.microsecond / 1e6) * 1000
                    if gap_ms > ((1001/SAMPLING_RATE)):
                        segment_id += 1
                max_time  = t
                prev_time = t           
                clean_rows.append(row)
                segments.append(segment_id)
            else:
                dropped_rows += 1
        df = pd.DataFrame(clean_rows)
        df = df.reset_index(drop=True)
        df['segment'] = segments
        df = df.set_index('segment')
        # clip the last samples to full seconds 
        df = df.groupby('segment', group_keys=False).apply(trim_to_multiple, include_groups=False)
        df = df.reset_index()

        if debug:
            print(f"Dropped {dropped_rows} rows.")
            print(f"Found {segment_id+1} segments. ")
            print(f"Kept {len(clean_rows)} rows. \n")

        # df.columns are now :
        # ['yyyy-MM-dd', 'HH:mm:ss.fff', 'gyrX', 'gyrY', 'gyrZ', 
        # 'accX', 'accY', 'accZ', 'magX', 'magY', 'magZ', 
        # 'Marker', 'Energy', 'Angle', 'Classification', 'Label', 'segment']
        
    except Exception as e:
        print(f"{file_name[:25]:<25} | ERROR: {str(e)}")
    
    return df

def run_gsd_on_window(window_df: pd.DataFrame) -> np.ndarray:

    acc_cols_r = [c for c in window_df.columns if "acc" in c.lower() and c.endswith("_r")]
    acc_cols_l = [c for c in window_df.columns if "acc" in c.lower() and c.endswith("_l")]

    has_r = len(acc_cols_r) == 3 and not window_df[acc_cols_r].isna().all().all()
    has_l = len(acc_cols_l) == 3 and not window_df[acc_cols_l].isna().all().all()

    gsd = KheirkhahanGSD(threshold_still=THRESHOLD_STILL)

    # CASE 1: both wrists
    if has_r and has_l:
        seg_r = window_df[acc_cols_r].rename(columns={c: c[:-2] for c in acc_cols_r})
        seg_l = window_df[acc_cols_l].rename(columns={c: c[:-2] for c in acc_cols_l})

        seg_r = seg_r.interpolate().bfill().ffill()
        seg_l = seg_l.interpolate().bfill().ffill()

        gs = gsd.detect(seg_r.reset_index(drop=True),
                        seg_l.reset_index(drop=True),
                        sampling_rate_hz=SAMPLING_RATE)

    # CASE 2: only right
    elif has_r:
        seg_r = window_df[acc_cols_r].rename(columns={c: c[:-2] for c in acc_cols_r})
        seg_r = seg_r.interpolate().fillna(method="bfill").fillna(method="ffill")

        gs = gsd.detect(seg_r.reset_index(drop=True), None,
                        sampling_rate_hz=SAMPLING_RATE)

    # CASE 3: only left
    elif has_l:
        seg_l = window_df[acc_cols_l].rename(columns={c: c[:-2] for c in acc_cols_l})
        seg_l = seg_l.interpolate().fillna(method="bfill").fillna(method="ffill")

        gs = gsd.detect(None, seg_l.reset_index(drop=True),
                        sampling_rate_hz=SAMPLING_RATE)

    else:
        return np.zeros(len(window_df))  # no data

    # Convert GS → binary signal
    y_window = np.zeros(len(window_df))
    if hasattr(gs, 'gs_list_') and not gs.gs_list_.empty:
        for _, row in gs.gs_list_.iterrows():
            y_window[int(row['start']):int(row['end'])] = 1

    return y_window

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
            if np.isnan(y_pred[global_i]):
                y_pred[global_i] = y_window[local_i]

    return y_pred

def print_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    valid      = ~np.isnan(y_pred)
    yt         = (y_true[valid] == 1).astype(int)
    yp         = y_pred[valid].astype(int)

    acc  = accuracy_score(yt, yp)
    prec = precision_score(yt, yp, zero_division=0)
    rec  = recall_score(yt, yp, zero_division=0)
    f1   = f1_score(yt, yp, zero_division=0)

    print("\n" + "=" * 60)
    print(f"Results for: {label}")
    print("=" * 60)
    print(f"  Evaluated samples : {valid.sum()}  (skipped {(~valid).sum()} NaN samples)")
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

    time_all_sec = time_series.dt.total_seconds() # seconds from midnight, accurate to 2 decimals
    
    # Segment gap markers
    jump_times_sec_r = []
    jump_times_sec_l = []
    for seg_id, grp in df_sync.groupby("segment_r", sort=True):
        first_idx = grp.index[0]
        if first_idx != df_sync.index[0]:
            jump_times_sec_r.append(time_all_sec.iloc[first_idx])

    for seg_id, grp in df_sync.groupby("segment_l", sort=True):
        first_idx = grp.index[0]
        if first_idx != df_sync.index[0]:
            jump_times_sec_l.append(time_all_sec.iloc[first_idx])

    def _add_truth_bands(ax):
        ax.fill_between(time_all_sec, -0.1, 1.4,
                        where=(y_true == 1), alpha=0.2, color="green",
                        transform=ax.get_xaxis_transform(), label="Ground truth (walking)")
        if 2 in y_true:
            ax.fill_between(time_all_sec, -0.1, 1.4,
                            where=(y_true == 2), alpha=0.2, color="purple",
                            transform=ax.get_xaxis_transform(), label="Functional arm use")

    def _add_gaps(ax, labeled=True):
        for i, jt in enumerate(jump_times_sec_r):
            ax.axvline(x=jt, color="orange", linewidth=1.0, linestyle="--", alpha=0.8,
                       label="Time gap right" if (labeled and i == 0) else None)
        for i, jt in enumerate(jump_times_sec_l):
            ax.axvline(x=jt, color="yellow", linewidth=1.0, linestyle="--", alpha=0.8,
                       label="Time gap left" if (labeled and i == 0) else None)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=13)
    # ── Panel 1: raw acc R ────────────────────────────────────────────────────
    ax = axes[0]
    _add_truth_bands(ax)
    for col in [c for c in df_sync.columns if "acc" in c.lower() and c.endswith("_r")]:
        ax.plot(time_all_sec, df_sync[col].astype(float),
                label=col, alpha=0.8, marker=".", linestyle="None", markersize=2)
    _add_gaps(ax)
    ax.set_ylabel("Acc R (m/s²)")
    ax.legend(loc="upper left", fontsize=7)

    # ── Panel 2: raw acc L ────────────────────────────────────────────────────
    ax = axes[1]
    _add_truth_bands(ax)
    for col in [c for c in df_sync.columns if "acc" in c.lower() and c.endswith("_l")]:
        ax.plot(time_all_sec, df_sync[col].astype(float),
                label=col, alpha=0.8, marker=".", linestyle="None", markersize=2)
    _add_gaps(ax, labeled=False)
    ax.set_ylabel("Acc L (m/s²)")
    ax.legend(loc="upper left", fontsize=7)

    # ── figure 5: y_pred and y_true ──────────────────────────────────────────────
    ax = axes[2]
    _add_truth_bands(ax)

    ax.plot(time_all_sec, y_pred, label='y_pred (GSD)', alpha=0.8, 
            linewidth=1, color='steelblue')

    _add_gaps(ax, labeled=False)

    ax.set_ylabel('Walking (1) / Not (0)')
    ax.legend(loc='upper left')
    ax.set_ylim(-0.1, 1.4)
    

    # Format x-axis as HH:MM:SS
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x//3600):02d}:{int((x%3600)//60):02d}:{int(x%60):02d}"
    ))
    fig.autofmt_xdate()
    plt.tight_layout()

def extract_true_labels(df_sync: pd.DataFrame, data_path: str) -> np.ndarray:
    """
    Prefer the Label column from R; fall back to L; fall back to path heuristic.
    After merge_asof some rows may have NaN on one side, so we combine_first
    across both suffix variants.
    """
    label_r = df_sync.get("Label_r")
    label_l = df_sync.get("Label_l")
    # Also handle case where merge produced a single un-suffixed Label column
    label_plain = df_sync.get("Label")

    combined = None
    for col in (label_r, label_plain, label_l):
        if col is None:
            continue
        s = pd.to_numeric(col, errors="coerce")
        combined = s if combined is None else combined.combine_first(s)

    if combined is not None and combined.notna().any():
        return combined.fillna(0).astype(int).to_numpy()

    # Fallback: infer from path
    if "walk" in str(data_path).lower():
        return np.ones(len(df_sync), dtype=int)
    return np.zeros(len(df_sync), dtype=int)

def sync_wrists(df_r: pd.DataFrame, df_l: pd.DataFrame) -> pd.DataFrame:
    cols = ['yyyy-MM-dd', 'HH:mm:ss.fff',
            'accX', 'accY', 'accZ',
            'Label', 'segment']

    df_r = df_r[cols].copy()
    df_l = df_l[cols].copy()

    # Rename columns (EXCEPT timestamp!)
    df_r = df_r.rename(columns={
        'accX': 'accX_r', 'accY': 'accY_r', 'accZ': 'accZ_r',
        'Label': 'Label_r', 'segment': 'segment_r',
        'yyyy-MM-dd': 'date_r'
    })

    df_l = df_l.rename(columns={
        'accX': 'accX_l', 'accY': 'accY_l', 'accZ': 'accZ_l',
        'Label': 'Label_l', 'segment': 'segment_l',
        'yyyy-MM-dd': 'date_l'
    })

    # outer merge on timestamp string
    df_sync = pd.merge(
        df_r,
        df_l,
        on='HH:mm:ss.fff',
        how='outer',
        sort=True
    )

    # Sort explicitly (string sort works because format is HH:mm:ss.fff)
    df_sync = df_sync.sort_values('HH:mm:ss.fff').reset_index(drop=True)

    # Combine date columns
    df_sync['yyyy-MM-dd'] = df_sync['date_r'].combine_first(df_sync['date_l'])

    # Final column order
    df_sync = df_sync[[
        'yyyy-MM-dd', 'HH:mm:ss.fff',
        'accX_r', 'accY_r', 'accZ_r',
        'accX_l', 'accY_l', 'accZ_l',
        'Label_r', 'Label_l',
        'segment_r', 'segment_l'
    ]]

    return df_sync

if __name__ == "__main__":
    # 1. Load and scale both wrists
    df_r = load_segmented(DATA_PATH, FILE_NAME_R, debug=DEBUG)
    df_l = load_segmented(DATA_PATH, FILE_NAME_L, debug=DEBUG)

    for df in (df_r, df_l):
        acc_cols = [c for c in df.columns if "acc" in c.lower()]
        df[acc_cols] = df[acc_cols].astype(float) * 9.8

    # 2. Synchronise

    df_sync = sync_wrists(df_r, df_l)
    # print(df_sync)

    # 3. True labels (combined from whichever wrist has them)
    y_true = extract_true_labels(df_sync, DATA_PATH)

    # 4. Run GSD per segment, per wrist
    n = len(df_sync)
    y_fused = np.full(n, np.nan)

    y_fused = simulate_realtime(df_sync)

    # 6. Metrics — fused, plus per-wrist for comparison
    # print_metrics(y_true, pred_r,  f"Right wrist only  ({FILE_NAME_R})")
    # print_metrics(y_true, pred_l,  f"Left wrist only   ({FILE_NAME_L})")
    print_metrics(y_true, y_fused, f"Fused             ({DATA_PATH})")
    plot_results(df_sync, y_fused, y_true, DATA_PATH)

    # # 7. Plot
    # plot_results_fused(
    #     df_sync=df_sync,
    #     pred_r=pred_r,
    #     pred_l=pred_l,
    #     y_fused=y_fused,
    #     y_true=y_true,
    #     title=f"Fused evaluation — {os.path.basename(data_path)}",
    # )
    plt.show()

