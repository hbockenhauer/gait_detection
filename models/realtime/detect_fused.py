import sys
import os
import numpy as np
import warnings
from collections import deque
from datetime import time
import csv

from models.realtime.GSD3 import KheirkhahanGSD

import pandas as pd
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


# ── Config ───────────────────────────────────────────────────────────────────
FILE_NAME_R = 's1_1RW.txt'
FILE_NAME_L = 's2_2LW.txt'

SAMPLING_RATE  = 50
WINDOW_SIZE    = 9 *SAMPLING_RATE   # 450 samples
STEP_SIZE      = 1 * SAMPLING_RATE   # 50 samples
THRESHOLD_STILL = 0.1
DEBUG          = True

BUFFER_SIZE = 13 * SAMPLING_RATE   # 2s padding each side + 9s window
TRUST_START = 2 * SAMPLING_RATE    # to skip first 2s
TRUST_END   = 11 * SAMPLING_RATE   # to skip last 2s
# ─────────────────────────────────────────────────────────────────────────────
def parse_time(t_str):
    h, m, s_ms = t_str.strip().split(':')
    s, ms = s_ms.split('.')
    return time(int(h), int(m), int(s), int(ms) * 1000)

def trim_to_multiple(group, factor=SAMPLING_RATE):
    n = len(group)
    trimmed = n - (n % factor)
    return group.iloc[:trimmed]

def load_segmented(data_path, file_name) -> pd.DataFrame:
    '''
    Load the txt file, file_name, located in the data_path directory. 
    Outputs df, pd.Dataframe, of the data. 
    The data is segmented into the parts of non-faulty data
    '''
    try:
        # open the file 
        filepath = os.path.join(data_path, file_name)
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)

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

        if DEBUG:
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

def sync_wrists(df_r: pd.DataFrame, df_l: pd.DataFrame) -> pd.DataFrame:
    """
    Outer-join the two DataFrames on HH:mm:ss.fff with a 20ms tolerance.
    Columns from the right wrist get suffix _r, left wrist get suffix _l.
    """
    # Sort both by timestamp (required by merge_asof)
    df_r = df_r.sort_values("HH:mm:ss.fff").reset_index(drop=True)
    df_l = df_l.sort_values("HH:mm:ss.fff").reset_index(drop=True)

    # Convert timestamp string → float seconds for merge_asof
    def ts_to_sec(col):
        return col.apply(lambda t: (
            int(t[0:2]) * 3600 + int(t[3:5]) * 60 +
            float(t[6:])
        ))

    df_r = df_r.copy()
    df_l = df_l.copy()
    df_r["_ts"] = ts_to_sec(df_r["HH:mm:ss.fff"])
    df_l["_ts"] = ts_to_sec(df_l["HH:mm:ss.fff"])

    merged = pd.merge_asof(
        df_r.sort_values("_ts"),
        df_l.sort_values("_ts"),
        on="_ts",
        direction="nearest",
        tolerance=0.020,          # 20 ms — one sample at 50 Hz
        suffixes=("_r", "_l"),
    )

    merged = merged.drop(columns=["_ts"], errors="ignore")

    return merged

def fuse_predictions(pred_r: np.ndarray, pred_l: np.ndarray) -> np.ndarray:
    """
    Element-wise fusion:
      - both numeric  → 1 if BOTH == 1, else 0  (AND logic)
      - one is NaN    → use the other directly
      - both NaN      → NaN
    """
    fused = np.full(len(pred_r), np.nan)
    for i in range(len(pred_r)):
        r, l = pred_r[i], pred_l[i]
        r_nan = np.isnan(r)
        l_nan = np.isnan(l)

        if r_nan and l_nan:
            fused[i] = np.nan
        elif r_nan:
            fused[i] = l
        elif l_nan:
            fused[i] = r
        else:
            fused[i] = 1.0 if (r == 1 and l == 1) else 0.0

    return fused


def detect_fused(data_path):
    # 1. Load both wrists
    df_r = load_segmented(data_path, FILE_NAME_R)
    df_l = load_segmented(data_path, FILE_NAME_L)

    # 2. Scale accelerometers
    for df in (df_r, df_l):
        acc_cols = [c for c in df.columns if "acc" in c.lower()]
        df[acc_cols] = df[acc_cols].astype(float) * 9.8

    # 3. Synchronise on timestamp → one aligned DataFrame
    df_sync = sync_wrists(df_r, df_l)

    # 4. Run GSD per segment on each wrist independently, then fuse
    n = len(df_sync)
    pred_r = np.full(n, np.nan)
    pred_l = np.full(n, np.nan)

    print(f"\nStarting fused real-time simulation "
          f"(window={WINDOW_SIZE}, step={STEP_SIZE}) …\n")

    # ── Right wrist: iterate over R segments ──────────────────────────────────
    for _, grp in df_sync.groupby("segment_r", sort=True):
        idx = grp.index
        acc_cols_r = [c for c in grp.columns if "acc" in c.lower() and c.endswith("_r")]
        if len(grp) < WINDOW_SIZE or len(acc_cols_r) != 3:
            pred_r[idx] = np.nan
            continue
        seg = grp[acc_cols_r].rename(columns={c: c[:-2] for c in acc_cols_r}).reset_index(drop=True)
        pred_r[idx] = simulate_realtime(seg)

    # ── Left wrist: iterate over L segments ───────────────────────────────────
    for _, grp in df_sync.groupby("segment_l", sort=True):
        idx = grp.index
        acc_cols_l = [c for c in grp.columns if "acc" in c.lower() and c.endswith("_l")]
        if len(grp) < WINDOW_SIZE or len(acc_cols_l) != 3:
            pred_l[idx] = np.nan
            continue
        seg = grp[acc_cols_l].rename(columns={c: c[:-2] for c in acc_cols_l}).reset_index(drop=True)
        pred_l[idx] = simulate_realtime(seg)

    # 5. Fuse
    y_fused = fuse_predictions(pred_r, pred_l)

    if DEBUG:
        n_both   = np.sum(~np.isnan(pred_r) & ~np.isnan(pred_l))
        n_r_only = np.sum(~np.isnan(pred_r) &  np.isnan(pred_l))
        n_l_only = np.sum( np.isnan(pred_r) & ~np.isnan(pred_l))
        n_none   = np.sum( np.isnan(pred_r) &  np.isnan(pred_l))
        print(f"Both wrists available : {n_both}")
        print(f"R only                : {n_r_only}")
        print(f"L only                : {n_l_only}")
        print(f"Neither               : {n_none}")

    return y_fused

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    process = psutil.Process(os.getpid())
    DATA_PATH = sys.argv[1]
    detect_fused(DATA_PATH)

