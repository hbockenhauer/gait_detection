import sys
import os
import numpy as np
import warnings
from collections import deque
from datetime import time
import csv

from GSD3 import KheirkhahanGSD

import pandas as pd
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


# ── Config ───────────────────────────────────────────────────────────────────
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

def detect_per_wrist(data_path):

    # 1. Load data 
    folder_path, filename = os.path.split(data_path)
    df = load_segmented(folder_path, filename)

    # 2. Build acc columns (×9.8 as in original)
    acc_cols = [c for c in df.columns if 'acc' in c.lower()]
    df[acc_cols] = df[acc_cols].astype(float) * 9.8

    # 3. Simulate real-time processing
    y_pred = np.zeros(len(df))
    print(f"\nStarting real-time simulation  "
          f"(window={WINDOW_SIZE} samples, step={STEP_SIZE} samples) …\n")
    for _, grp_seg in df.groupby('segment', sort=True): 
                    if len(grp_seg) < WINDOW_SIZE:
                        y_pred[grp_seg.index] = np.nan
                        continue
                    seg_pred = simulate_realtime(grp_seg.reset_index(drop=True))
                    y_pred[grp_seg.index] = seg_pred
    if DEBUG: 
        print(f"Found {len(y_pred)} predictions")
    return y_pred
    
# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_PATH = sys.argv[1]
    detect_per_wrist(DATA_PATH)

