import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from GSD3_test import KheirkhahanGSD
# from multimob.GSD.GSD3 import KheirkhahanGSD
# from multimob.GSD.GSD4 import MacLeanGSD
# from multimob.GSD.GSD5 import KerenGSD
# from GSD2a import HickeyGSD
import csv
from datetime import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

"""
ONLY for running QSense files 
"""

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

DATA_PATHS = [
    # r"C:\Users\orlov\intern\gait_detection\QSense_data_edge",
    # r"C:\Users\orlov\intern\gait_detection\QSense_data_mixed",
    # r"C:\Users\orlov\intern\gait_detection\QSense_data",
    r"C:\Users\orlov\intern\gait_detection\QSense_data_clinic"
]

SAMPLING_RATE = 50 
GAIT_CLASSES = {'walking', 'stairs'}
CONDITION_KEYWORDS = ['pockets', 'phone', 'rail', 'free', 'crutches', 'walker', 
                      'cane', 'limp', 'armfixed', 'stroke']
MIN_SEGMENT_SAMPLES = 9*SAMPLING_RATE 
THRESHOLD_STILL = 0.1

DEBUG = False
PRINT_STATS = True 

SAVE_RESULTS = False 
OUTPUT_FILE = "Results/KheirkhahanGSD_Results_wrist.csv"

PLOT = False
OUT_FOLDER = r"C:\Users\orlov\intern\gait_detection\Plots\Robust_Kheirkhahan\wHickey"

# ── Wrist functions ────────────────────────────────────────────────────
def _pred_series(grp_wrist: pd.DataFrame, y_pred: np.ndarray) -> pd.Series:
    """Return y_pred as a Series indexed by 'yyyy-MM-dd HH:mm:ss.fff'."""
    timestamps = (grp_wrist['yyyy-MM-dd'].str.strip()
                  + ' '
                  + grp_wrist['HH:mm:ss.fff'].str.strip())
    return pd.Series(y_pred, index=timestamps.values, name='y_pred')

def _true_series(grp_wrist: pd.DataFrame) -> pd.Series:
    """Return y_true as a Series indexed by 'yyyy-MM-dd HH:mm:ss.fff'."""
    timestamps = (grp_wrist['yyyy-MM-dd'].str.strip()
                  + ' '
                  + grp_wrist['HH:mm:ss.fff'].str.strip())
    return pd.Series(grp_wrist['y_true'].to_numpy(), index=timestamps.values, name='y_true')


def extract_condition(folder_name: str) -> str:
    folder_lower = folder_name.lower()
    for kw in CONDITION_KEYWORDS:
        if kw in folder_lower:
            return kw
    return 'normal' 

def parse_time(t_str):
    h, m, s_ms = t_str.strip().split(':')
    s, ms = s_ms.split('.')
    return time(int(h), int(m), int(s), int(ms) * 1000)

def load_segmented(DATA_PATH, file_name) -> pd.DataFrame:
    try:
        # open the file 
        filepath = os.path.join(DATA_PATH, file_name)
        with open(filepath, newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)

        # clip the first 10 seconds depending on the data path 
        rows = rows if "mixed" in DATA_PATH else rows[500:]
        if DEBUG == True:
            print("Data taken fully.") if "mixed" in DATA_PATH else print("First 10s are clipped.")

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

def merge_all_wrists(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk data_path, load every s1_1RW.txt and s2_2LW.txt, attach metadata,
    and return (rw_merged, lw_merged).

    Each returned DataFrame has columns:
        subject | condition | y_true | acc_is | acc_ml | acc_pa
    """
    rw_chunks: list[pd.DataFrame] = []
    lw_chunks: list[pd.DataFrame] = []
    chunks: list[pd.DataFrame] = []

    acc_col = ['acc_is', 'acc_ml', 'acc_pa']

    if PRINT_STATS == True:
        print(f"Scanning: {data_path}\n")
        print(f"{'Folder':<35} | {'Cond':<10} | {'RW rows':>8} | {'LW rows':>8}")
        print("-" * 80)

    for folder in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue

        condition = extract_condition(folder)
        subject   = folder

        rw_path = os.path.join(folder_path, 's1_1RW.txt')
        lw_path = os.path.join(folder_path, 's2_2LW.txt')
        rw_rows = lw_rows = 0

        if os.path.exists(rw_path):
            rw_df = load_segmented(folder_path, 's1_1RW.txt')
        #    ['yyyy-MM-dd', 'HH:mm:ss.fff', 'gyrX', 'gyrY', 'gyrZ', 
        # 'accX', 'accY', 'accZ', 'magX', 'magY', 'magZ', 
        # 'Marker', 'Energy', 'Angle', 'Classification', 'Label', 'segment']
            if rw_df is not None:
                # assign true values 
                if 'test' in folder.lower() or 'sub' in folder.lower():
                    rw_df['y_true']    = rw_df['Label'].astype(int).to_numpy()
                else:
                    activity = folder.split('_')[0].lower()
                    rw_df['y_true'] =  np.ones(len(rw_df)) if activity in GAIT_CLASSES else np.zeros(len(rw_df))

                acc_cols = [c for c in rw_df.columns if 'acc' in c]
                if len(acc_cols) < 3:
                    print("Incorrect number of columns.")
                    print(f"On {rw_path}, {len(acc_cols)} columns found instead.")
                rw_df[acc_col] = rw_df[acc_cols[:3]].copy().astype(float) * 9.8

                rw_df['condition'] = condition
                rw_df['subject']   = subject
                rw_df['wrist'] = "right"
                chunks.append(rw_df)
                rw_rows = len(rw_df)
        if os.path.exists(lw_path):
            lw_df = load_segmented(folder_path, 's2_2LW.txt')
            if lw_df is not None:
                # print(" in folder:", folder)
                if 'test' in folder.lower() or 'sub' in folder.lower():
                    lw_df['y_true']    = lw_df['Label'].astype(int).to_numpy()
                else: 
                    activity = folder.split('_')[0].lower()
                    # print("activity is", activity)
                    lw_df['y_true'] =  np.ones(len(lw_df)) if activity in GAIT_CLASSES else np.zeros(len(lw_df))

                acc_cols = [c for c in lw_df.columns if 'acc' in c]
                if len(acc_cols) < 3:
                    print("Incorrect number of columns.")
                    print(f"On {lw_path}, {len(acc_cols)} columns found instead.")
                lw_df[acc_col] = lw_df[acc_cols[:3]].copy().astype(float) * 9.8

                lw_df['condition'] = condition
                lw_df['subject']   = subject
                lw_df['wrist'] = "left"
                chunks.append(lw_df)
                lw_rows = len(lw_df)
  
        if (rw_rows > 0 or lw_rows > 0) and PRINT_STATS:
            print(f"{folder[:35]:<35} | {condition:<10} | {rw_rows:>8} | {lw_rows:>8}")

    if PRINT_STATS == True:
        print("-" * 80)

    col_order = ['yyyy-MM-dd', 'HH:mm:ss.fff', 
                 'acc_is', 'acc_ml', 'acc_pa', 
                 'segment', 'y_true', 'condition', 'subject']
    col_or = ['yyyy-MM-dd', 'HH:mm:ss.fff', 
              'acc_is', 'acc_ml', 'acc_pa', 
              'segment', 'y_true', 'condition', 'subject', 'wrist']

    # rw_merged = (pd.concat(rw_chunks, ignore_index=True)[col_order]
    #              if rw_chunks else pd.DataFrame(columns=col_order))
    # lw_merged = (pd.concat(lw_chunks, ignore_index=True)[col_order]
    #              if lw_chunks else pd.DataFrame(columns=col_order))
    
    df_merged = (pd.concat(chunks, ignore_index=True)[col_or]
                 if chunks else pd.DataFrame(columns=col_or))

    return df_merged

def run_gsd_on_segment(grp) : 
    """
    Input grp: columns = 'yyyy-MM-dd', 'HH:mm:ss.fff', 
                         'accX', 'accY', 'accZ', 
                         'segment', 'y_true', 'condition', 'subject'
    """
    grp_start_idx = grp.index[0]  # offset into the full df
    # print("global_start_idx", grp_start_idx)
    # find the acceleration columns 
    acc_cols = [c for c in grp.columns if 'acc' in c]
    if len(acc_cols) < 3:
        print("Incorrect number of columns.")
        print(f"{len(acc_cols)} columns found instead.")
    # rename the columns and run the gsd on them
    seg_imu = grp[acc_cols].copy().astype(float) 
    seg_imu.columns = ['acc_is', 'acc_ml', 'acc_pa']
    # seg_imu.reset_index(drop=True)
    seg_imu = seg_imu.reset_index(drop=True)
    
    gsd = KheirkhahanGSD(threshold_still=THRESHOLD_STILL)
    bout_result = gsd.detect(seg_imu, sampling_rate_hz=SAMPLING_RATE)
    activity_counts, _ = gsd.get_activity(seg_imu, sampling_rate_hz=SAMPLING_RATE)
    std_norm = gsd.get_std_norm(seg_imu, sampling_rate_hz=SAMPLING_RATE)

    return bout_result, activity_counts, std_norm, grp_start_idx

def process_gait_from_wrists(df_merged: pd.DataFrame, #lw_merged: pd.DataFrame,
                 save_results: bool = True, print_stats: bool = False) -> pd.DataFrame:
    """
    Run GSD on every (subject, wrist) segment inside the merged DataFrames.
    Prints a per-file table and condition/wrist averages, saves HickeyGSD_Results.csv.
    """
    results = []

    print(f"\n{'Subject':<35} | {'Wrist':<5} | {'Cond':<10} | "
          f"{'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 90)

    for subject, grp_sub in df_merged.groupby('subject', sort=True):
        condition = grp_sub['condition'].iloc[0]

        wrist_preds = {}
        wrist_trues = {}
        wrist_grps  = {} 

        for wrist_label, grp_wrist in grp_sub.groupby('wrist', sort=True):
            wrist_grps[wrist_label] = grp_wrist.copy()  
            grp_wrist = grp_wrist.reset_index(drop=True)
            y_true    = grp_wrist['y_true'].to_numpy()
            y_pred    = np.full(len(grp_wrist), np.nan)

            activity_counts_timeline = {}
            std_timeline = {}

            for segment, grp_seg in grp_wrist.groupby('segment', sort=True):
                if len(grp_seg) < MIN_SEGMENT_SAMPLES:
                    # leave as NaN — already initialised that way
                    if DEBUG:
                        print("Segment is too small, skipping.")
                    continue

                bout_result, activity_counts, std_norm, global_start_idx = run_gsd_on_segment(grp_seg)

                global_start_sec = global_start_idx // SAMPLING_RATE
                for i, val in enumerate(activity_counts):
                    activity_counts_timeline[global_start_sec + i] = val
                for i, val in enumerate(std_norm):
                    std_timeline[global_start_sec + i] = val

                # Default non-skipped segment to 0 before filling bouts
                y_pred[grp_seg.index] = 0

                if hasattr(bout_result, 'gs_list_') and not bout_result.gs_list_.empty:
                    for _, bout_row in bout_result.gs_list_.iterrows():
                        local_bout_start = int(bout_row['start']) + global_start_idx
                        local_bout_end   = int(bout_row['end'])   + global_start_idx
                        y_pred[local_bout_start:local_bout_end] = 1

            wrist_preds[wrist_label] = y_pred
            wrist_trues[wrist_label] = y_true

        # ── Combine wrists ────────────────────────────────────────────────────
        if len(wrist_preds) == 2:
            rw_key = next(k for k in wrist_preds if 'right' in k.lower())
            lw_key = next(k for k in wrist_preds if 'left'  in k.lower())

            pred_rw = wrist_preds[rw_key]
            pred_lw = wrist_preds[lw_key]
            min_len    = min(len(pred_rw), len(pred_lw))

            # Build timestamp-indexed Series for predictions and ground truth
            # grp_wrist DataFrames were stored before reset_index so we need them too
            pred_rw_s = _pred_series(wrist_grps[rw_key], wrist_preds[rw_key])
            pred_lw_s = _pred_series(wrist_grps[lw_key], wrist_preds[lw_key])
            true_s    = _true_series(wrist_grps[rw_key])   # same activity = same label

            # Align: only keep timestamps present in BOTH wrists
            aligned = pd.DataFrame({
                'pred_rw': pred_rw_s,
                'pred_lw': pred_lw_s,
                'y_true':  true_s,
            }).dropna(subset=['pred_rw', 'pred_lw'])       # drop rows where either wrist was NaN-skipped

            y_true_comb = aligned['y_true'].to_numpy()
            y_pred_comb = ((aligned['pred_rw'] == 1) & (aligned['pred_lw'] == 1)).astype(float).to_numpy()
            valid_mask  = np.ones(len(aligned), dtype=bool)   # already clean after dropna

        elif len(wrist_preds) == 1:
            only_key    = list(wrist_preds.keys())[0]
            y_pred_comb = wrist_preds[only_key]
            y_true_comb = wrist_trues[only_key]
            valid_mask  = ~np.isnan(y_pred_comb)
        else:
            continue   # no data at all for this subject

        # ── Score the combined prediction ─────────────────────────────────────
        vm = valid_mask  # shorthand
        acc  = accuracy_score (y_true_comb[vm], y_pred_comb[vm])
        prec = precision_score(y_true_comb[vm], y_pred_comb[vm], zero_division=0)
        rec  = recall_score   (y_true_comb[vm], y_pred_comb[vm], zero_division=0)
        f1   = f1_score       (y_true_comb[vm], y_pred_comb[vm], zero_division=0)

        results.append({
            'Subject':   subject,
            'Wrist':     'both' if len(wrist_preds) == 2 else list(wrist_preds.keys())[0],
            'Folder':    subject,
            'Condition': condition,
            'Accuracy':  acc, 'Precision': prec, 'Recall': rec, 'F1': f1,
            'TP': int(np.sum((y_pred_comb[vm] == 1) & (y_true_comb[vm] == 1))),
            'FP': int(np.sum((y_pred_comb[vm] == 1) & (y_true_comb[vm] == 0))),
            'FN': int(np.sum((y_pred_comb[vm] == 0) & (y_true_comb[vm] == 1))),
            'TN': int(np.sum((y_pred_comb[vm] == 0) & (y_true_comb[vm] == 0))),
        })

        if print_stats:
            print(f"{subject[:35]:<35} | {"fused":<5} | {condition:<10} | "
                  f"{acc:.2f}   | {prec:.2f}   | {rec:.2f}   | {f1:.2f}")

    # ── Everything below this line is unchanged ───────────────────────────────
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
        print("-" * 90)
        _print_avg("AVERAGE (RW  Right Wrist)", res_df[res_df['Wrist'] == 'RW'])
        _print_avg("AVERAGE (LW  Left Wrist)",  res_df[res_df['Wrist'] == 'LW'])
        print()
    for condition in sorted(res_df['Condition'].unique()):
        _print_avg(f"AV(cond={condition})", res_df[res_df['Condition'] == condition])
    print("-" * 90)
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

if __name__ == "__main__":
    all_rw: list[pd.DataFrame] = []
    all_lw: list[pd.DataFrame] = []
    all_df: list[pd.DataFrame] = []

    for data_path in DATA_PATHS:
        dataset_name = os.path.basename(data_path.rstrip('/\\'))
        if PRINT_STATS:
            print(f"\n{'=' * 80}")
            print(f"  Merging: {dataset_name}")
            print(f"{'=' * 80}")

        # chech which dataset to process 
        # Tag each row with its source dataset for traceability
        df = merge_all_wrists(data_path)
        df['dataset'] = dataset_name
        # lw['dataset'] = dataset_name
        # all_rw.append(rw)
        all_df.append(df)
        
    # Pool across all datasets
    rw_merged = pd.concat(all_rw, ignore_index=True) if all_rw else pd.DataFrame()
    lw_merged = pd.concat(all_lw, ignore_index=True) if all_lw else pd.DataFrame()
    df_merged = pd.concat(all_df, ignore_index=True) if all_df else pd.DataFrame()

    print(f"\n{'=' * 80}")
    print(f"  Running GSD on pooled data ({len(DATA_PATHS)} dataset(s))")
    print(f"{'=' * 80}")
    process_gait_from_wrists(df_merged, save_results=SAVE_RESULTS, print_stats=PRINT_STATS)