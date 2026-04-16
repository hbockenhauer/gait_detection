"""
Runs the Hickey on all QSense data
Robust to faulty data
"""
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from models.Hickey.GSD2a import HickeyGSD
from datetime import time
from models.Kheirkhahan.free_living_test import merge_csv
from models.Kheirkhahan.singleGSD_robust import load_segmented

from config.paths import (
    QSENSE_DATA,
    QSENSE_EDGE,
    QSENSE_MIXED,
    QSENSE_CLINIC,
    FREELIVING_PATH,
    PLOTS_DIR, 
    RESULTS_DIR
)

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

############# can be adjusted #################
DEBUG = False
PRINT_STATS = True 

SAVE_RESULTS = True 
out_file = "Hickey/Hickey_own.csv"

PLOT = False
OUT_FOLDER = PLOTS_DIR
###############################################

DATA_PATHS = [ 
    QSENSE_DATA,
    QSENSE_EDGE,
    QSENSE_MIXED,
    QSENSE_CLINIC,
    FREELIVING_PATH
    ]

SAMPLING_RATE = 50 
GAIT_CLASSES = {'walking', 'stairs'}
CONDITION_KEYWORDS = ['pockets', 'phone', 'rail', 'free', 'crutches', 'walker', 
                      'cane', 'limp', 'armfixed', 'stroke', 
                      'sub1', 'sub2','sub3', 'sub4', "sub5"]
MIN_SEGMENT_SAMPLES = 0.1*SAMPLING_RATE 


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

def merge_all_wrists(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk data_path, load every s1_1RW.txt and s2_2LW.txt, attach metadata,
    and return (rw_merged, lw_merged).

    Each returned DataFrame has columns:
        subject | condition | y_true | acc_is | acc_ml | acc_pa
    """
    rw_chunks: list[pd.DataFrame] = []
    lw_chunks: list[pd.DataFrame] = []

    acc_col = ['acc_is', 'acc_ml', 'acc_pa']

    if PRINT_STATS == True:
        print(f"Scanning: {data_path}\n")
        print(f"{'Folder':<35} | {'Cond':<10} | {'RW rows':>8} | {'LW rows':>8}")
        print("-" * 80)

    for folder in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue

        #y_label   = 
        condition = extract_condition(folder)
        subject   = folder

        rw_path = os.path.join(folder_path, 's1_1RW.txt')
        lw_path = os.path.join(folder_path, 's2_2LW.txt')
        rw_rows = lw_rows = 0

        if os.path.exists(rw_path):
            rw_df = load_segmented(folder_path, 's1_1RW.txt', DEBUG)
        #    ['yyyy-MM-dd', 'HH:mm:ss.fff', 'gyrX', 'gyrY', 'gyrZ', 
        # 'accX', 'accY', 'accZ', 'magX', 'magY', 'magZ', 
        # 'Marker', 'Energy', 'Angle', 'Classification', 'Label', 'segment']
            if rw_df is not None:
                # assign true values 
                if 'Label' in rw_df.columns:
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
                rw_chunks.append(rw_df)
                rw_rows = len(rw_df)
        if os.path.exists(lw_path):
            lw_df = load_segmented(folder_path, 's2_2LW.txt', DEBUG)
            if lw_df is not None:
                # print(" in folder:", folder)
                if 'Label' in lw_df.columns:
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
                lw_chunks.append(lw_df)
                lw_rows = len(lw_df)
  
        if (rw_rows > 0 or lw_rows > 0) and PRINT_STATS:
            print(f"{folder[:35]:<35} | {condition:<10} | {rw_rows:>8} | {lw_rows:>8}")

    if PRINT_STATS == True:
        print("-" * 80)

    col_order = ['yyyy-MM-dd', 'HH:mm:ss.fff', 
                 'acc_is', 'acc_ml', 'acc_pa', 
                 'segment', 'y_true', 'condition', 'subject']

    rw_merged = (pd.concat(rw_chunks, ignore_index=True)[col_order]
                 if rw_chunks else pd.DataFrame(columns=col_order))
    lw_merged = (pd.concat(lw_chunks, ignore_index=True)[col_order]
                 if lw_chunks else pd.DataFrame(columns=col_order))

    return rw_merged, lw_merged

def run_Hickey_on_segment(grp) : 
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
    
    gsd = HickeyGSD()
    bout_result = gsd.preprocess(seg_imu, sampling_rate_hz=SAMPLING_RATE).detect_wrist()

    return bout_result, grp_start_idx

def process_Hickey(rw_merged: pd.DataFrame, lw_merged: pd.DataFrame,
                 fl_merged: pd.DataFrame, 
                 print_stats: bool = False, 
                 save_results: bool = True, output_file: str = "Hickey/Own_data.csv") -> pd.DataFrame:
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

            detected_bouts = []
            y_pred = np.zeros(len(grp_sub))
            skipped_seg = 0

            for _, grp_seg in grp_sub.groupby('segment', sort=True):
                if len(grp_seg) < MIN_SEGMENT_SAMPLES:
                    y_pred[grp_seg.index] = np.nan
                    skipped_seg += 1
                    continue
                bout_result, global_start_idx = run_Hickey_on_segment(grp_seg)


                if hasattr(bout_result, 'gs_list_') and not bout_result.gs_list_.empty:
                    for _, bout_row in bout_result.gs_list_.iterrows():
                        # Offset local segment indices to global df indices
                        local_bout_start = int(bout_row['start']) + global_start_idx
                        local_bout_end   = int(bout_row['end'])   + global_start_idx
                        detected_bouts.append((local_bout_start, local_bout_end))
                        y_pred[local_bout_start:local_bout_end] = 1

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
        out_path = os.path.join(RESULTS_DIR, output_file)
        csv_df.to_csv(out_path, index=False)
        print(f"\nSaved → {out_path}")

    return res_df

if __name__ == "__main__":
    all_rw: list[pd.DataFrame] = []
    all_lw: list[pd.DataFrame] = []
    all_fl: list[pd.DataFrame] = []

    for data_path in DATA_PATHS:
        dataset_name = os.path.basename(data_path.rstrip('/\\'))
        if PRINT_STATS:
            print(f"\n{'=' * 80}")
            print(f"  Merging: {dataset_name}")
            print(f"{'=' * 80}")

        # chech which dataset to process 
        if any(x in dataset_name for x in ["Baseline", "Clinical", "Edge_Cases", "Multiple"]):
            # Tag each row with its source dataset for traceability
            rw, lw = merge_all_wrists(data_path)
            rw['dataset'] = dataset_name
            lw['dataset'] = dataset_name
            all_rw.append(rw)
            all_lw.append(lw)
        else: 
            fl = merge_csv(data_path, PRINT_STATS)
            fl['dataset'] = dataset_name
            all_fl.append(fl)
        
    # Pool across all datasets
    rw_merged = pd.concat(all_rw, ignore_index=True) if all_rw else pd.DataFrame()
    lw_merged = pd.concat(all_lw, ignore_index=True) if all_lw else pd.DataFrame()
    fl_merged = pd.concat(all_fl, ignore_index=True) if all_fl else pd.DataFrame()

    print(f"\n{'=' * 80}")
    print(f"  Running HickeyGSD on pooled data ({len(DATA_PATHS)} dataset(s))")
    print(f"{'=' * 80}")
    process_Hickey(rw_merged, lw_merged, fl_merged, 
                   print_stats=PRINT_STATS, save_results=SAVE_RESULTS, 
                   output_file=out_file)