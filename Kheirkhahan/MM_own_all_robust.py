"""
Runs the GSD on all QSense data
Robust to faulty data
"""

import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from Kheirkhahan.GSD3_test import KheirkhahanGSD
from Hickey.GSD2a import HickeyGSD
import csv
from datetime import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from Kheirkhahan.free_living_test import merge_csv
from Kheirkhahan.singleGSD_robust import load_segmented

from config.paths import (
    QSENSE_CLINIC, 
    QSENSE_DATA, 
    QSENSE_EDGE, 
    QSENSE_MIXED,  
    FREELIVING_PATH,  
    RESULTS_DIR, 
    PLOTS_DIR
)

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

######### can be adjusted #####################
THRESHOLD_STILL = 0.1 # set to 0.0 to disable the Hickey element

DEBUG = False
PRINT_STATS = True 

SAVE_RESULTS = False 
OUTPUT_FILE = "Kheirkhahan/KheirkhahanGSD_Results_wHickey.csv"

PLOT = False # plots and saves all plots
folder = "Kheirkhahan_plots" # folder to be saved in 
##############################


DATA_PATHS = [
    QSENSE_CLINIC, 
    QSENSE_DATA, 
    QSENSE_EDGE, 
    QSENSE_MIXED, 
    FREELIVING_PATH
]

SAMPLING_RATE = 50 
GAIT_CLASSES = {'walking', 'stairs'}
CONDITION_KEYWORDS = ['pockets', 'phone', 'rail', 'free', 'crutches', 'walker', 
                      'cane', 'limp', 'armfixed', 'stroke', 
                      'sub1', 'sub2','sub3', 'sub4', "sub5"]
MIN_SEGMENT_SAMPLES = 9*SAMPLING_RATE 
OUT_FOLDER = os.path.join(PLOTS_DIR, folder)


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
                rw_chunks.append(rw_df)
                rw_rows = len(rw_df)
        if os.path.exists(lw_path):
            lw_df = load_segmented(folder_path, 's2_2LW.txt', DEBUG)
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

def plot_results(df: pd.DataFrame, activity_counts_timeline, 
                 std_norm_timeline, y_pred, y_true, title,
                 threshold: float = THRESHOLD_STILL):
    # Parse timestamps from df into timedeltas
    time_series = pd.to_timedelta(df['HH:mm:ss.fff'].str.strip())
    # Convert to total seconds (float) for plotting
    time_per_second_sec = time_series.iloc[::SAMPLING_RATE].reset_index(drop=True).dt.total_seconds()

    total_seconds = len(time_per_second_sec)
    ac_plot = np.full(total_seconds, np.nan)
    for sec_idx, val in activity_counts_timeline.items():
        if sec_idx < total_seconds:
            ac_plot[sec_idx] = val
    
    std_plot = np.full(total_seconds, np.nan)
    for sec_idx, val in std_norm_timeline.items():
        if sec_idx < total_seconds:
            std_plot[sec_idx] = val

    all_segment_first_rows = df.groupby('segment').nth(0).index
    all_segment_last_rows = df.groupby('segment').nth(-1).index
    jump_row_indices = all_segment_first_rows[1:]
    jump_times_sec = [time_series.iloc[idx].total_seconds() for idx in jump_row_indices]


    time_all_sec = time_series.dt.total_seconds() # seconds from midnight, accurate to 2 decimals
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(13, 9), sharex=True)

    # ── figure 1: raw data ────────────────────────────────────────────────────
    ax1.fill_between(time_all_sec, -1, 2, where=(y_true == 1),
                    alpha=0.2, color='green', transform=ax1.get_xaxis_transform(),
                    label='Ground truth (walking)')
    acc_cols = [c for c in df.columns if 'acc' in c]
    for acc in acc_cols:
        ax1.plot(time_all_sec, df[acc], label=acc, alpha=0.8, marker='.', linestyle='None', markersize=3)

    for i, jt in enumerate(jump_times_sec):
        ax1.axvline(x=jt, color='orange', linewidth=1.0, linestyle='--', alpha=0.8,
                    label='Time gap' if i == 0 else None)

    ax1.set_ylabel(f'Acceleration (m/s^{2})')
    ax1.set_title(f'{title}')
    ax1.legend(loc='upper left')

    # ── figure 2: activity counts ───────────────────────────────────────────────
    ax2.fill_between(time_all_sec, -1, 2, where=(y_true == 1),
                    alpha=0.2, color='green', transform=ax2.get_xaxis_transform(),
                    label='Ground truth (walking)')
    ax2.plot(time_per_second_sec, ac_plot, label='Activity count', 
            linewidth=1, color='steelblue')

    for i, jt in enumerate(jump_times_sec):
        ax2.axvline(x=jt, color='orange', linewidth=1.0, linestyle='--', alpha=0.8,
                    label='Time gap' if i == 0 else None)

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Activity count')
    ax2.legend(loc='upper left')

    # ── figure 3: std norm ───────────────────────────────────────────────
    ax3.fill_between(time_all_sec, -1, 2, where=(y_true == 1),
                    alpha=0.2, color='green', label='Ground truth (walking)')
    ax3.plot(time_per_second_sec, std_plot, label='std norm', alpha=0.8, 
            linewidth=1, color='steelblue')
    for i, jt in enumerate(jump_times_sec):
        ax3.axvline(x=jt, color='orange', linewidth=1.0, linestyle='--', alpha=0.8,
                    label='Time gap' if i == 0 else None)
    ax3.axhline(y=threshold, color='red', linewidth=1.0, linestyle='--', alpha=0.8,
                    label='threshold')
    ax3.set_ylim(-0.1, np.nanmax(std_plot)+0.1)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Std of the norm')
    ax3.legend(loc='upper left')

    # ── figure 4: y_pred and y_true ──────────────────────────────────────────────
    ax4.fill_between(time_all_sec, -1, 2, where=(y_true == 1),
                    alpha=0.2, color='green', label='Ground truth (walking)')
    ax4.plot(time_all_sec, y_pred, label='y_pred (GSD)', alpha=0.8, 
            linewidth=1, color='steelblue')

    for i, jt in enumerate(jump_times_sec):
        ax4.axvline(x=jt, color='orange', linewidth=1.0, linestyle='--', alpha=0.8,
                    label='Time gap' if i == 0 else None)

    ax4.set_ylabel('Walking (1) / Not (0)')
    ax4.legend(loc='upper left')
    ax4.set_ylim(-0.1, 1.4)
    

    # Format x-axis as HH:MM:SS
    ax4.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x//3600):02d}:{int((x%3600)//60):02d}:{int(x%60):02d}"
    ))
    fig.autofmt_xdate()
    plt.tight_layout()

    out_path = os.path.join(OUT_FOLDER, f"{title}_predictions.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")

def process_gait(rw_merged: pd.DataFrame, lw_merged: pd.DataFrame,
                 fl_merged: pd.DataFrame, 
                 print_stats: bool = False, save_results: bool = True,
                 out_file: str = "Kheirkhahan/Own_data.cvs", 
                 plot: bool = False) -> pd.DataFrame:
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
            activity_counts_timeline = {}
            std_timeline = {}
            skipped_seg = 0

            for segment, grp_seg in grp_sub.groupby('segment', sort=True):
                if len(grp_seg) < MIN_SEGMENT_SAMPLES:
                    y_pred[grp_seg.index] = np.nan
                    skipped_seg += 1
                    continue
                bout_result, activity_counts, std_norm, global_start_idx = run_gsd_on_segment(grp_seg)

                global_start_sec = global_start_idx // SAMPLING_RATE
                for i, val in enumerate(activity_counts):
                    activity_counts_timeline[global_start_sec + i] = val
                for i, val in enumerate(std_norm):
                    std_timeline[global_start_sec + i] = val

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

            if plot == True: 
                title = f"{subject}{wrist_label}"
                plot_results(grp_sub, activity_counts_timeline, std_timeline, y_pred, y_true, title)

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
        os.path.join(RESULTS_DIR, out_file)
        csv_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved → {OUTPUT_FILE}")

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
        if "QSense" in dataset_name: 
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
    print(f"  Running GSD on pooled data ({len(DATA_PATHS)} dataset(s))")
    print(f"{'=' * 80}")
    process_gait(rw_merged, lw_merged, fl_merged, print_stats=PRINT_STATS, save_results=SAVE_RESULTS)