import os
import numpy as np
import pandas as pd
import warnings
from collections import deque
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from models.Kheirkhahan.free_living_test import merge_csv
from models.Kheirkhahan.process_datasets import (
    process_weargait, process_wisdm,  process_HMP, process_bioclite, 
    run_gsd_on_wrist)
from models.Kheirkhahan.MM_own_all_robust import (merge_all_wrists)
from models.realtime.detect_per_wrist import (simulate_realtime, WINDOW_SIZE)

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
from config.paths import (
    HMP_PATH,
    WISDM_PATH,
    WEARGAIT_PD,
    WEARGAIT_CTRL,
    QSENSE_DATA,
    QSENSE_EDGE,
    QSENSE_MIXED,
    QSENSE_CLINIC,
    FREELIVING_PATH,
    BIOCLITE_PATH, 
    RESULTS_DIR, 
    PLOTS_DIR
)

############## can be adjusted ####################
DEBUG = False
PRINT_STATS = True 

SAVE_RESULTS = True 
OUTPUT_FILE = "Realtime/Realtime_results_own.csv"
###################################################


DATA_PATHS = [ 
    # HMP_PATH, 
    # WISDM_PATH, 
    # WEARGAIT_PD,
    # WEARGAIT_CTRL,
    # QSENSE_DATA,
    QSENSE_EDGE,
    QSENSE_MIXED,
    QSENSE_CLINIC,
    # FREELIVING_PATH,
    # BIOCLITE_PATH
    ]

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
        save_path = os.path.join(RESULTS_DIR,OUTPUT_FILE)
        csv_df.to_csv(save_path, index=False)
        print(f"\nSaved → {save_path}")

    return res_df

if __name__ == "__main__":
    for data_path in DATA_PATHS:
        dataset_name = os.path.basename(data_path.rstrip('/\\'))
        if PRINT_STATS:
            print(f"\n{'=' * 80}")
            print(f"  Merging: {dataset_name}")
            print(f"{'=' * 80}")

        # check which dataset to process 
        if any(x in dataset_name for x in ["Baseline", "Clinical", "Edge_Cases", "Multiple"]):
            rw, lw = merge_all_wrists(data_path)
            rw['dataset'] = dataset_name
            lw['dataset'] = dataset_name
            fl= pd.DataFrame()
            process_realtime(rw, lw, fl, save_results=SAVE_RESULTS)

        elif "Free_living" in dataset_name: 
            fl = merge_csv(data_path, PRINT_STATS)
            fl['dataset'] = dataset_name
            rw = pd.DataFrame()
            lw = pd.DataFrame()
            process_realtime(rw, lw, fl, save_results=SAVE_RESULTS)

        elif "WearGait" in dataset_name:
            fl = process_weargait(data_path, PRINT_STATS,SAVE_RESULTS, 
                                  realtime=True, 
                                  output_file="Realtime/WearGait.csv")
        
        elif "accel" in dataset_name:
            process_wisdm(data_path, PRINT_STATS, SAVE_RESULTS,
                           realtime=True, 
                           output_file="Realtime/WISDM.csv")

        elif "HMP" in dataset_name: 
            process_HMP(data_path, PRINT_STATS, SAVE_RESULTS, 
                        realtime=True, 
                        output_file="Realtime/HMP.csv")

        elif "6activities_plain.mat" in dataset_name:
            process_bioclite(data_path, PRINT_STATS, SAVE_RESULTS, 
                             realtime=True, 
                             output_file="Realtime/Bioclite.csv")


    ###############################