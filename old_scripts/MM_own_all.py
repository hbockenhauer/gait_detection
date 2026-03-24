import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from GSD3_test import KheirkhahanGSD
# from multimob.GSD.GSD3 import KheirkhahanGSD
from multimob.GSD.GSD4 import MacLeanGSD
from multimob.GSD.GSD5 import KerenGSD
from GSD2a import HickeyGSD

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
DATA_PATHS = [
    r"C:\Users\orlov\intern\gait_detection\QSense_data_edge",
    r"C:\Users\orlov\intern\gait_detection\QSense_data_mixed",
    r"C:\Users\orlov\intern\gait_detection\QSense_data"
]
GSD_n = 3
SAMPLING_RATE = 50 
DEBUG = False; 
GAIT_CLASSES = {'walking', 'stairs'}
CONDITION_KEYWORDS = ['pockets', 'phone', 'rail', 'free', 'crutches', 'walker', 'cane']
SAVE_RESULTS = True 
PRINT_STATS = True 

def extract_condition(folder_name: str) -> str:
    folder_lower = folder_name.lower()
    for kw in CONDITION_KEYWORDS:
        if kw in folder_lower:
            return kw
    return 'normal' 

def is_gait(folder_name: str, df:pd.DataFrame = None) -> int:
    if 'test' in folder_name.lower():
        return df['Label']
    else:
        activity = folder_name.split('_')[0].lower()
        return 1 if activity in GAIT_CLASSES else 0


def load_wrist_file(filepath: str) -> pd.DataFrame | None:
    """Read one sensor file, scale acc to m/s², rename columns. Returns None on failure."""
    try:
        df = pd.read_csv(filepath, sep=None, engine="python")
        df = df.reset_index(drop=True)

        # clip the first 10 seconds for files with no 'test' in the name
        df = df #if 'test' in filepath else df = df[:500]

        acc_cols = [c for c in df.columns if 'acc' in c.lower()]
        if len(acc_cols) < 3:
            print(f"  [SKIP] Not enough acc columns in {filepath} (found {len(acc_cols)})")
            return None
        
        imu_df = df[acc_cols[:3]].copy()
        imu_df = imu_df * 9.81          # convert g → m/s²
        imu_df.columns = ['acc_pa', 'acc_ml', 'acc_is']

        return imu_df, df
    except Exception as e:
        print(f"  [ERROR] Failed to load {filepath}: {e}")
        return None


def merge_all_wrists(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk data_path, load every s1_1RW.txt and s2_2LW.txt, attach metadata,
    and return (rw_merged, lw_merged).

    Each returned DataFrame has columns:
        subject | condition | y_true | acc_pa | acc_ml | acc_is
    """
    rw_chunks: list[pd.DataFrame] = []
    lw_chunks: list[pd.DataFrame] = []

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
            rw_df, df_full = load_wrist_file(rw_path)
            if rw_df is not None:
                rw_df['y_true']    = is_gait(folder, df_full)
                rw_df['condition'] = condition
                rw_df['subject']   = subject
                rw_chunks.append(rw_df)
                rw_rows = len(rw_df)

        if os.path.exists(lw_path):
            lw_df, df_full = load_wrist_file(lw_path)
            if lw_df is not None:
                lw_df['y_true']    = is_gait(folder, df_full)
                lw_df['condition'] = condition
                lw_df['subject']   = subject
                lw_chunks.append(lw_df)
                lw_rows = len(lw_df)

        
        if (rw_rows > 0 or lw_rows > 0) and PRINT_STATS:
            print(f"{folder[:35]:<35} | {condition:<10} | {rw_rows:>8} | {lw_rows:>8}")

    if PRINT_STATS == True:
        print("-" * 80)

    col_order = ['subject', 'condition', 'y_true', 'acc_pa', 'acc_ml', 'acc_is']

    rw_merged = (pd.concat(rw_chunks, ignore_index=True)[col_order]
                 if rw_chunks else pd.DataFrame(columns=col_order))
    lw_merged = (pd.concat(lw_chunks, ignore_index=True)[col_order]
                 if lw_chunks else pd.DataFrame(columns=col_order))

    return rw_merged, lw_merged

def _run_gsd_on_group(imu_df: pd.DataFrame, y_true: np.ndarray,
                      label: str) -> dict | None:
    """
    Run GSD on a single contiguous imu_df block, evaluate against y_true,
    and return a result dict (or None on error).
    Input: 
     - data
     - true labels 
     - label (for finding the error)
    Output: 
     - metrics 
     - name of the algo for the file 
    """
    try:
        match GSD_n:
            case 2:
                # Run the Hickey GSD method
                gsd = HickeyGSD(debug=DEBUG)
                detected_bouts = (gsd.preprocess(imu_df, sampling_rate_hz=SAMPLING_RATE, target_sampling_rate_hz=SAMPLING_RATE)
                          .detect_wrist())
                output_name ='Results/HickeyGSD_Results.csv'
            case 3:
                # Run Kheirkhahan GSD
                gsd = KheirkhahanGSD()
                detected_bouts = gsd.detect(imu_df, sampling_rate_hz=SAMPLING_RATE)
                output_name = 'Results/KheirkhahanGSD_Results.csv'
            case 4: 
                # Run MacLean GSD
                gsd = MacLeanGSD()
                detected_bouts = gsd.detect(imu_df)
                output_name = 'Results/MacLeanGSD_Results.csv'
            case 5:
                # Run Keren GSD
                gsd = KerenGSD()
                detected_bouts = gsd.detect(imu_df, sampling_rate_hz=SAMPLING_RATE)
                output_name = 'Results/KerenGSD_Results.csv'

        # Convert bout list → binary mask
        y_pred = np.zeros(len(imu_df), dtype=int)
        if hasattr(detected_bouts, 'gs_list_') and not detected_bouts.gs_list_.empty:
            for _, row in detected_bouts.gs_list_.iterrows():
                start = int(max(0, row['start']))
                end   = int(min(len(imu_df), row['end']))
                y_pred[start:end] = 1

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        if DEBUG:
            print(f"\n[{label}] pred walking: {y_pred.sum()} / {len(y_pred)} samples")
            print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

        return {
            'Accuracy':  accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall':    recall_score(y_true, y_pred, zero_division=0),
            'F1':        f1_score(y_true, y_pred, zero_division=0),
            'TP': tp, 
            'FP': fp, 
            'FN': fn,
            'TN': tn,
        }, output_name

    except Exception as e:
        print(f"  [ERROR] GSD failed on {label}: {e}")
        return None


def process_gait(rw_merged: pd.DataFrame,
                 lw_merged: pd.DataFrame, 
                 save_results: bool = True) -> pd.DataFrame:
    """
    Run GSD on every (subject, wrist) segment inside the merged DataFrames.
    Prints a per-file table and condition/wrist averages, saves HickeyGSD_Results.csv.
    """
    results = []

    print(f"\n{'Subject':<35} | {'Wrist':<5} | {'Cond':<10} | "
          f"{'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 90)

    for wrist_label, merged_df in [('RW', rw_merged), ('LW', lw_merged)]:
        if merged_df.empty:
            print(f"[{wrist_label}] No data — skipping.")
            continue

        imu_cols = ['acc_pa', 'acc_ml', 'acc_is']

        # Process each recording (subject folder) separately so the GSD sees
        # one continuous, coherent signal — not a mix of activities concatenated.
        for subject, grp in merged_df.groupby('subject', sort=True):
            imu_df    = grp[imu_cols].reset_index(drop=True)
            y_true    = grp['y_true'].to_numpy()
            condition = grp['condition'].iloc[0]
            label     = f"{subject}/{wrist_label}"

            result = _run_gsd_on_group(imu_df, y_true, label)
            if result is None:
                continue
            metrics, output_name = result
            

            results.append({
                'Subject':   label,
                'Wrist':     wrist_label,
                'Folder':    subject,
                'Condition': condition,
                **metrics,
            })

            if PRINT_STATS == True: 
                print(f"{label[:35]:<35} | {wrist_label:<5} | {condition:<10} | "
                      f"{metrics['Accuracy']:.2f}   | {metrics['Precision']:.2f}   | "
                      f"{metrics['Recall']:.2f}   | {metrics['F1']:.2f}")

    if not results:
        print("No results to summarise.")
        return pd.DataFrame()

    res_df = pd.DataFrame(results)

    #METRIC_COLS = ['Accuracy', 'Precision', 'Recall', 'F1']
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
            #**{m: round(subset[m].mean(), 4) for m in METRIC_COLS},
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
        csv_df.to_csv(output_name, index=False)
        print(f"\nSaved → {output_name}")

    return res_df

if __name__ == "__main__":
    all_rw: list[pd.DataFrame] = []
    all_lw: list[pd.DataFrame] = []

    for data_path in DATA_PATHS:
        dataset_name = os.path.basename(data_path.rstrip('/\\'))
        print(f"\n{'=' * 80}")
        print(f"  Merging: {dataset_name}")
        print(f"{'=' * 80}")
        rw, lw = merge_all_wrists(data_path)
        # Tag each row with its source dataset for traceability
        rw['dataset'] = dataset_name
        lw['dataset'] = dataset_name
        all_rw.append(rw)
        all_lw.append(lw)

    # Pool across all datasets
    rw_merged = pd.concat(all_rw, ignore_index=True) if all_rw else pd.DataFrame()
    lw_merged = pd.concat(all_lw, ignore_index=True) if all_lw else pd.DataFrame()

    # Save pooled merged files
    # rw_merged.to_csv('merged_RW.csv', index=False)
    # lw_merged.to_csv('merged_LW.csv', index=False)
    # print(f"\n[POOLED RW] {len(rw_merged):,} rows → merged_RW.csv")
    # print(f"[POOLED LW] {len(lw_merged):,} rows → merged_LW.csv")

    print(f"\n{'=' * 80}")
    print(f"  Running GSD on pooled data ({len(DATA_PATHS)} dataset(s))")
    print(f"{'=' * 80}")
    process_gait(rw_merged, lw_merged, SAVE_RESULTS)