import os
import pandas as pd
import numpy as np
import glob
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from multimob.GSD.GSD3 import KheirkhahanGSD
from GSD2a import HickeyGSD

# Suppress the DtypeWarning for the walkway columns
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
DATA_PATH = r"C:\Users\orlov\intern\gait_detection\QSense_data_edge"
SAMPLING_RATE = 50 
DEBUG = False; 
GAIT_CLASSES = {'walking', 'stairs'}
CONDITION_KEYWORDS = ['pockets', 'phone', 'limp', 'armfixed', 'rail', 'free']

def extract_condition(folder_name: str) -> str:
    folder_lower = folder_name.lower()
    for kw in CONDITION_KEYWORDS:
        if kw in folder_lower:
            return kw
    return 'normal' 

def load_wrist_file(filepath: str) -> pd.DataFrame | None:
    """Load a wrist IMU file and return a DataFrame with renamed acc columns."""
    try:
        df = pd.read_csv(filepath, sep='\t', low_memory=False)
        acc_cols = [c for c in df.columns if 'acc' in c.lower()]
        if len(acc_cols) < 3:
            print(f"  [SKIP] Not enough acc columns in {filepath} (found {len(acc_cols)})")
            return None

        imu_df = df[acc_cols[:3]].copy()
        imu_df = imu_df * 9.81  # convert to m/s²
        imu_df.columns = ['acc_pa', 'acc_ml', 'acc_is']
        return imu_df

    except Exception as e:
        print(f"  [ERROR] Failed to load {filepath}: {e}")
        return None

def merge_all_wrists() -> tuple[pd.DataFrame, pd.DataFrame]:
    rw_chunks: list[pd.DataFrame] = []
    lw_chunks: list[pd.DataFrame] = []

    print(f"Scanning: {DATA_PATH}\n")
    print(f"{'Folder':<35} | {'Cond':<10} | {'Gait':<5} | {'RW rows':>8} | {'LW rows':>8}")
    print("-" * 80)

    for folder in sorted(os.listdir(DATA_PATH)):
        folder_path = os.path.join(DATA_PATH, folder)
        if not os.path.isdir(folder_path):
            continue

        activity = folder.split('_')[0].lower()
        
        y_label   = 1 if activity in GAIT_CLASSES else 0
        condition = extract_condition(folder)
        subject   = folder

        rw_path = os.path.join(folder_path, 's1_1RW.txt')
        lw_path = os.path.join(folder_path, 's2_2LW.txt')
        rw_rows = lw_rows = 0

        if os.path.exists(rw_path):
            rw_df = load_wrist_file(rw_path)
            if rw_df is not None:
                rw_df['y_true']    = y_label
                rw_df['condition'] = condition
                rw_df['subject']   = subject
                rw_chunks.append(rw_df)
                rw_rows = len(rw_df)

        if os.path.exists(lw_path):
            lw_df = load_wrist_file(lw_path)
            if lw_df is not None:
                lw_df['y_true']    = y_label
                lw_df['condition'] = condition
                lw_df['subject']   = subject
                lw_chunks.append(lw_df)
                lw_rows = len(lw_df)

        if rw_rows > 0 or lw_rows > 0:
            print(f"{folder[:35]:<35} | {condition:<10} | {y_label:<5} | {rw_rows:>8} | {lw_rows:>8}")

    print("-" * 80)

    col_order = ['subject', 'condition', 'y_true', 'acc_pa', 'acc_ml', 'acc_is']

    rw_merged = (pd.concat(rw_chunks, ignore_index=True)[col_order]
                 if rw_chunks else pd.DataFrame(columns=col_order))
    lw_merged = (pd.concat(lw_chunks, ignore_index=True)[col_order]
                 if lw_chunks else pd.DataFrame(columns=col_order))

    # Save copies so you have them on disk too
    rw_merged.to_csv('merged_RW.csv', index=False)
    lw_merged.to_csv('merged_LW.csv', index=False)
    print(f"\n[RW] {len(rw_merged):,} rows saved → merged_RW.csv")
    print(f"[LW] {len(lw_merged):,} rows saved → merged_LW.csv\n")

    return rw_merged, lw_merged

def _run_gsd_on_group(imu_df: pd.DataFrame, y_true: np.ndarray,
                      label: str) -> dict | None:
    """
    Run HickeyGSD on a single contiguous imu_df block, evaluate against y_true,
    and return a result dict (or None on error).
    """
    try:
        gsd = HickeyGSD(debug=DEBUG)
        detected_bouts = (gsd
                          .preprocess(imu_df,
                                      sampling_rate_hz=SAMPLING_RATE,
                                      target_sampling_rate_hz=SAMPLING_RATE)
                          .detect_wrist())

        # Convert bout list → binary mask
        y_pred = np.zeros(len(imu_df), dtype=int)
        if hasattr(detected_bouts, 'gs_list_') and not detected_bouts.gs_list_.empty:
            for _, row in detected_bouts.gs_list_.iterrows():
                start = int(max(0, row['start']))
                end   = int(min(len(imu_df), row['end']))
                y_pred[start:end] = 1

        if DEBUG:
            print(f"\n[{label}] pred walking: {y_pred.sum()} / {len(y_pred)} samples")
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

        return {
            'Accuracy':  accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall':    recall_score(y_true, y_pred, zero_division=0),
            'F1':        f1_score(y_true, y_pred, zero_division=0),
        }

    except Exception as e:
        print(f"  [ERROR] GSD failed on {label}: {e}")
        return None

def process_weargait():
    results = []
    
    print(f"\n{'Subject':<35} | {'Wrist':<5} | {'Cond':<10} | "
          f"{'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 90)

    rw_merged, lw_merged = merge_all_wrists()

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

            metrics = _run_gsd_on_group(imu_df, y_true, label)
            if metrics is None:
                continue

            results.append({
                'Subject':   label,
                'Wrist':     wrist_label,
                'Folder':    subject,
                'Condition': condition,
                **metrics,
            })

            print(f"{label[:35]:<35} | {wrist_label:<5} | {condition:<10} | "
                  f"{metrics['Accuracy']:.2f}   | {metrics['Precision']:.2f}   | "
                  f"{metrics['Recall']:.2f}   | {metrics['F1']:.2f}")

    # Summary
    if not results:
        print("No results to summarise.")
        return pd.DataFrame()

    res_df = pd.DataFrame(results)

    METRIC_COLS = ['Accuracy', 'Precision', 'Recall', 'F1']

    def _avg_row(row_type: str, label: str,
                 wrist: str, condition: str,
                 subset: pd.DataFrame) -> dict:
        """Build a single summary dict from a subset of res_df."""
        return {
            'row_type':  row_type,
            'Subject':   label,
            'Wrist':     wrist,
            'Folder':    '',
            'Condition': condition,
            **{m: round(subset[m].mean(), 4) for m in METRIC_COLS},
        }

    def _print_avg(label: str, subset: pd.DataFrame):
        if subset.empty:
            return
        print(f"{label:<35} | {'':5} | {'':10} | "
              f"{subset['Accuracy'].mean():.2f}   | "
              f"{subset['Precision'].mean():.2f}   | "
              f"{subset['Recall'].mean():.2f}   | "
              f"{subset['F1'].mean():.2f}")

    avg_rows: list[dict] = []

    # Per-wrist averages
    for wrist in ['RW', 'LW']:
        sub = res_df[res_df['Wrist'] == wrist]
        if not sub.empty:
            avg_rows.append(_avg_row('avg_wrist',
                                     f"AVERAGE ({wrist})",
                                     wrist, '', sub))

    # Per-condition averages (all wrists combined)
    for condition in sorted(res_df['Condition'].unique()):
        sub = res_df[res_df['Condition'] == condition]
        avg_rows.append(_avg_row('avg_condition',
                                  f"AVERAGE (cond={condition})",
                                  '', condition, sub))

    # Per-wrist × per-condition averages
    for wrist in ['RW', 'LW']:
        for condition in sorted(res_df['Condition'].unique()):
            sub = res_df[(res_df['Wrist'] == wrist) &
                         (res_df['Condition'] == condition)]
            if not sub.empty:
                avg_rows.append(_avg_row('avg_wrist_condition',
                                          f"AVERAGE ({wrist}, cond={condition})",
                                          wrist, condition, sub))

    # Overall average
    avg_rows.append(_avg_row('avg_overall', 'AVERAGE (Overall)', '', '', res_df))

    # Print summary to console
    print("-" * 90)
    _print_avg("AVERAGE (RW – Right Wrist)", res_df[res_df['Wrist'] == 'RW'])
    _print_avg("AVERAGE (LW – Left Wrist)",  res_df[res_df['Wrist'] == 'LW'])
    print()
    for condition in sorted(res_df['Condition'].unique()):
        _print_avg(f"AV(cond={condition})", res_df[res_df['Condition'] == condition])
    print("-" * 90)
    _print_avg("AVERAGE (Overall)", res_df)

    # 
    res_df.insert(0, 'row_type', 'result')

    avg_df    = pd.DataFrame(avg_rows)
    blank_row = pd.DataFrame([{c: '' for c in res_df.columns}])

    csv_df = pd.concat(
        [res_df,
         blank_row,
         avg_df],
        ignore_index=True
    )

    csv_df.to_csv('HickeyGSD_Results.csv', index=False)
    print("\nSaved → HickeyGSD_Results.csv")

    return res_df

if __name__ == "__main__":
    process_weargait()
