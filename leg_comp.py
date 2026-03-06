import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from GSD3_test import KheirkhahanGSD
from multimob.GSD.GSD4 import MacLeanGSD
from multimob.GSD.GSD5 import KerenGSD
from GSD2a import HickeyGSD
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
DATA_PATHS = [
    #r"C:\Users\orlov\intern\gait_detection\QSense_data",
    #r"C:\Users\orlov\intern\gait_detection\QSense_data_edge",
    r"C:\Users\orlov\intern\gait_detection\QSense_data_mixed", 
]
GSD_n = 3
SAMPLING_RATE = 50 
DEBUG = False; 
GAIT_CLASSES = {'walking', 'stairs'}
CONDITION_KEYWORDS = ['pockets', 'phone', 'rail', 'free', 'crutches', 'walker', 'cane', 'mixed']
SAVE_RESULTS = False 
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

def load_file(filepath: str) -> pd.DataFrame | None:
    """Read one sensor file, scale acc to m/s², rename columns. Returns None on failure."""
    try:
        df = pd.read_csv(filepath, sep=None, engine="python")
        df = df.reset_index(drop=True)

        # clip the first 10 seconds 
        if "s3_3RL.txt" in filepath:
            df = df
        else:
            df = df[500:]

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


def merge_all(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk data_path, load every s1_1RW.txt and s2_2LW.txt, attach metadata,
    and return (rw_merged, lw_merged).

    Each returned DataFrame has columns:
        subject | condition | y_true | acc_pa | acc_ml | acc_is
    """
    rw_chunks: list[pd.DataFrame] = []
    lw_chunks: list[pd.DataFrame] = []
    rl_chunks: list[pd.DataFrame] = []

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
        rl_path = os.path.join(folder_path, 's3_3RL.txt')


        rw_rows = lw_rows = rl_rows = 0

        if os.path.exists(rw_path):
            rw_df, df_full = load_file(rw_path)
            if rw_df is not None:
                rw_df['y_true']    = is_gait(folder, df_full)
                rw_df['condition'] = condition
                rw_df['subject']   = subject
                rw_chunks.append(rw_df)
                rw_rows = len(rw_df)

        if os.path.exists(lw_path):
            lw_df, df_full = load_file(lw_path)
            if lw_df is not None:
                lw_df['y_true']    = is_gait(folder, df_full)
                lw_df['condition'] = condition
                lw_df['subject']   = subject
                lw_chunks.append(lw_df)
                lw_rows = len(lw_df)

        if os.path.exists(rl_path):
            rl_df, df_full = load_file(rl_path)
            if rl_df is not None:
                rl_df['y_true']    = is_gait(folder, df_full)
                rl_df['condition'] = condition
                rl_df['subject']   = subject
                rl_df['energy'] = df_full['Energy']
                rl_df['classification']   = df_full['Classification']
                rl_chunks.append(rl_df)
                rl_rows = len(rl_df)        

        
        if (rw_rows > 0 or lw_rows > 0 or rl_rows > 0) and PRINT_STATS:
            print(f"{folder[:35]:<35} | {condition:<10} | {rw_rows:>8} | {lw_rows:>8}")

    if PRINT_STATS == True:
        print("-" * 80)

    col_order = ['subject', 'condition', 'y_true', 'acc_pa', 'acc_ml', 'acc_is']
    col_order_leg = ['subject', 'condition', 'y_true', 'acc_pa', 'acc_ml', 'acc_is', 'energy', 'classification']

    rw_merged = (pd.concat(rw_chunks, ignore_index=True)[col_order]
                 if rw_chunks else pd.DataFrame(columns=col_order))
    lw_merged = (pd.concat(lw_chunks, ignore_index=True)[col_order]
                 if lw_chunks else pd.DataFrame(columns=col_order))
    rl_merged = (pd.concat(rl_chunks, ignore_index=True)[col_order_leg]
                 if rl_chunks else pd.DataFrame(columns=col_order_leg))

    return rw_merged, lw_merged, rl_merged

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
                output_name ='HickeyGSD_Results.csv'
            case 3:
                # Run Kheirkhahan GSD
                gsd = KheirkhahanGSD(visual=False)
                detected_bouts = gsd.detect(imu_df, sampling_rate_hz=SAMPLING_RATE)
                output_name = 'KheirkhahanGSD_Results.csv'
            case 4: 
                # Run MacLean GSD
                gsd = MacLeanGSD()
                detected_bouts = gsd.detect(imu_df)
                output_name = 'MacLeanGSD_Results.csv'
            case 5:
                # Run Keren GSD
                gsd = KerenGSD()
                detected_bouts = gsd.detect(imu_df, sampling_rate_hz=SAMPLING_RATE)
                output_name = 'KerenGSD_Results.csv'

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

def process_leg(rl_merged: pd.DataFrame):
            
    results = []

    print(f"\n{'Subject':<35} | "
          f"{'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
    print("-" * 90)

    if rl_merged.empty:
        print(f"[rl_merged] No data found.")
        return

    imu_cols           = ['acc_pa', 'acc_ml', 'acc_is']
    classification_col = 'classification'
    energy_col         = 'energy'

    for subject, grp in rl_merged.groupby('subject', sort=True):
        imu_df         = grp[imu_cols].reset_index(drop=True)
        y_true         = grp['y_true'].to_numpy()
        condition      = grp['condition'].iloc[0]
        label          = f"{subject}"
        energy         = grp[energy_col].to_numpy()
        classification = grp[classification_col].to_numpy()

        if DEBUG:
            print('group', grp)
            print('classification is', classification)

        # classification == 3 means walking; produce a per-sample binary mask
        y_pred = (classification == 3).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        if PRINT_STATS:
            print(f"{label[:35]:<35} | "
                  f"{acc:.4f}   | {prec:.4f}   | "
                  f"{rec:.4f}   | {f1:.4f}")

        results.append({
            'Subject':   label,
            'Condition': condition,
            'Accuracy':  acc,
            'Precision': prec,
            'Recall':    rec,
            'F1':        f1,
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        })

        # # plot 
        time = np.arange(len(y_pred)) / SAMPLING_RATE
        # fig, ax = plt.subplots(figsize=(10, 4))
        # ax.plot(time, y_pred, label='Predicted')
        # ax.plot(time, y_true, label='True')
        # ax.set_xlabel('Time(s)')
        # ax.set_xlabel('Walking detected')
        # ax.legend()
        # ax.set_title(label)
        #Here's the best approach for comparing overlap with binary walking signals:
        fig, ax = plt.subplots(figsize=(12, 4))

        # Fill between for clear overlap visualization
        ax.fill_between(time, y_true, alpha=0.5, color='steelblue', label='False Negative', step='post')
        ax.fill_between(time, y_pred, alpha=0.5, color='tomato',    label='False Positive', step='post')

        # Highlight agreement region
        agreement = (np.array(y_true) == 1) & (np.array(y_pred) == 1)
        ax.fill_between(time, agreement, color='mediumseagreen', alpha=0.6, step='post', label='True Positive')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Walking detected')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No', 'Yes'])
        ax.set_ylim(-0.05, 1.3)
        ax.legend(loc='upper right')
        ax.set_title(label)
        plt.tight_layout()




    if results:
        res_df = pd.DataFrame(results)
        tp = res_df['TP'].sum()
        fp = res_df['FP'].sum()
        fn = res_df['FN'].sum()
        tn = res_df['TN'].sum()
        total = tp + fp + fn + tn

        acc_av  = (tp + tn) / total                                                          if total > 0             else 0.0
        prec_av = tp / (tp + fp)                                                             if (tp + fp) > 0         else 0.0
        rec_av  = tp / (tp + fn)                                                             if (tp + fn) > 0         else 0.0
        f1_av   = 2 * prec_av * rec_av / (prec_av + rec_av) if (prec_av + rec_av) > 0 else 0.0

        print("-" * 90)
        print(f"{'AVERAGE (Leg Overall)':<35} | "
              f"{acc_av:.4f}   | {prec_av:.4f}   | "
              f"{rec_av:.4f}   | {f1_av:.4f}")
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    all_rw: list[pd.DataFrame] = []
    all_lw: list[pd.DataFrame] = []
    all_rg: list[pd.DataFrame] = []

    for data_path in DATA_PATHS:
        dataset_name = os.path.basename(data_path.rstrip('/\\'))
        print(f"\n{'=' * 80}")
        print(f"  Merging: {dataset_name}")
        print(f"{'=' * 80}")
        rw, lw, rl = merge_all(data_path)
        # Tag each row with its source dataset for traceability
        # rw['dataset'] = dataset_name
        # lw['dataset'] = dataset_name
        rl['dataset'] = dataset_name
        # all_rw.append(rw)
        # all_lw.append(lw)
        all_rg.append(rl)

    # Pool across all datasets
    # rw_merged = pd.concat(all_rw, ignore_index=True) if all_rw else pd.DataFrame()
    # lw_merged = pd.concat(all_lw, ignore_index=True) if all_lw else pd.DataFrame()
    rl_merged = pd.concat(all_rg, ignore_index=True) if all_rg else pd.DataFrame()

    print(f"\n{'=' * 80}")
    print(f"  Running GSD on pooled data ({len(DATA_PATHS)} dataset(s))")
    print(f"{'=' * 80}")
    # process_gait(rw_merged, lw_merged, SAVE_RESULTS)
    process_leg(rl_merged)
    plt.show()