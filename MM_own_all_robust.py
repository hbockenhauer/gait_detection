import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from GSD3_test import KheirkhahanGSD
# from multimob.GSD.GSD4 import MacLeanGSD
# from multimob.GSD.GSD5 import KerenGSD
# from GSD2a import HickeyGSD
import csv
from datetime import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
DATA_PATHS = [
    r"C:\Users\orlov\intern\gait_detection\QSense_data_edge",
    # r"C:\Users\orlov\intern\gait_detection\QSense_data_mixed",
    # r"C:\Users\orlov\intern\gait_detection\QSense_data"
]
# GSD_n = 3
SAMPLING_RATE = 50 
DEBUG = False; 
GAIT_CLASSES = {'walking', 'stairs'}
CONDITION_KEYWORDS = ['pockets', 'phone', 'rail', 'free', 'crutches', 'walker', 'cane']
SAVE_RESULTS = True 
PRINT_STATS = True 
MIN_SEGMENT_SAMPLES = 9*SAMPLING_RATE 

PLOT = True
OUT_FOLDER = r"C:\Users\orlov\intern\gait_detection\Plots\Robust_Kheirkhahan\edge"

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

def parse_time(t_str):
    h, m, s_ms = t_str.strip().split(':')
    s, ms = s_ms.split('.')
    return time(int(h), int(m), int(s), int(ms) * 1000)

def load_segmented(DATA_PATH, file_name) -> pd.DataFrame:
    try:
        # open the file 
        with open(os.path.join(DATA_PATH, file_name), newline='') as f:
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
        
        if DEBUG:
            print(f"Dropped {dropped_rows} rows.")
            print(f"Found {segment_id+1} segments. ")
            print(f"Kept {len(clean_rows)} rows. \n")
        df = pd.DataFrame(clean_rows)
        df = df.reset_index(drop=True)
        df['segment'] = segments
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
            rw_df = load_segmented(folder_path, 's1_1RW.txt')
            if rw_df is not None:
                # assign true values 
                if 'test' in folder.lower():
                    rw_df['y_true']    = rw_df['Label'].astype(int).to_numpy()
                else:
                    activity = folder.split('_')[0].lower()
                    rw_df['y_true'] =  np.ones(len(rw_df)) if activity in GAIT_CLASSES else np.zeros(len(rw_df))
                # lebal the conditions and the subject 
                rw_df['condition'] = condition
                rw_df['subject']   = subject
                rw_chunks.append(rw_df)
                rw_rows = len(rw_df)

        if os.path.exists(lw_path):
            lw_df = load_segmented(folder_path, 's2_2LW.txt')
            if lw_df is not None:
                # print(" in folder:", folder)
                if 'test' in folder.lower():
                    lw_df['y_true']    = lw_df['Label'].astype(int).to_numpy()
                else: 
                    activity = folder.split('_')[0].lower()
                    # print("activity is", activity)
                    lw_df['y_true'] =  np.ones(len(lw_df)) if activity in GAIT_CLASSES else np.zeros(len(lw_df))
                    # print(lw_df['y_true'])
                lw_df['condition'] = condition
                lw_df['subject']   = subject
                lw_chunks.append(lw_df)
                lw_rows = len(lw_df)
  
        if (rw_rows > 0 or lw_rows > 0) and PRINT_STATS:
            print(f"{folder[:35]:<35} | {condition:<10} | {rw_rows:>8} | {lw_rows:>8}")

    if PRINT_STATS == True:
        print("-" * 80)

    col_order = ['yyyy-MM-dd', 'HH:mm:ss.fff', 
                 'accX', 'accY', 'accZ', 
                 'segment', 'y_true', 'condition', 'subject']

    rw_merged = (pd.concat(rw_chunks, ignore_index=True)[col_order]
                 if rw_chunks else pd.DataFrame(columns=col_order))
    lw_merged = (pd.concat(lw_chunks, ignore_index=True)[col_order]
                 if lw_chunks else pd.DataFrame(columns=col_order))

    return rw_merged, lw_merged

def run_gsd_on_segment(grp) : 
    global_start_idx = grp.index[0]  # offset into the full df
    # fing the acceleration columns 
    acc_cols = [c for c in grp.columns if 'acc' in c]
    if len(acc_cols) < 3:
        print("Incorrect number of columns.")
        print(f"{len(acc_cols)} columns found instead.")
    # rename the columns and run the gsd on them
    seg_imu = grp[acc_cols[:3]].copy().astype(float) * 9.81
    seg_imu.columns = ['acc_pa', 'acc_ml', 'acc_is']
    seg_imu.reset_index(drop=True)
    
    gsd = KheirkhahanGSD(cwb=False)
    bout_result = gsd.detect(seg_imu, sampling_rate_hz=SAMPLING_RATE)
    activity_counts = gsd.get_activity(seg_imu, sampling_rate_hz=SAMPLING_RATE)

    return bout_result, activity_counts, global_start_idx

def plot_predictions(df: pd.DataFrame, activity_counts_timeline, y_pred, y_true, file_name, folder):
    time_series = pd.to_timedelta(df['HH:mm:ss.fff'].str.strip())
    # Convert to total seconds (float) for plotting
    time_per_second_sec = time_series.iloc[::SAMPLING_RATE].reset_index(drop=True).dt.total_seconds()

    total_seconds = len(time_per_second_sec)
    ac_plot = np.full(total_seconds, np.nan)
    for sec_idx, val in activity_counts_timeline.items():
        if sec_idx < total_seconds:
            ac_plot[sec_idx] = val

    all_segment_first_rows = df.groupby('segment').nth(0).index
    jump_row_indices = all_segment_first_rows[1:]
    jump_times_sec = [time_series.iloc[idx].total_seconds() for idx in jump_row_indices]


    time_all_sec = time_series.dt.total_seconds() # seconds from midnight, accurate to 2 decimals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # ── Top: y_pred and y_true ────────────────────────────────────────────────
    ax1.fill_between(time_all_sec, 0, 1, where=(y_true == 1),
                    alpha=0.3, color='green', label='Ground truth (walking)')
    ax1.plot(time_all_sec, y_pred, label='y_pred (GSD)', alpha=0.8, 
            linewidth=1, color='steelblue')

    for i, jt in enumerate(jump_times_sec):
        ax1.axvline(x=jt, color='orange', linewidth=1.0, linestyle='--', alpha=0.8,
                    label='Time gap' if i == 0 else None)

    ax1.set_ylabel('Walking (1) / Not (0)')
    ax1.set_title(f'{folder}/{file_name}')
    ax1.legend(loc='upper left')
    ax1.set_ylim(-0.1, 1.4)

    # ── Bottom: activity counts ───────────────────────────────────────────────
    ax2.fill_between(time_all_sec, 0, 1, where=(y_true == 1),
                    alpha=0.2, color='green', transform=ax2.get_xaxis_transform(),
                    label='Ground truth (walking)')
    ax2.plot(time_per_second_sec, ac_plot, label='Activity count', 
            linewidth=1, color='steelblue')

    for i, jt in enumerate(jump_times_sec):
        ax2.axvline(x=jt, color='orange', linewidth=1.0, linestyle='--', alpha=0.8,
                    label='Time gap' if i == 0 else None)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Activity count')
    ax2.legend(loc='upper left')

    # Format x-axis as HH:MM:SS
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x//3600):02d}:{int((x%3600)//60):02d}:{int(x%60):02d}"
    ))
    fig.autofmt_xdate()
    plt.tight_layout()

    out_path = os.path.join(OUT_FOLDER, f"{folder}_{file_name}_predictions.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")

    return 

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

        # imu_cols = ['acc_pa', 'acc_ml', 'acc_is']

        # Process each recording (subject folder) separately so the GSD sees
        # one continuous, coherent signal — not a mix of activities concatenated.
        for subject, grp_sub in merged_df.groupby('subject', sort=True):
            grp_sub = grp_sub.reset_index(drop=True)
            # print("subject",subject)
            # print("the grp_sub is ")
            # print(grp_sub)
            y_true    = grp_sub['y_true'].to_numpy()
            condition = grp_sub['condition'].iloc[0]
            label     = f"{subject}/{wrist_label}"

            detected_bouts = []
            y_pred = np.zeros(len(grp_sub))
            # print()
            # print(f"Im working on {subject}, {label}")
            # print("y_pred", len(y_pred))
            activity_counts_timeline = {}
            skipped_seg = 0

            for segment, grp_seg in grp_sub.groupby('segment', sort=True):
                # if segment < 5:
                #     print("segment",segment)
                #     print("the grp_seg is ")
                #     print(grp_seg)
                if len(grp_seg) < MIN_SEGMENT_SAMPLES:
                    y_pred[grp_seg.index] = np.nan
                    skipped_seg += 1
                    # print(f"skipped {skipped_seg} segments")
                    continue

                bout_result, activity_counts, global_start_idx = run_gsd_on_segment(grp_seg)
                # if segment <2: 
                #     print("bout_result", bout_result)

                global_start_sec = global_start_idx // SAMPLING_RATE
                for i, val in enumerate(activity_counts):
                    activity_counts_timeline[global_start_sec + i] = val

                if hasattr(bout_result, 'gs_list_') and not bout_result.gs_list_.empty:
                    for _, bout_row in bout_result.gs_list_.iterrows():
                        # Offset local segment indices to global df indices
                        # global_bout_start = int(bout_row['start']) + global_start_idx
                        # global_bout_end = int(bout_row['end']) + global_start_idx
                        # detected_bouts.append((global_bout_start, global_bout_end))
                        # y_pred[global_bout_start:global_bout_end] = 1
                        local_bout_start = int(bout_row['start']) + global_start_idx
                        local_bout_end   = int(bout_row['end'])   + global_start_idx
                        detected_bouts.append((local_bout_start, local_bout_end))
                        y_pred[local_bout_start:local_bout_end] = 1
            #             if segment < 2: 
            #                 print("local_bout_start", local_bout_start)
            # print("last segment was", segment)
            # print('assigned 1 to elements',len(y_pred==1))

            valid_mask = ~np.isnan(y_pred)
            acc  = accuracy_score(y_true[valid_mask], y_pred[valid_mask])
            prec = precision_score(y_true[valid_mask], y_pred[valid_mask], zero_division=0)
            rec  = recall_score(y_true[valid_mask], y_pred[valid_mask], zero_division=0)
            f1   = f1_score(y_true[valid_mask], y_pred[valid_mask], zero_division=0)

            if PLOT == True: 
                plot_predictions(grp_sub, activity_counts_timeline, y_pred, y_true, wrist_label, subject)

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

            if PRINT_STATS == True: 
                print(f"{label[:35]:<35} | {wrist_label:<5} | {condition:<10} | "
                      f"{acc:.2f}   | {prec:.2f}   | "
                      f"{rec:.2f}   | {f1:.2f}")

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
    output_name = "Results/KheirkhahanGSD_Results_no_faulty_data.csv"
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