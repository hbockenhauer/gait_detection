import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from models.Kheirkhahan.GSD3_test import KheirkhahanGSD
import scipy.io as sio
from collections import deque

################# can be adjusted ##################
THRESHOLD_STILL = 0.0
DEBUG = False
####################################################

MIN_SEC_PER_WINDOW = 9

def simulate_realtime(df, sampling_rate):
    BUFFER_SIZE = 13 * sampling_rate
    TRUST_START= 2 * sampling_rate
    TRUST_END = 11 * sampling_rate
    STEP_SIZE = sampling_rate

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
        y_window = run_gsd_on_wrist(window_df, sampling_rate)   # length = BUFFER_SIZE

        # Only trust the middle portion — skip the 2s edges
        for local_i in range(TRUST_START, TRUST_END):
            global_i = window_indices[local_i]
            if np.isnan(y_pred[global_i]) or y_pred[global_i]==0:
                y_pred[global_i] = y_window[local_i]

    return y_pred

# --------------------------------------------------------------------------------
#                                  WearGait dataset 
# --------------------------------------------------------------------------------

SAMPLING_RATE_WEARGATE = 100
WEARGAIT_GAIT_KEYWORDS = ['walk', 'stair', 'gait', 'jog', 'run', 'climb']

def run_gsd_on_wrist(imu_df: pd.DataFrame, sampling_rate: float = 50) -> np.ndarray:
    """
    Run GSD on a dataframe with columns [acc_is, acc_ml, acc_pa].
    Returns y_pred array of same length.
    """
    seg_imu = imu_df[['acc_is', 'acc_ml', 'acc_pa']].copy().astype(float).reset_index(drop=True)

    y_pred = np.zeros(len(seg_imu))

    if len(seg_imu) < MIN_SEC_PER_WINDOW*sampling_rate:
        print("the issue is",len(seg_imu))
        return np.full(len(seg_imu), np.nan)

    gsd = KheirkhahanGSD(threshold_still=THRESHOLD_STILL)
    bout_result = gsd.detect(seg_imu, sampling_rate_hz=sampling_rate)

    if hasattr(bout_result, 'gs_list_') and not bout_result.gs_list_.empty:
        for _, bout_row in bout_result.gs_list_.iterrows():
            start = int(bout_row['start'])
            end   = int(bout_row['end'])
            y_pred[start:end] = 1

    return y_pred

def _pooled_metrics(subset: pd.DataFrame) -> tuple:
    tp, fp, fn, tn = subset['TP'].sum(), subset['FP'].sum(), subset['FN'].sum(), subset['TN'].sum()
    total    = tp + fp + fn + tn
    acc_av   = (tp + tn) / total          if total > 0            else 0.0
    prec_av  = tp / (tp + fp)             if (tp + fp) > 0        else 0.0
    rec_av   = tp / (tp + fn)             if (tp + fn) > 0        else 0.0
    f1_av    = 2 * prec_av * rec_av / (prec_av + rec_av) \
                                            if (prec_av + rec_av) > 0 else 0.0
    return acc_av, prec_av, rec_av, f1_av

def process_weargait(data_path: str, print_stats: bool = True, 
                     save_results: bool = False, realtime: bool = False,
                     output_file: str = "Results/WearGait_Results.csv") -> pd.DataFrame:
    files = sorted([
        f for f in os.listdir(data_path)
        if f.endswith('.csv') and (f.startswith('W') or f.startswith('N'))
    ])

    if not files:
        print(f"No WearGait CSV files found in {data_path}")
        return pd.DataFrame()

    if print_stats:
        print(f"Scanning: {data_path}\n")
        print(f"{'File':<35} | {'Wrist':<5} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
        print("-" * 75)

    results = []

    for file_name in files:
        filepath = os.path.join(data_path, file_name)
        subject  = os.path.splitext(file_name)[0]

        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception as e:
            print(f"{file_name[:25]:<25} | ERROR loading: {e}")
            continue

        if 'GeneralEvent' not in df.columns:
            print(f"{file_name[:25]:<25} | ERROR: 'GeneralEvent' column not found, skipping.")
            continue

        pattern = '|'.join(WEARGAIT_GAIT_KEYWORDS)
        y_true        = df['GeneralEvent'].str.contains(pattern, case=False, na=False).astype(int).to_numpy()
        event_labels  = df['GeneralEvent'].fillna('unknown').to_numpy()
        unique_events = sorted(df['GeneralEvent'].dropna().unique())

        wrist_configs = [
            ('LW', 'L_Wrist_Acc_X', 'L_Wrist_Acc_Y', 'L_Wrist_Acc_Z'),
            ('RW', 'R_Wrist_Acc_X', 'R_Wrist_Acc_Y', 'R_Wrist_Acc_Z'),
        ]

        for wrist_label, col_x, col_y, col_z in wrist_configs:
            if not all(c in df.columns for c in [col_x, col_y, col_z]):
                print(f"{file_name[:25]:<25} | {wrist_label} | missing acc columns, skipping.")
                continue

            wrist_df = df[[col_x, col_y, col_z]].rename(columns={
                col_x: 'acc_is', col_y: 'acc_ml', col_z: 'acc_pa',
            })

            if realtime:
                y_pred = simulate_realtime(wrist_df, SAMPLING_RATE_WEARGATE)
            else:
                y_pred = run_gsd_on_wrist(wrist_df, SAMPLING_RATE_WEARGATE)

            valid_mask = ~np.isnan(y_pred)
            if valid_mask.sum() == 0:
                print(f"{file_name[:25]:<25} | {wrist_label} | too short, skipped.")
                continue

            acc  = accuracy_score (y_true[valid_mask], y_pred[valid_mask])
            prec = precision_score(y_true[valid_mask], y_pred[valid_mask], zero_division=0)
            rec  = recall_score   (y_true[valid_mask], y_pred[valid_mask], zero_division=0)
            f1   = f1_score       (y_true[valid_mask], y_pred[valid_mask], zero_division=0)

            tp = int(np.sum((y_pred[valid_mask] == 1) & (y_true[valid_mask] == 1)))
            fp = int(np.sum((y_pred[valid_mask] == 1) & (y_true[valid_mask] == 0)))
            fn = int(np.sum((y_pred[valid_mask] == 0) & (y_true[valid_mask] == 1)))
            tn = int(np.sum((y_pred[valid_mask] == 0) & (y_true[valid_mask] == 0)))

            if print_stats:
                print(f"{f'{subject}/{wrist_label}':<35} | {wrist_label:<5} | "
                      f"{acc:.2f}   | {prec:.2f}   | {rec:.2f}   | {f1:.2f}")

            # Overall result row for this subject/wrist
            results.append({
                'row_type': 'result',
                'Subject':  f"{subject}/{wrist_label}",
                'Wrist':    wrist_label,
                'Folder':   subject,
                'Activity': 'overall',
                'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1,
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            })

            # Per-event breakdown for this subject/wrist
            for event in unique_events:
                event_mask = (event_labels == event) & valid_mask
                if event_mask.sum() == 0:
                    continue
                results.append({
                    'row_type': 'result_per_activity',
                    'Subject':  f"{subject}/{wrist_label}",
                    'Wrist':    wrist_label,
                    'Folder':   subject,
                    'Activity': event,
                    'Accuracy':  accuracy_score (y_true[event_mask], y_pred[event_mask]),
                    'Precision': precision_score(y_true[event_mask], y_pred[event_mask], zero_division=0),
                    'Recall':    recall_score   (y_true[event_mask], y_pred[event_mask], zero_division=0),
                    'F1':        f1_score       (y_true[event_mask], y_pred[event_mask], zero_division=0),
                    'TP': int(np.sum((y_pred[event_mask] == 1) & (y_true[event_mask] == 1))),
                    'FP': int(np.sum((y_pred[event_mask] == 1) & (y_true[event_mask] == 0))),
                    'FN': int(np.sum((y_pred[event_mask] == 0) & (y_true[event_mask] == 1))),
                    'TN': int(np.sum((y_pred[event_mask] == 0) & (y_true[event_mask] == 0))),
                })

    if not results:
        print("No results to summarise.")
        return pd.DataFrame()

    res_df     = pd.DataFrame(results)
    overall_df = res_df[res_df['row_type'] == 'result']
    per_act_df = res_df[res_df['row_type'] == 'result_per_activity']

    def _print_avg(label: str, subset: pd.DataFrame):
        if subset.empty:
            return
        acc_av, prec_av, rec_av, f1_av = _pooled_metrics(subset)
        print(f"{label:<40} | {'':5} | "
              f"{acc_av:.5f}   | {prec_av:.5f}   | {rec_av:.5f}   | {f1_av:.5f}")

    def _avg_row(row_type, label, wrist, activity, subset) -> dict:
        acc_av, prec_av, rec_av, f1_av = _pooled_metrics(subset)
        return {
            'row_type': row_type, 'Subject': label,
            'Wrist':    wrist,    'Folder':  '',
            'Activity': activity,
            'Accuracy': acc_av, 'Precision': prec_av,
            'Recall':   rec_av, 'F1':        f1_av,
            'TP': subset['TP'].sum(), 'FP': subset['FP'].sum(),
            'FN': subset['FN'].sum(), 'TN': subset['TN'].sum(),
        }

    avg_rows = []
    print("-" * 75)

    # Per-wrist averages (from overall rows)
    for wrist in ['RW', 'LW']:
        sub = overall_df[overall_df['Wrist'] == wrist]
        if not sub.empty:
            _print_avg(f"AVERAGE ({wrist})", sub)
            avg_rows.append(_avg_row('avg_wrist', f"{wrist} average", wrist, '', sub))

    print()

    # Per-event averages (from per-activity rows, pooled across all subjects/wrists)
    print("Per-event averages:")
    for event in sorted(per_act_df['Activity'].unique()):
        sub = per_act_df[per_act_df['Activity'] == event]
        _print_avg(f"AV(event={event})", sub)
        avg_rows.append(_avg_row('avg_activity', f"Event={event} average", '', event, sub))

    print("-" * 75)
    _print_avg("AVERAGE (Overall)", overall_df)
    avg_rows.append(_avg_row('avg_overall', 'AVERAGE (Overall)', '', '', overall_df))

    if save_results:
        avg_df    = pd.DataFrame(avg_rows)
        blank_row = pd.DataFrame([{c: '' for c in res_df.columns}])
        csv_df    = pd.concat([res_df, blank_row, avg_df], ignore_index=True)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        csv_df.to_csv(output_file, index=False)
        print(f"\nSaved → {output_file}")

    return overall_df

# --------------------------------------------------------------------------------
#                                  WISDM dataset 
# --------------------------------------------------------------------------------

SAMPLING_RATE_WISDM = 20  # adjust to actual dataset sampling rate

GAIT_LABELS_WISM = {'A', 'B', 'C'}  # walking, jogging, stairs

ACTIVITY_MAP_WISM = {
    'A': 'walking', 'B': 'jogging', 'C': 'stairs',
    'D': 'sitting', 'E': 'standing', 'F': 'typing',
    'G': 'teeth',   'H': 'soup',     'I': 'chips',
    'J': 'pasta',   'K': 'drinking', 'L': 'sandwich',
    'M': 'kicking', 'O': 'catch',    'P': 'dribbling',
    'Q': 'writing', 'R': 'clapping', 'S': 'folding',
}


def load_wismd_txt_file(filepath: str) -> pd.DataFrame | None:
    """
    Parse lines of the form:
        subject_id,activity_label,timestamp,x,y,z;
    Returns a DataFrame with columns:
        subject, activity, timestamp, acc_is, acc_ml, acc_pa, y_true
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().rstrip(';')
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 6:
                continue
            try:
                subject   = str(parts[0]).strip()
                activity  = str(parts[1]).strip()
                timestamp = int(parts[2])
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
                rows.append((subject, activity, timestamp, x, y, z))
            except ValueError:
                continue

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=['subject', 'activity', 'timestamp', 'acc_is', 'acc_ml', 'acc_pa'])
    df['y_true'] = df['activity'].isin(GAIT_LABELS_WISM).astype(int)
    return df

def process_wisdm(data_path: str, print_stats: bool = True,
                    save_results: bool = False, realtime: bool = False, 
                    output_file: str = "Results/WISDM_Results.csv") -> pd.DataFrame:
    """
    Process all .txt files in data_path.
    Each file may contain multiple subjects and activities.
    GSD is run per subject (all data for that subject treated as one recording).
    """
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.txt')])

    if not files:
        print(f"No .txt files found in {data_path}")
        return pd.DataFrame()

    # Load and concatenate all files
    chunks = []
    for file_name in files:
        df = load_wismd_txt_file(os.path.join(data_path, file_name))
        if df is not None:
            chunks.append(df)

    if not chunks:
        print("No valid data loaded.")
        return pd.DataFrame()

    all_data = pd.concat(chunks, ignore_index=True)

    if print_stats:
        print(f"Scanning: {data_path}")
        print(f"Loaded {len(all_data)} rows from {len(chunks)} file(s).")
        print(f"Subjects found: {sorted(all_data['subject'].unique())}\n")
        print(f"{'Subject':<20} | " # {'Cond':<10} | "
              f"{'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
        print("-" * 70)

    results = []

    for subject, grp in all_data.groupby('subject', sort=True):
        grp = grp.reset_index(drop=True)
        grp['activity'] = grp['activity'].map(ACTIVITY_MAP_WISM).fillna('unknown')
        y_true = grp['y_true'].to_numpy()

        # Identify condition from which activities are present
        activities_present = grp['activity'].unique()

        imu_df = grp[['acc_is', 'acc_ml', 'acc_pa']]
        if realtime: 
            y_pred = simulate_realtime(imu_df, sampling_rate=SAMPLING_RATE_WISDM)
        else:
            y_pred = run_gsd_on_wrist(imu_df, sampling_rate=SAMPLING_RATE_WISDM)

        valid_mask = ~np.isnan(y_pred)
        if valid_mask.sum() == 0:
            print(f"{str(subject):<20} | too short, skipped.")
            continue

        acc  = accuracy_score (y_true[valid_mask], y_pred[valid_mask])
        prec = precision_score(y_true[valid_mask], y_pred[valid_mask], zero_division=0)
        rec  = recall_score   (y_true[valid_mask], y_pred[valid_mask], zero_division=0)
        f1   = f1_score       (y_true[valid_mask], y_pred[valid_mask], zero_division=0)

        tp = int(np.sum((y_pred[valid_mask] == 1) & (y_true[valid_mask] == 1)))
        fp = int(np.sum((y_pred[valid_mask] == 1) & (y_true[valid_mask] == 0)))
        fn = int(np.sum((y_pred[valid_mask] == 0) & (y_true[valid_mask] == 1)))
        tn = int(np.sum((y_pred[valid_mask] == 0) & (y_true[valid_mask] == 0)))

        if print_stats:
            print(f"{str(subject):<20} |" #  {condition:<10} | 
                  f"{acc:.2f}   | {prec:.2f}   | {rec:.2f}   | {f1:.2f}")

        results.append({
            'row_type':  'result',
            'Subject':   subject,
            'Folder':    subject,
            'Accuracy':  acc, 'Precision': prec, 'Recall': rec, 'F1': f1,
            'TP': tp,    'FP': fp,    'FN': fn,    'TN': tn,
        })
        for act in activities_present:
                    act_mask = (grp['activity'] == act).to_numpy() & valid_mask
                    if act_mask.sum() == 0:
                        continue
                    results.append({
                        'row_type':  'result_per_activity',
                        'Subject':   subject,
                        'Folder':    subject,
                        # 'Condition': condition,
                        'Activity':  act,
                        'Accuracy':  accuracy_score (y_true[act_mask], y_pred[act_mask]),
                        'Precision': precision_score(y_true[act_mask], y_pred[act_mask], zero_division=0),
                        'Recall':    recall_score   (y_true[act_mask], y_pred[act_mask], zero_division=0),
                        'F1':        f1_score       (y_true[act_mask], y_pred[act_mask], zero_division=0),
                        'TP': int(np.sum((y_pred[act_mask] == 1) & (y_true[act_mask] == 1))),
                        'FP': int(np.sum((y_pred[act_mask] == 1) & (y_true[act_mask] == 0))),
                        'FN': int(np.sum((y_pred[act_mask] == 0) & (y_true[act_mask] == 1))),
                        'TN': int(np.sum((y_pred[act_mask] == 0) & (y_true[act_mask] == 0))),
                    })
    
    if not results:
        print("No results to summarise.")
        return pd.DataFrame()

    res_df        = pd.DataFrame(results)
    overall_df    = res_df[res_df['row_type'] == 'result']
    per_act_df    = res_df[res_df['row_type'] == 'result_per_activity']

    avg_rows = []
    print("-" * 70)

    # ── Summary helpers ────────────────────────────────────────────────────────
    def _print_avg(label: str, subset: pd.DataFrame):
        if subset.empty:
            return
        acc_av, prec_av, rec_av, f1_av = _pooled_metrics(subset)
        print(f"{label:<20} |" # {'':10} | "
              f"{acc_av:.5f}   | {prec_av:.5f}   | {rec_av:.5f}   | {f1_av:.5f}")

    def _avg_row(row_type, label, subset) -> dict:
        acc_av, prec_av, rec_av, f1_av = _pooled_metrics(subset)
        return {
            'row_type':  row_type,  'Subject':   label,
            'Folder':    '',     #   'Condition': condition,
            'Accuracy':  acc_av,    'Precision': prec_av,
            'Recall':    rec_av,    'F1':        f1_av,
            'TP': subset['TP'].sum(), 'FP': subset['FP'].sum(),
            'FN': subset['FN'].sum(), 'TN': subset['TN'].sum(),
        }

    avg_rows = []
    print("-" * 70)

    # Per-activity pooled summary (uses per-activity rows)
    print("Per-activity averages:")
    for act in sorted(per_act_df['Activity'].unique()):
        sub = per_act_df[per_act_df['Activity'] == act]
        _print_avg(f"AV(activity={act})", sub)
        avg_rows.append(_avg_row('avg_activity', f"Activity={act} average", sub))

    print()

    # Per-condition summary (uses overall rows)
    # for cond in sorted(overall_df['Condition'].unique()):
    #     sub = overall_df[overall_df['Condition'] == cond]
    #     _print_avg(f"AV(cond={cond})", sub)
    #     avg_rows.append(_avg_row('avg_condition', f"Cond={cond} average", cond, sub))

    print("-" * 70)
    _print_avg("AVERAGE (Overall)", overall_df)
    avg_rows.append(_avg_row('avg_overall', 'AVERAGE (Overall)', overall_df))

    if save_results:
        avg_df    = pd.DataFrame(avg_rows)
        blank_row = pd.DataFrame([{c: '' for c in res_df.columns}])
        csv_df    = pd.concat([res_df, blank_row, avg_df], ignore_index=True)
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        csv_df.to_csv(output_file, index=False)
        print(f"\nSaved → {output_file}")

    return overall_df 
    


# --------------------------------------------------------------------------------
#                                  HMP dataset 
# --------------------------------------------------------------------------------

SAMPLING_RATE_HMP = 32  # adjust to actual dataset sampling rate

GAIT_FOLDERS_HMP = {'walk', 'stairs'}  # folder names (or substrings) that count as gait


def folder_is_gait(folder_name: str) -> bool:
    """Return True if any gait keyword appears in the folder name."""
    folder_lower = folder_name.lower()
    return any(kw in folder_lower for kw in GAIT_FOLDERS_HMP)

def load_HMP_txt(filepath: str) -> pd.DataFrame | None:
    """
    Load a plain accelerometer txt file (x, y, z — one sample per row).
    Accepts comma, semicolon, tab, or space-separated values.
    Returns DataFrame with columns [acc_is, acc_ml, acc_pa] or None on failure.
    """
    try:
        # Try to sniff the separator
        df = pd.read_csv(filepath, header=None, sep=None, engine='python',
                         comment='#', skip_blank_lines=True)

        # Keep only the first three numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 3:
            return None

        out = df[numeric_cols[:3]].copy()
        out.columns = ['acc_is', 'acc_ml', 'acc_pa']
        out = out.dropna().reset_index(drop=True)
        return out.astype(float)

    except Exception as e:
        print(f"  Could not load {filepath}: {e}")
        return None

def process_HMP(data_path: str, print_stats: bool = True,
                            save_results: bool = False,
                            realtime: bool = False, 
                            output_file: str = "Results/FolderDataset_Results.csv") -> pd.DataFrame:
    """
    Dataset structure:
        data_path/
            walk_normal/       ← gait (y_true = 1)
                subject1.txt
                subject2.txt
            stairs_up/         ← gait (y_true = 1)
            sitting/           ← non-gait (y_true = 0)
    Each txt file is treated as one recording (one subject/trial).
    GSD is run on the full file as a single segment.
    """
    activity_folders = sorted([
        f for f in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, f))
    ])

    if not activity_folders:
        print(f"No subfolders found in {data_path}")
        return pd.DataFrame()

    if print_stats:
        print(f"Scanning: {data_path}")
        print(f"Activity folders: {activity_folders}\n")
        print(f"{'File':<35} | {'Activity':<20} | {'Label':>5} | "
              f"{'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
        print("-" * 95)

    results = []

    for activity_folder in activity_folders:
        folder_path = os.path.join(data_path, activity_folder)
        y_label     = 1 if folder_is_gait(activity_folder) else 0

        txt_files = sorted([
            f for f in os.listdir(folder_path)
            if f.endswith('.txt')
        ])

        for file_name in txt_files:
            filepath = os.path.join(folder_path, file_name)
            subject  = os.path.splitext(file_name)[0]

            imu_df = load_HMP_txt(filepath)
            if imu_df is None or len(imu_df) == 0:
                print(f"  {file_name:<30} | empty or unreadable, skipping.")
                continue

            # All rows in this file share the same ground-truth label
            y_true = np.full(len(imu_df), y_label, dtype=int)

            if len(imu_df) < MIN_SEC_PER_WINDOW*SAMPLING_RATE_HMP:
                if print_stats:
                    print(f"  {file_name:<30} | {activity_folder:<20} | {y_label:>5} | "
                          f"too short ({len(imu_df)} samples), skipped.")
                continue

            if realtime: 
                y_pred = simulate_realtime(
                    imu_df[['acc_is', 'acc_ml', 'acc_pa']].reset_index(drop=True),
                    sampling_rate=SAMPLING_RATE_HMP
                )
            else: 
                y_pred = np.zeros(len(imu_df))
                
                gsd = KheirkhahanGSD(threshold_still=THRESHOLD_STILL)
                bout_result = gsd.detect(
                    imu_df[['acc_is', 'acc_ml', 'acc_pa']].reset_index(drop=True),
                    sampling_rate_hz=SAMPLING_RATE_HMP
                )
                if hasattr(bout_result, 'gs_list_') and not bout_result.gs_list_.empty:
                    for _, bout_row in bout_result.gs_list_.iterrows():
                        y_pred[int(bout_row['start']):int(bout_row['end'])] = 1
            
            valid_mask = ~np.isnan(y_pred)
            if valid_mask.sum() == 0:
                print(f"{str(subject):<20} | Valid mask at 0, skipped.")
                continue

            acc  = accuracy_score (y_true[valid_mask], y_pred[valid_mask])
            prec = precision_score(y_true[valid_mask], y_pred[valid_mask], zero_division=0)
            rec  = recall_score   (y_true[valid_mask], y_pred[valid_mask], zero_division=0)
            f1   = f1_score       (y_true[valid_mask], y_pred[valid_mask], zero_division=0)

            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            tn = int(np.sum((y_pred == 0) & (y_true == 0)))

            if DEBUG:
                print(f"{f'{file_name}':<35} | {activity_folder:<20} | {y_label:>5} | "
                      f"{acc:.2f}   | {prec:.2f}   | {rec:.2f}   | {f1:.2f}")

            results.append({
                'row_type':  'result',
                'Subject':   f"{activity_folder}/{subject}",
                'Folder':    activity_folder,
                'Condition': 'gait' if y_label == 1 else 'non_gait',
                'Accuracy':  acc, 'Precision': prec, 'Recall': rec, 'F1': f1,
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            })

    if not results:
        print("No results to summarise.")
        return pd.DataFrame()

    res_df = pd.DataFrame(results)

    # ── Summary helpers ────────────────────────────────────────────────────────
    def _print_avg(label: str, subset: pd.DataFrame):
        if subset.empty:
            return
        acc_av, prec_av, rec_av, f1_av = _pooled_metrics(subset)
        print(f"{label:<35} | {'':20} | {'':5} | "
              f"{acc_av:.5f}   | {prec_av:.5f}   | {rec_av:.5f}   | {f1_av:.5f}")

    def _avg_row(row_type, label, condition, subset) -> dict:
        acc_av, prec_av, rec_av, f1_av = _pooled_metrics(subset)
        return {
            'row_type':  row_type,  'Subject':   label,
            'Folder':    '',        'Condition': condition,
            'Accuracy':  acc_av,    'Precision': prec_av,
            'Recall':    rec_av,    'F1':        f1_av,
            'TP': subset['TP'].sum(), 'FP': subset['FP'].sum(),
            'FN': subset['FN'].sum(), 'TN': subset['TN'].sum(),
        }

    avg_rows = []
    print("-" * 95)

    # Per activity-folder averages
    for folder in sorted(res_df['Folder'].unique()):
        sub = res_df[res_df['Folder'] == folder]
        _print_avg(f"AV(activity={folder})", sub)
        avg_rows.append(_avg_row('avg_activity', f"Activity={folder} average", folder, sub))

    print()

    # Per condition (gait / non_gait)
    for cond in sorted(res_df['Condition'].unique()):
        sub = res_df[res_df['Condition'] == cond]
        _print_avg(f"AV(cond={cond})", sub)
        avg_rows.append(_avg_row('avg_condition', f"Cond={cond} average", cond, sub))

    print("-" * 95)
    _print_avg("AVERAGE (Overall)", res_df)
    avg_rows.append(_avg_row('avg_overall', 'AVERAGE (Overall)', '', res_df))

    if save_results:
        avg_df    = pd.DataFrame(avg_rows)
        blank_row = pd.DataFrame([{c: '' for c in res_df.columns}])
        csv_df    = pd.concat([res_df, blank_row, avg_df], ignore_index=True)
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        csv_df.to_csv(output_file, index=False)
        print(f"\nSaved → {output_file}")

    return res_df

# --------------------------------------------------------------------------------
#                                Bioclite dataset 
# --------------------------------------------------------------------------------

SAMPLING_RATE_BIOCLITE = 50

BIOCLITE_GAIT_LABEL = 6

BIOCLITE_LABEL_MAP  = {
    0: 'Transitions/Activity Change',
    1: 'Drawing a spiral',
    2: 'Typing with a keyboard',
    3: 'Resting in a chair',
    4: 'Beating a mixture',
    5: 'Brushing teeth',
    6: 'Walking 50 meters',
}

def process_bioclite(mat_path: str, print_stats: bool = True,
                     save_results: bool = False, realtime: bool = False,
                     output_file: str = "Results/Bioclite_Results.csv") -> pd.DataFrame:
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    print(f"Loading: {mat_path}")
    mat  = sio.loadmat(mat_path, squeeze_me=True)
    Data = mat['Data_plain']
    print(f"Found {len(Data)} trial(s).\n")

    if print_stats:
        print(f"{'Trial/Participant':<30} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
        print("-" * 65)

    results = []

    for i, trial in enumerate(Data):
        try:
            acc         = trial[:, 1:4].astype(float)
            participant = int(trial[0, 7])
            act_labels  = trial[:, 8].astype(int)
        except Exception as e:
            print(f"  Trial {i+1:02d} | ERROR reading columns: {e}")
            continue

        label  = f"trial{i+1:02d}/P{participant:02d}"
        y_true = (act_labels == BIOCLITE_GAIT_LABEL).astype(int)

        seg_imu = pd.DataFrame(acc, columns=['acc_is', 'acc_ml', 'acc_pa'])
        if realtime: 
            y_pred  = simulate_realtime(seg_imu, SAMPLING_RATE_BIOCLITE)
        else:
            y_pred = run_gsd_on_wrist(seg_imu, SAMPLING_RATE_BIOCLITE)

        valid_mask = ~np.isnan(y_pred)
        if valid_mask.sum() == 0:
            if print_stats:
                print(f"{label:<30} | too short / no valid segments, skipped.")
            continue

        acc_sc = accuracy_score (y_true[valid_mask], y_pred[valid_mask])
        prec   = precision_score(y_true[valid_mask], y_pred[valid_mask], zero_division=0)
        rec    = recall_score   (y_true[valid_mask], y_pred[valid_mask], zero_division=0)
        f1     = f1_score       (y_true[valid_mask], y_pred[valid_mask], zero_division=0)

        tp = int(np.sum((y_pred[valid_mask] == 1) & (y_true[valid_mask] == 1)))
        fp = int(np.sum((y_pred[valid_mask] == 1) & (y_true[valid_mask] == 0)))
        fn = int(np.sum((y_pred[valid_mask] == 0) & (y_true[valid_mask] == 1)))
        tn = int(np.sum((y_pred[valid_mask] == 0) & (y_true[valid_mask] == 0)))

        if print_stats:
            print(f"{label:<30} | {acc_sc:.2f}   | {prec:.2f}   | {rec:.2f}   | {f1:.2f}")

        # Overall result row for this trial
        results.append({
            'row_type':      'result',
            'Subject':       label,
            'Trial':         i + 1,
            'ParticipantID': participant,
            'Activity':      'overall',
            'Accuracy':  acc_sc, 'Precision': prec, 'Recall': rec, 'F1': f1,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        })

        # Per-activity breakdown using act_labels
        for act_code in np.unique(act_labels):
            act_mask = (act_labels == act_code) & valid_mask
            if act_mask.sum() == 0:
                continue
            act_name = BIOCLITE_LABEL_MAP.get(int(act_code), f'unknown_{act_code}')
            results.append({
                'row_type':      'result_per_activity',
                'Subject':       label,
                'Trial':         i + 1,
                'ParticipantID': participant,
                'Activity':      act_name,
                'Accuracy':  accuracy_score (y_true[act_mask], y_pred[act_mask]),
                'Precision': precision_score(y_true[act_mask], y_pred[act_mask], zero_division=0),
                'Recall':    recall_score   (y_true[act_mask], y_pred[act_mask], zero_division=0),
                'F1':        f1_score       (y_true[act_mask], y_pred[act_mask], zero_division=0),
                'TP': int(np.sum((y_pred[act_mask] == 1) & (y_true[act_mask] == 1))),
                'FP': int(np.sum((y_pred[act_mask] == 1) & (y_true[act_mask] == 0))),
                'FN': int(np.sum((y_pred[act_mask] == 0) & (y_true[act_mask] == 1))),
                'TN': int(np.sum((y_pred[act_mask] == 0) & (y_true[act_mask] == 0))),
            })

    if not results:
        print("No results to summarise.")
        return pd.DataFrame()

    res_df     = pd.DataFrame(results)
    overall_df = res_df[res_df['row_type'] == 'result']
    per_act_df = res_df[res_df['row_type'] == 'result_per_activity']

    def _print_avg(label: str, subset: pd.DataFrame):
        if subset.empty:
            return
        a, p, r, f = _pooled_metrics(subset)
        print(f"{label:<40} | {a:.5f}   | {p:.5f}   | {r:.5f}   | {f:.5f}")

    def _avg_row(row_type, label, activity, subset) -> dict:
        a, p, r, f = _pooled_metrics(subset)
        return {
            'row_type': row_type, 'Subject': label,
            'Trial': '', 'ParticipantID': '', 'Activity': activity,
            'Accuracy': a, 'Precision': p, 'Recall': r, 'F1': f,
            'TP': subset['TP'].sum(), 'FP': subset['FP'].sum(),
            'FN': subset['FN'].sum(), 'TN': subset['TN'].sum(),
        }

    avg_rows = []
    print("-" * 65)

    # Per-activity averages pooled across all trials
    print("Per-activity averages:")
    for act_name in sorted(per_act_df['Activity'].unique()):
        sub = per_act_df[per_act_df['Activity'] == act_name]
        _print_avg(f"AV(activity={act_name})", sub)
        avg_rows.append(_avg_row('avg_activity', f"Activity={act_name} average", act_name, sub))

    print("-" * 65)
    _print_avg("AVERAGE (Overall)", overall_df)
    avg_rows.append(_avg_row('avg_overall', 'AVERAGE (Overall)', '', overall_df))

    if save_results:
        avg_df    = pd.DataFrame(avg_rows)
        blank_row = pd.DataFrame([{c: '' for c in res_df.columns}])
        csv_df    = pd.concat([res_df, blank_row, avg_df], ignore_index=True)
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        csv_df.to_csv(output_file, index=False)
        print(f"\nSaved → {output_file}")

    return overall_df


