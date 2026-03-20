import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from GSD3_test import KheirkhahanGSD
import scipy.io as sio

# --------------------------------------------------------------------------------
#                                  WearGait dataset 
# --------------------------------------------------------------------------------

SAMPLING_RATE_WEARGATE = 100
MIN_SEC_PER_WINDOW = 9 
WEARGAIT_GAIT_KEYWORDS = ['walk', 'stair', 'gait', 'jog', 'run', 'climb']

def run_gsd_on_wrist(imu_df: pd.DataFrame, sampling_rate: float = 50) -> np.ndarray:
    """
    Run GSD on a dataframe with columns [acc_is, acc_ml, acc_pa].
    Returns y_pred array of same length.
    """
    seg_imu = imu_df[['acc_is', 'acc_ml', 'acc_pa']].copy().astype(float).reset_index(drop=True)

    y_pred = np.zeros(len(seg_imu))

    if len(seg_imu) < MIN_SEC_PER_WINDOW*sampling_rate:
        return np.full(len(seg_imu), np.nan)

    gsd = KheirkhahanGSD()
    bout_result = gsd.detect(seg_imu, sampling_rate_hz=sampling_rate)

    if hasattr(bout_result, 'gs_list_') and not bout_result.gs_list_.empty:
        for _, bout_row in bout_result.gs_list_.iterrows():
            start = int(bout_row['start'])
            end   = int(bout_row['end'])
            y_pred[start:end] = 1

    return y_pred

# def run_gsd_on_segment(imu_df: pd.DataFrame) -> np.ndarray:
    """
    Run GSD on a DataFrame with columns [acc_is, acc_ml, acc_pa].
    Returns y_pred array of same length.
    """
    seg_imu = imu_df[['acc_is', 'acc_ml', 'acc_pa']].copy().astype(float).reset_index(drop=True)

    if len(seg_imu) < MIN_SEGMENT_SAMPLES:
        return np.full(len(seg_imu), np.nan)

    gsd = KheirkhahanGSD()
    bout_result = gsd.detect(seg_imu, sampling_rate_hz=SAMPLING_RATE)

    y_pred = np.zeros(len(seg_imu))
    if hasattr(bout_result, 'gs_list_') and not bout_result.gs_list_.empty:
        for _, bout_row in bout_result.gs_list_.iterrows():
            y_pred[int(bout_row['start']):int(bout_row['end'])] = 1

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

def process_weargait(data_path: str, print_stats: bool = True, save_results: bool = False,
                     output_file: str = "Results/WearGait_Results.csv") -> pd.DataFrame:
    """
    Process all WearGait CSV files in data_path.
    Each file is one subject; GSD is run on left and right wrist independently.
    Returns a results DataFrame similar to process_gait().
    """
    files = sorted([
        f for f in os.listdir(data_path)
        if f.endswith('.csv') and (f.startswith('W') or f.startswith('N'))
    ])

    if not files:
        print(f"No WearGait CSV files found in {data_path}")
        return pd.DataFrame()

    if print_stats:
        print(f"Scanning: {data_path}\n")
        print(f"{'File':<35} | {'Wrist':<5} | {'Cond':<10} | "
              f"{'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
        print("-" * 90)

    results = []

    for file_name in files:
        filepath = os.path.join(data_path, file_name)
        subject  = os.path.splitext(file_name)[0]     # e.g. "W001" or "N003"
        # W = with PD, N = neurotypical — adjust to your dataset's convention
        condition = 'PD' if file_name.startswith('W') else 'control'

        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception as e:
            print(f"{file_name[:25]:<25} | ERROR loading: {e}")
            continue

        # Build y_true from the label column
        if 'GeneralEvent' not in df.columns:
            print(f"{file_name[:25]:<25} | ERROR: 'GeneralEvent' column not found, skipping.")
            continue

        pattern = '|'.join(WEARGAIT_GAIT_KEYWORDS)
        y_true = df['GeneralEvent'].str.contains(pattern, case=False, na=False).astype(int).to_numpy()

        # Define both wrists as (label, x_col, y_col, z_col)
        wrist_configs = [
            ('LW', 'L_Wrist_Acc_X', 'L_Wrist_Acc_Y', 'L_Wrist_Acc_Z'),
            ('RW', 'R_Wrist_Acc_X', 'R_Wrist_Acc_Y', 'R_Wrist_Acc_Z'),
        ]

        for wrist_label, col_x, col_y, col_z in wrist_configs:
            required_cols = [col_x, col_y, col_z]
            if not all(c in df.columns for c in required_cols):
                print(f"{file_name[:25]:<25} | {wrist_label} | missing acc columns, skipping.")
                continue

            wrist_df = df[[col_x, col_y, col_z]].rename(columns={
                col_x: 'acc_is',
                col_y: 'acc_ml',
                col_z: 'acc_pa',
            })

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
                print(f"{f'{subject}/{wrist_label}':<35} | {wrist_label:<5} | {condition:<10} | "
                      f"{acc:.2f}   | {prec:.2f}   | {rec:.2f}   | {f1:.2f}")

            results.append({
                'row_type':  'result',
                'Subject':   f"{subject}/{wrist_label}",
                'Wrist':     wrist_label,
                'Folder':    subject,
                'Condition': condition,
                'Accuracy':  acc,
                'Precision': prec,
                'Recall':    rec,
                'F1':        f1,
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            })

    if not results:
        print("No results to summarise.")
        return pd.DataFrame()

    res_df = pd.DataFrame(results)

    # ── Summary helper (same pooled-TP logic as process_gait) ──────────────


    def _print_avg(label: str, subset: pd.DataFrame):
        if subset.empty:
            return
        acc_av, prec_av, rec_av, f1_av = _pooled_metrics(subset)
        print(f"{label:<35} | {'':5} | {'':10} | "
              f"{acc_av:.5f}   | {prec_av:.5f}   | {rec_av:.5f}   | {f1_av:.5f}")

    def _avg_row(row_type, label, wrist, condition, subset) -> dict:
        acc_av, prec_av, rec_av, f1_av = _pooled_metrics(subset)
        return {
            'row_type':  row_type,
            'Subject':   label,
            'Wrist':     wrist,
            'Folder':    '',
            'Condition': condition,
            'Accuracy':  acc_av, 'Precision': prec_av,
            'Recall':    rec_av, 'F1':        f1_av,
            'TP': subset['TP'].sum(), 'FP': subset['FP'].sum(),
            'FN': subset['FN'].sum(), 'TN': subset['TN'].sum(),
        }

    avg_rows = []

    print("-" * 90)
    for wrist in ['RW', 'LW']:
        sub = res_df[res_df['Wrist'] == wrist]
        if not sub.empty:
            _print_avg(f"AVERAGE ({wrist})", sub)
            avg_rows.append(_avg_row('avg_wrist', f"{wrist} average", wrist, '', sub))

    for cond in sorted(res_df['Condition'].unique()):
        sub = res_df[res_df['Condition'] == cond]
        _print_avg(f"AV(cond={cond})", sub)
        avg_rows.append(_avg_row('avg_condition', f"Cond={cond} average", '', cond, sub))

    print("-" * 90)
    _print_avg("AVERAGE (Overall)", res_df)
    avg_rows.append(_avg_row('avg_overall', 'AVERAGE (Overall)', '', '', res_df))

    if save_results:
        avg_df    = pd.DataFrame(avg_rows)
        blank_row = pd.DataFrame([{c: '' for c in res_df.columns}])
        csv_df    = pd.concat([res_df, blank_row, avg_df], ignore_index=True)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        csv_df.to_csv(output_file, index=False)
        print(f"\nSaved → {output_file}")

    return res_df

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
                    save_results: bool = False,
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
        print(f"{'Subject':<20} | {'Cond':<10} | "
              f"{'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
        print("-" * 70)

    results = []

    for subject, grp in all_data.groupby('subject', sort=True):
        grp = grp.reset_index(drop=True)
        y_true = grp['y_true'].to_numpy()

        # Identify condition from which activities are present
        activities_present = grp['activity'].unique()
        has_gait     = any(a in GAIT_LABELS_WISM for a in activities_present)
        has_non_gait = any(a not in GAIT_LABELS_WISM for a in activities_present)
        if has_gait and has_non_gait:
            condition = 'mixed'
        elif has_gait:
            condition = 'gait_only'
        else:
            condition = 'non_gait'

        imu_df = grp[['acc_is', 'acc_ml', 'acc_pa']]
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
            print(f"{str(subject):<20} | {condition:<10} | "
                  f"{acc:.2f}   | {prec:.2f}   | {rec:.2f}   | {f1:.2f}")

        results.append({
            'row_type':  'result',
            'Subject':   subject,
            'Folder':    subject,
            'Condition': condition,
            'Accuracy':  acc, 'Precision': prec, 'Recall': rec, 'F1': f1,
            'TP': tp,    'FP': fp,    'FN': fn,    'TN': tn,
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
        print(f"{label:<20} | {'':10} | "
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
    print("-" * 70)

    for cond in sorted(res_df['Condition'].unique()):
        sub = res_df[res_df['Condition'] == cond]
        _print_avg(f"AV(cond={cond})", sub)
        avg_rows.append(_avg_row('avg_condition', f"Cond={cond} average", cond, sub))

    print("-" * 70)
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
                            output_file: str = "Results/FolderDataset_Results.csv") -> pd.DataFrame:
    """
    Dataset structure:
        data_path/
            walk_normal/       ← gait (y_true = 1)
                subject1.txt
                subject2.txt
                ...
            stairs_up/         ← gait (y_true = 1)
                ...
            sitting/           ← non-gait (y_true = 0)
                ...

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

            gsd = KheirkhahanGSD()
            bout_result = gsd.detect(
                imu_df[['acc_is', 'acc_ml', 'acc_pa']].reset_index(drop=True),
                sampling_rate_hz=SAMPLING_RATE_HMP
            )

            y_pred = np.zeros(len(imu_df))
            if hasattr(bout_result, 'gs_list_') and not bout_result.gs_list_.empty:
                for _, bout_row in bout_result.gs_list_.iterrows():
                    y_pred[int(bout_row['start']):int(bout_row['end'])] = 1

            acc  = accuracy_score (y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score   (y_true, y_pred, zero_division=0)
            f1   = f1_score       (y_true, y_pred, zero_division=0)

            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            tn = int(np.sum((y_pred == 0) & (y_true == 0)))

            if print_stats:
                print(f"{f'{activity_folder}/{file_name}':<35} | {activity_folder:<20} | {y_label:>5} | "
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
                     save_results: bool = False,
                     output_file: str = "Results/Bioclite_Results.csv") -> pd.DataFrame:
    """
    Process BIOCLITE .mat file (Data_plain format).

    Data_plain is a (N,) array where each entry is one trial with columns:
        0:      ts_ms         — timestamp in milliseconds
        1:4     acc_x/y/z     — accelerometer (m/s², 50 Hz)
        4:7     gyr_x/y/z     — gyroscope (not used here)
        7:      participant   — participant ID
        8:      activity      — activity label (see BIOCLITE_LABEL_MAP)

    Gaps in the timestamp are detected and split into sub-segments,
    matching the logic of the original evaluate_bioclite() code.
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    print(f"Loading: {mat_path}")
    mat  = sio.loadmat(mat_path, squeeze_me=True)
    Data = mat['Data_plain']
    print(f"Found {len(Data)} trial(s).\n")

    if print_stats:
        print(f"{'Trial/Participant':<30} | {'Condition':<10} | "
              f"{'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
        print("-" * 75)

    results = []

    for i, trial in enumerate(Data):
        try:
            ts_ms       = trial[:, 0].astype(float)
            acc         = trial[:, 1:4].astype(float)   # x, y, z
            participant = int(trial[0, 7])
            act_labels  = trial[:, 8].astype(int)
        except Exception as e:
            print(f"  Trial {i+1:02d} | ERROR reading columns: {e}")
            continue

        # From the Zenodo page: 24 PD patients + 16 healthy (40 total)
        # Adjust this boundary once you know which IDs are which group
        condition = 'PD' if participant <= 24 else 'healthy'
        label     = f"trial{i+1:02d}/P{participant:02d}"

        y_true = (act_labels == BIOCLITE_GAIT_LABEL).astype(int)

        # ── Single segment per trial ────────────────────────────────────────
        seg_imu = pd.DataFrame(acc, columns=['acc_is', 'acc_ml', 'acc_pa'])
        y_pred  = run_gsd_on_wrist(seg_imu, SAMPLING_RATE_BIOCLITE)

        valid_mask = ~np.isnan(y_pred)
        if valid_mask.sum() == 0:
            if print_stats:
                print(f"{label:<30} | {condition:<10} | too short / no valid segments, skipped.")
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
            print(f"{label:<30} | {condition:<10} | "
                  f"{acc_sc:.2f}   | {prec:.2f}   | {rec:.2f}   | {f1:.2f}")

        results.append({
            'row_type':      'result',
            'Subject':       label,
            'Trial':         i + 1,
            'ParticipantID': participant,
            'Condition':     condition,
            'Accuracy':  acc_sc, 'Precision': prec, 'Recall': rec, 'F1': f1,
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
        a, p, r, f = _pooled_metrics(subset)
        print(f"{label:<30} | {'':10} | "
              f"{a:.5f}   | {p:.5f}   | {r:.5f}   | {f:.5f}")

    def _avg_row(row_type, label, condition, subset) -> dict:
        a, p, r, f = _pooled_metrics(subset)
        return {
            'row_type': row_type, 'Subject': label,
            'Trial': '', 'ParticipantID': '', 'Condition': condition,
            'Accuracy': a, 'Precision': p, 'Recall': r, 'F1': f,
            'TP': subset['TP'].sum(), 'FP': subset['FP'].sum(),
            'FN': subset['FN'].sum(), 'TN': subset['TN'].sum(),
        }

    avg_rows = []
    print("-" * 75)

    for cond in sorted(res_df['Condition'].unique()):
        sub = res_df[res_df['Condition'] == cond]
        _print_avg(f"AV(cond={cond})", sub)
        avg_rows.append(_avg_row('avg_condition', f"Cond={cond} average", cond, sub))

    print("-" * 75)
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