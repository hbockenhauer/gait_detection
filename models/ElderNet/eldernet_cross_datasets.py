"""Cross-dataset evaluation using the ElderNet model.

Evaluates WISDM, WearGait, HMP, BIOCLITE, QSense variants, and Free-Living,
mirroring the StrokeNet cross-dataset evaluation but with ElderNet parameters:
  - 30 Hz target sampling rate
  - 300-sample windows (10 s)
  - 30-sample stride (1 s)
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import signal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    PLOTS_DIR as OUTPUT_PLOTS_DIR,
    RESULTS_DIR,
)
from models.ElderNet.eldernet_WearGait import load_weargait_data, detect_sampling_rate
from models.ElderNet.eldernet_ADL import extract_subject_id_and_timestamp
from utils.hub_utils import safe_hub_load
from utils import plot_ROC_PR

# ============================================================
# CONFIGURATION
# ============================================================

ADL_PATH = HMP_PATH
WEARGAIT_PD_PATH = WEARGAIT_PD
WEARGAIT_CTRL_PATH = WEARGAIT_CTRL
QSENSE_PATHS = [QSENSE_DATA, QSENSE_EDGE, QSENSE_MIXED, QSENSE_CLINIC]
FREE_LIVING_PATH = FREELIVING_PATH
REPO_NAME = 'yonbrand/ElderNet'
RESULTS_DIR = os.path.join(RESULTS_DIR, 'ElderNet')

# ElderNet native parameters (pre-trained at 30 Hz with 300-sample windows)
TARGET_FS = 30.0
WINDOW_SIZE = 300      # 10 s at 30 Hz
STEP_SIZE = 30         # 1 s stride at 30 Hz
GAP_THRESHOLD = 0.1    # seconds — time-domain gap detection

CONF_THRESH = 0.65

WISDM_GAIT_CODES = {'A', 'B', 'C'}
WEARGAIT_PATTERNS = ['walk', 'jog', 'run', 'stair', 'climb', 'freewalk', 'gait']
HMP_GAIT_ACTIVITIES = {'Walk', 'Climb_stairs', 'Descend_stairs'}
WISDM_GAIT_ACTIVITIES = {'Walk', 'Jog', 'Stairs'}
QSENSE_GAIT_ACTIVITIES = {'walking', 'stairs'}
FREE_LIVING_DATASET_NAME = 'free_living'

BIOCLITE_GAIT_LABEL = 6
BIOCLITE_LABEL_MAP = {
    0: 'Transitions/Activity Change',
    1: 'Drawing a spiral',
    2: 'Typing with a keyboard',
    3: 'Resting in a chair',
    4: 'Beating a mixture',
    5: 'Brushing teeth',
    6: 'Walking 50 meters',
}

ACTIVITY_MAP = {
    'A': 'Walk', 'B': 'Jog', 'C': 'Stairs', 'D': 'Sit', 'E': 'Stand',
    'F': 'Type', 'G': 'Teeth', 'H': 'Soup', 'I': 'Chips', 'J': 'Pasta',
    'K': 'Drink', 'L': 'Sandwich', 'M': 'Kicking', 'O': 'Catch',
    'P': 'Dribbling', 'Q': 'Writing', 'R': 'Clapping', 'S': 'Folding',
}

# ============================================================
# MODEL LOADING
# ============================================================

def load_eldernet_model():
    """Load base ElderNet fine-tuned model from Torch Hub."""
    model = safe_hub_load(REPO_NAME, 'eldernet_ft', trust_repo=True)
    model.eval()
    return model


# ============================================================
# INFERENCE / METRICS
# ============================================================

def run_inference(model, windows_np, device):
    wins = torch.FloatTensor(windows_np).to(device)
    with torch.no_grad():
        logits = model(wins)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if y_true.sum() == 0:
        return 0.0, 0.0, 0.0, accuracy_score(y_true, y_pred), cm
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average='binary', zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return prec, rec, f1, acc, cm


def update_activity_confusions(by_act, y_true, y_pred, win_activities):
    for act_name in np.unique(win_activities):
        mask = (win_activities == act_name)
        if mask.sum() == 0:
            continue
        by_act[act_name][0] += int(((y_pred[mask] == 1) & (y_true[mask] == 1)).sum())
        by_act[act_name][1] += int(((y_pred[mask] == 1) & (y_true[mask] == 0)).sum())
        by_act[act_name][2] += int(((y_pred[mask] == 0) & (y_true[mask] == 1)).sum())
        by_act[act_name][3] += int(((y_pred[mask] == 0) & (y_true[mask] == 0)).sum())


def print_by_activity_table(by_act, dataset_name):
    print(f"\n{dataset_name} - By activity (pooled across all subjects):")
    print(f"  {'Activity':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} "
          f"{'Accuracy':>10} {'Windows':>10}")
    print(f"  {'-'*76}")
    for act_name, (tp, fp, fn, tn) in sorted(by_act.items()):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        a = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        print(f"  {str(act_name):<22} {p:>10.3f} {r:>10.3f} {f:>10.3f} "
              f"{a:>10.3f} {tp + fp + fn + tn:>10}")


# ============================================================
# GAP / DISCONTINUITY HELPERS
# ============================================================

def get_discontinuity_times(times, gap_threshold=GAP_THRESHOLD):
    """Return timestamps where consecutive time-gaps exceed the threshold."""
    if times is None or len(times) < 2:
        return np.array([])
    times = np.asarray(times, dtype=float)
    dt = np.diff(times)
    gap_idx = np.where(dt > gap_threshold)[0] + 1
    return times[gap_idx]


def insert_nan_breaks(x, y_values, discontinuity_times):
    """Insert NaNs into series so plotted lines break at discontinuities."""
    x_arr = np.asarray(x, dtype=float)
    y_arrs = [np.asarray(y, dtype=float) for y in y_values]

    if discontinuity_times is None or len(x_arr) < 2 or len(discontinuity_times) == 0:
        return x_arr, y_arrs

    gap_times = np.asarray(discontinuity_times, dtype=float)
    gap_times = gap_times[np.isfinite(gap_times)]
    if len(gap_times) == 0:
        return x_arr, y_arrs

    break_positions = set()
    for gap_t in gap_times:
        idx = int(np.searchsorted(x_arr, gap_t, side='left'))
        if 0 < idx < len(x_arr):
            break_positions.add(idx)

    if not break_positions:
        return x_arr, y_arrs

    x_list = x_arr.tolist()
    y_lists = [a.tolist() for a in y_arrs]
    for pos in sorted(break_positions, reverse=True):
        x_list.insert(pos, np.nan)
        for y_list in y_lists:
            y_list.insert(pos, np.nan)

    return np.asarray(x_list, dtype=float), [np.asarray(y, dtype=float) for y in y_lists]


def is_git_lfs_pointer(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            line1 = f.readline().strip()
            line2 = f.readline().strip()
        return (
            line1 == 'version https://git-lfs.github.com/spec/v1'
            and line2.startswith('oid sha256:')
        )
    except OSError:
        return False


def _col(df, name_candidates, fallback_idx):
    for name in name_candidates:
        if name in df.columns:
            return df[name]
    if fallback_idx is not None and fallback_idx < len(df.columns):
        return df.iloc[:, fallback_idx]
    return None


# ============================================================
# WINDOWING (ElderNet: 300 samples, 30-sample stride, 30 Hz)
# ============================================================

def extract_windows(times, acc_data, labels, activities):
    """Gap-aware windowing using ElderNet-native parameters (300 samples @ 30 Hz)."""
    dt = np.diff(times)
    gap_idx = np.where(dt > GAP_THRESHOLD)[0] + 1
    bounds = np.concatenate([[0], gap_idx, [len(times)]])

    windows, targets, win_times, win_activities = [], [], [], []

    for k in range(len(bounds) - 1):
        seg_start = bounds[k]
        seg_end = bounds[k + 1]
        if (seg_end - seg_start) < WINDOW_SIZE:
            continue
        for i in range(seg_start + WINDOW_SIZE, seg_end, STEP_SIZE):
            win = acc_data[i - WINDOW_SIZE:i]
            lab_win = labels[i - WINDOW_SIZE:i]
            act_win = activities[i - WINDOW_SIZE:i]
            windows.append(win.T)
            targets.append(int(np.mean(lab_win) > 0.5))
            win_times.append(times[i - 1])
            unique, counts = np.unique(act_win, return_counts=True)
            win_activities.append(str(unique[np.argmax(counts)]))

    if not windows:
        return None, None, None, None

    return (
        np.array(windows, dtype=np.float32),
        np.array(targets),
        np.array(win_times),
        np.array(win_activities, dtype=object),
    )


# ============================================================
# QSense FILE LOADER (returns native 50 Hz; caller resamples)
# ============================================================

def _load_qsense_file(filepath, folder_name):
    df = pd.read_csv(filepath, sep=None, engine='python').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(
        df.iloc[:, 0].astype(str) + ' ' + df.iloc[:, 1].astype(str),
        errors='coerce',
    )
    df = df.dropna(subset=['datetime']).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError('No valid datetime rows')

    running_max = df['datetime'].iloc[0]
    keep = []
    for t in df['datetime']:
        if t < running_max:
            keep.append(False)
        else:
            keep.append(True)
            running_max = t
    df = df[keep].reset_index(drop=True)

    dt = df['datetime'].diff()
    jump_idx = dt[abs(dt) > pd.Timedelta(days=100)].index
    for idx in jump_idx:
        false_gap = dt[idx] - pd.Timedelta(seconds=1 / 50)
        df.loc[idx:, 'datetime'] = df.loc[idx:, 'datetime'] - false_gap
        dt = df['datetime'].diff()

    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.drop_duplicates(subset='datetime', keep='first').reset_index(drop=True)
    df['time_sec'] = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds()

    acc_x = _col(df, ['ax', 'accX', 'AccX'], 5)
    acc_y = _col(df, ['ay', 'accY', 'AccY'], 6)
    acc_z = _col(df, ['az', 'accZ', 'AccZ'], 7)
    if acc_x is None or acc_y is None or acc_z is None:
        raise ValueError('Missing accelerometer columns in QSense file')

    acc = np.column_stack([
        pd.to_numeric(acc_x, errors='coerce').values,
        pd.to_numeric(acc_y, errors='coerce').values,
        pd.to_numeric(acc_z, errors='coerce').values,
    ])

    label_series = None
    for candidate in ['Label', 'label']:
        if candidate in df.columns:
            label_series = pd.to_numeric(df[candidate], errors='coerce').fillna(0).astype(int).values
            break
    if label_series is None:
        is_gait_folder = any(tag in folder_name.lower() for tag in QSENSE_GAIT_ACTIVITIES)
        label_series = np.ones(len(df), dtype=int) if is_gait_folder else np.zeros(len(df), dtype=int)

    times = df['time_sec'].values.astype(float)
    valid = np.isfinite(times) & np.isfinite(acc).all(axis=1)
    times, acc, label_series = times[valid], acc[valid], label_series[valid]

    folder_activity = folder_name.split('_')[0]
    activities = np.array([folder_activity] * len(times), dtype=object)
    return times, acc, label_series, activities


# ============================================================
# DATASET EVALUATORS
# ============================================================

def evaluate_wisdm(model, device):
    print("\n" + "=" * 60)
    print("EVALUATING: WISDM (watch accelerometer)")
    print("=" * 60)

    files = sorted([f for f in os.listdir(WISDM_PATH) if f.endswith('.txt')])
    print(f"Found {len(files)} subject files")

    results = []
    total_tp = total_fp = total_fn = total_tn = 0
    by_act = defaultdict(lambda: [0, 0, 0, 0])

    for fname in files:
        fpath = os.path.join(WISDM_PATH, fname)
        subject = fname.replace('.txt', '')

        try:
            df = pd.read_csv(fpath, header=None, names=['user', 'activity', 'ts', 'x', 'y', 'z'])
            df['z'] = df['z'].astype(str).str.replace(';', '').astype(float)
            df = df.dropna(subset=['x', 'y', 'z'])

            acc_raw = df[['x', 'y', 'z']].values
            labels_raw = df['activity'].values

            # WISDM is 20 Hz → resample to 30 Hz
            n_new = int(len(acc_raw) * (TARGET_FS / 20.0))
            acc_rs = signal.resample(acc_raw, n_new)
            idx_map = np.linspace(0, len(labels_raw) - 1, n_new).astype(int)
            labels_rs = labels_raw[idx_map]
            times = np.linspace(0.0, n_new / TARGET_FS, n_new)

            y_binary = np.array([1 if l in WISDM_GAIT_CODES else 0 for l in labels_rs])
            activities_arr = np.array(
                [ACTIVITY_MAP.get(str(l), str(l)) for l in labels_rs], dtype=object
            )

            wins_np, y_true, win_times, win_activities = extract_windows(
                times, acc_rs, y_binary, activities_arr
            )
            if wins_np is None:
                continue

            probs = run_inference(model, wins_np, device)
            y_pred = (probs > CONF_THRESH).astype(int)
            update_activity_confusions(by_act, y_true, y_pred, win_activities)

            prec, rec, f1, acc, cm = compute_metrics(y_true, y_pred)
            total_tn += cm[0, 0]; total_fp += cm[0, 1]
            total_fn += cm[1, 0]; total_tp += cm[1, 1]

            print(f"  {subject:<20} | Prec={prec:.3f}  Rec={rec:.3f}  "
                  f"F1={f1:.3f}  Acc={acc:.3f}  [gait={y_true.sum()}/{len(y_true)} windows]")

            results.append({
                'subject': subject, 'dataset': 'WISDM',
                'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc,
                'confusion_matrix': cm.tolist(),
                'probs': probs, 'y_true': y_true, 'y_pred': y_pred,
                'win_times': win_times, 'win_activities': win_activities,
            })

        except Exception as e:
            print(f"  Error in {fname}: {e}")

    g_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    g_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    g_f1   = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0
    wisdm_total = total_tp + total_tn + total_fp + total_fn
    g_acc  = (total_tp + total_tn) / wisdm_total if wisdm_total > 0 else 0
    print(f"\nWISDM GLOBAL: Prec={g_prec:.3f} | Rec={g_rec:.3f} | F1={g_f1:.3f} | Acc={g_acc:.3f}")
    print_by_activity_table(by_act, "WISDM")
    return results, {'precision': g_prec, 'recall': g_rec, 'f1': g_f1, 'accuracy': g_acc}


def evaluate_weargait(model, device):
    print("\n" + "=" * 60)
    print("EVALUATING: WearGait-PD & CTRL")
    print("=" * 60)

    csv_PD = sorted([
        os.path.join(WEARGAIT_PD_PATH, f)
        for f in os.listdir(WEARGAIT_PD_PATH)
        if f.lower().endswith('.csv') and 'freewalk' in f.lower()
    ])
    csv_CTRL = sorted([
        os.path.join(WEARGAIT_CTRL_PATH, f)
        for f in os.listdir(WEARGAIT_CTRL_PATH)
        if f.lower().endswith('.csv') and 'freewalk' in f.lower()
    ])
    csv_files = csv_PD + csv_CTRL
    print(f"Found {len(csv_files)} CSV files ({len(csv_PD)} PD, {len(csv_CTRL)} CTRL)")

    results = []
    total_tp = total_fp = total_fn = total_tn = 0
    by_act = defaultdict(lambda: [0, 0, 0, 0])
    lfs_skipped = 0

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        subject = fname.replace('.csv', '')

        if is_git_lfs_pointer(fpath):
            lfs_skipped += 1
            print(f"  Skipping {fname}: Git LFS pointer (run 'git lfs pull')")
            continue

        for wrist in ['right', 'left']:
            try:
                df = load_weargait_data(fpath, wrist=wrist)
                original_fs = detect_sampling_rate(df['time'].values)
                t = df['time'].values
                acc_raw = df[['acc_x', 'acc_y', 'acc_z']].values
                activities_raw = df['activity'].values

                if abs(original_fs - TARGET_FS) > 0.5:
                    n_new = int((t[-1] - t[0]) * TARGET_FS) + 1
                    new_t = np.linspace(t[0], t[-1], n_new)
                    acc_rs = np.column_stack([
                        np.interp(new_t, t, acc_raw[:, 0]),
                        np.interp(new_t, t, acc_raw[:, 1]),
                        np.interp(new_t, t, acc_raw[:, 2]),
                    ])
                    idx_map = np.linspace(0, len(activities_raw) - 1, n_new).astype(int)
                    activities_rs = activities_raw[idx_map]
                    times_rs = new_t
                else:
                    acc_rs = acc_raw
                    activities_rs = activities_raw
                    times_rs = t

                y_binary = np.array([
                    1 if any(p in str(a).lower() for p in WEARGAIT_PATTERNS) else 0
                    for a in activities_rs
                ])

                wins_np, y_true, win_times, win_activities = extract_windows(
                    times_rs, acc_rs, y_binary, activities_rs
                )
                if wins_np is None:
                    continue

                probs = run_inference(model, wins_np, device)
                y_pred = (probs > CONF_THRESH).astype(int)
                update_activity_confusions(by_act, y_true, y_pred, win_activities)

                prec, rec, f1, acc, cm = compute_metrics(y_true, y_pred)
                total_tn += cm[0, 0]; total_fp += cm[0, 1]
                total_fn += cm[1, 0]; total_tp += cm[1, 1]

                print(f"  {subject:<20} {wrist:<6} | Prec={prec:.3f}  Rec={rec:.3f}  "
                      f"F1={f1:.3f}  Acc={acc:.3f}  [gait={y_true.sum()}/{len(y_true)} windows]")

                results.append({
                    'subject': subject, 'dataset': 'WearGait', 'wrist': wrist,
                    'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc,
                    'confusion_matrix': cm.tolist(),
                    'probs': probs, 'y_true': y_true, 'y_pred': y_pred,
                    'win_times': win_times, 'win_activities': win_activities,
                })

            except Exception as e:
                print(f"  Error in {fname} ({wrist}): {e}")

    wg_total = total_tp + total_tn + total_fp + total_fn
    g_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    g_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    g_f1   = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0
    g_acc  = (total_tp + total_tn) / wg_total if wg_total > 0 else 0
    print(f"\nWearGait GLOBAL: Prec={g_prec:.3f} | Rec={g_rec:.3f} | F1={g_f1:.3f} | Acc={g_acc:.3f}")
    if lfs_skipped:
        print(f"  Note: {lfs_skipped} WearGait files were Git LFS pointers and were skipped.")
    if wg_total == 0:
        print("  No valid WearGait windows were evaluated.")
    elif by_act:
        print_by_activity_table(by_act, "WearGait")
    return results, {'precision': g_prec, 'recall': g_rec, 'f1': g_f1, 'accuracy': g_acc}


def evaluate_hmp(model, device):
    print("\n" + "=" * 60)
    print("EVALUATING: HMP Dataset (ADL)")
    print("=" * 60)

    SOURCE_FS = 32.0

    categories = sorted([
        d for d in os.listdir(ADL_PATH)
        if os.path.isdir(os.path.join(ADL_PATH, d)) and '_MODEL' not in d
    ])

    subject_files = {}
    for cat in categories:
        for f in glob.glob(os.path.join(ADL_PATH, cat, '*.txt')):
            subj_id, ts = extract_subject_id_and_timestamp(os.path.basename(f))
            if subj_id and ts:
                subject_files.setdefault(subj_id, []).append((ts, cat, f))

    print(f"Found {len(subject_files)} subjects across {len(categories)} activities")

    results = []
    total_tp = total_fp = total_fn = total_tn = 0
    by_act = defaultdict(lambda: [0, 0, 0, 0])

    for subj_id in sorted(subject_files.keys()):
        files_sorted = sorted(subject_files[subj_id])
        all_data, all_labels, all_activities = [], [], []

        for ts, cat, f in files_sorted:
            try:
                raw = np.loadtxt(f)
                raw = np.asarray(raw, dtype=float)
                if raw.ndim == 1:
                    raw = raw.reshape(-1, 3)
                if raw.shape[1] < 3:
                    continue
                raw = raw[:, :3]
                if len(raw) < 150:
                    continue

                # HMP manual conversion: map [0..63] to [-14.709..+14.709], then median filter.
                data = -14.709 + (raw / 63.0) * (2 * 14.709)
                data = signal.medfilt(data, kernel_size=(3, 1))

                new_len = int(len(data) * (TARGET_FS / SOURCE_FS))
                data = signal.resample(data, new_len)
                label = 1 if cat in HMP_GAIT_ACTIVITIES else 0
                all_data.append(data)
                all_labels.extend([label] * len(data))
                all_activities.extend([cat] * len(data))
            except Exception:
                continue

        if not all_data:
            continue

        concat_data = np.vstack(all_data)
        concat_labels = np.array(all_labels)
        concat_activities = np.array(all_activities, dtype=object)
        times = np.linspace(0.0, len(concat_data) / TARGET_FS, len(concat_data))

        if len(concat_data) < WINDOW_SIZE:
            continue

        wins_np, y_true, win_times, win_activities = extract_windows(
            times, concat_data, concat_labels, concat_activities
        )
        if wins_np is None:
            continue

        probs = run_inference(model, wins_np, device)
        y_pred = (probs > CONF_THRESH).astype(int)
        update_activity_confusions(by_act, y_true, y_pred, win_activities)

        prec, rec, f1, acc, cm = compute_metrics(y_true, y_pred)
        total_tn += cm[0, 0]; total_fp += cm[0, 1]
        total_fn += cm[1, 0]; total_tp += cm[1, 1]

        print(f"  {subj_id:<10} | Prec={prec:.3f}  Rec={rec:.3f}  "
              f"F1={f1:.3f}  Acc={acc:.3f}  [gait={y_true.sum()}/{len(y_true)} windows]")

        results.append({
            'subject': subj_id, 'dataset': 'HMP',
            'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc,
            'confusion_matrix': cm.tolist(),
            'probs': probs, 'y_true': y_true, 'y_pred': y_pred,
            'win_times': win_times, 'win_activities': win_activities,
        })

    hmp_total = total_tp + total_tn + total_fp + total_fn
    g_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    g_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    g_f1   = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0
    g_acc  = (total_tp + total_tn) / hmp_total if hmp_total > 0 else 0
    print(f"\nHMP GLOBAL: Prec={g_prec:.3f} | Rec={g_rec:.3f} | F1={g_f1:.3f} | Acc={g_acc:.3f}")
    print_by_activity_table(by_act, "HMP")
    return results, {'precision': g_prec, 'recall': g_rec, 'f1': g_f1, 'accuracy': g_acc}


def evaluate_bioclite(model, device):
    print("\n" + "=" * 60)
    print("EVALUATING: BIOCLITE Free-Living Activities")
    print("=" * 60)

    import scipy.io
    mat = scipy.io.loadmat(BIOCLITE_PATH, squeeze_me=True)
    Data = mat['Data_plain']
    print(f"Found {len(Data)} subjects")

    SOURCE_FS_BIOCLITE = 50.0

    results = []
    total_tp = total_fp = total_fn = total_tn = 0

    for i, trial in enumerate(Data):
        try:
            ts_ms = trial[:, 0].astype(float)
            acc = trial[:, 1:4].astype(float)
            participant = int(trial[0, 7])
            act_labels = trial[:, 8].astype(int)

            times_orig = (ts_ms - ts_ms[0]) / 1000.0
            y_binary = (act_labels == BIOCLITE_GAIT_LABEL).astype(int)

            # Resample 50 Hz → 30 Hz
            n_new = int(len(acc) * (TARGET_FS / SOURCE_FS_BIOCLITE))
            acc_rs = signal.resample(acc, n_new)
            idx_map = np.linspace(0, len(act_labels) - 1, n_new).astype(int)
            act_labels_rs = act_labels[idx_map]
            y_binary_rs = y_binary[idx_map]
            times_rs = np.linspace(0.0, times_orig[-1], n_new)

            # Gap-aware windowing (inline since BIOCLITE has its own structure)
            dt = np.diff(times_rs)
            gap_idx = np.where(dt > GAP_THRESHOLD)[0] + 1
            bounds = np.concatenate([[0], gap_idx, [len(times_rs)]])

            windows, y_true, win_times, win_activities = [], [], [], []
            for k in range(len(bounds) - 1):
                seg_start, seg_end = bounds[k], bounds[k + 1]
                if (seg_end - seg_start) < WINDOW_SIZE:
                    continue
                for wi in range(seg_start + WINDOW_SIZE, seg_end, STEP_SIZE):
                    win = acc_rs[wi - WINDOW_SIZE:wi]
                    lab_win = y_binary_rs[wi - WINDOW_SIZE:wi]
                    act_win = act_labels_rs[wi - WINDOW_SIZE:wi]
                    windows.append(win.T)
                    y_true.append(int(np.mean(lab_win) > 0.5))
                    win_times.append(times_rs[wi - 1])
                    unique, counts = np.unique(act_win, return_counts=True)
                    win_activities.append(int(unique[np.argmax(counts)]))

            if not windows:
                print(f"  Trial {i+1:02d} P{participant:02d} | skipped (no valid windows)")
                continue

            wins_np = np.array(windows, dtype=np.float32)
            y_true = np.array(y_true)
            win_times = np.array(win_times)
            win_activities = np.array(win_activities)

            probs = run_inference(model, wins_np, device)
            y_pred = (probs > CONF_THRESH).astype(int)

            prec, rec, f1, acc_score, cm = compute_metrics(y_true, y_pred)
            total_tn += cm[0, 0]; total_fp += cm[0, 1]
            total_fn += cm[1, 0]; total_tp += cm[1, 1]

            print(f"  Trial {i+1:02d} P{participant:02d} | "
                  f"Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  Acc={acc_score:.3f}  "
                  f"[gait={y_true.sum()}/{len(y_true)} windows]")

            results.append({
                'subject': f'P{participant:02d}', 'dataset': 'BIOCLITE',
                'activity': 'FreeLiving', 'wrist': 'preferred',
                'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc_score,
                'confusion_matrix': cm.tolist(),
                'probs': probs, 'y_true': y_true, 'y_pred': y_pred,
                'win_times': win_times, 'win_activities': win_activities,
            })

        except Exception as e:
            print(f"  Error in trial {i+1}: {e}")

    total = total_tp + total_tn + total_fp + total_fn
    g_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    g_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    g_f1   = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0
    g_acc  = (total_tp + total_tn) / total if total > 0 else 0
    print(f"\nBIOCLITE GLOBAL: Prec={g_prec:.3f} | Rec={g_rec:.3f} | "
          f"F1={g_f1:.3f} | Acc={g_acc:.3f}")

    by_act = defaultdict(lambda: [0, 0, 0, 0])
    for r in results:
        yt, yp, wa = r['y_true'], r['y_pred'], r['win_activities']
        for act_idx, act_name in BIOCLITE_LABEL_MAP.items():
            mask = (wa == act_idx)
            if mask.sum() == 0:
                continue
            by_act[act_name][0] += int(((yp[mask] == 1) & (yt[mask] == 1)).sum())
            by_act[act_name][1] += int(((yp[mask] == 1) & (yt[mask] == 0)).sum())
            by_act[act_name][2] += int(((yp[mask] == 0) & (yt[mask] == 1)).sum())
            by_act[act_name][3] += int(((yp[mask] == 0) & (yt[mask] == 0)).sum())

    print("\nBy activity (pooled across all subjects):")
    print(f"  {'Activity':<28} {'Precision':>10} {'Recall':>10} {'F1':>10} "
          f"{'Accuracy':>10} {'Windows':>10}")
    print(f"  {'-'*70}")
    for act_name, (tp, fp, fn, tn) in sorted(by_act.items()):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        a = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        print(f"  {act_name:<28} {p:>10.3f} {r:>10.3f} {f:>10.3f} "
              f"{a:>10.3f} {tp+fp+fn+tn:>10}")

    return results, {'precision': g_prec, 'recall': g_rec, 'f1': g_f1, 'accuracy': g_acc}


def evaluate_qsense_dataset(model, device, dataset_path):
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    print("\n" + "=" * 60)
    print(f"EVALUATING: {dataset_name}")
    print("=" * 60)

    results = []
    total_tp = total_fp = total_fn = total_tn = 0
    by_act = defaultdict(lambda: [0, 0, 0, 0])

    if not os.path.isdir(dataset_path):
        print(f"Path not found: {dataset_path}")
        return results, {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0}

    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        parts = folder.split('_')
        activity_type = '_'.join(parts[:-1]) if len(parts) > 1 else folder
        subject = parts[-1] if len(parts) > 1 else 'Unknown'

        for wrist, fname in [('right', 's1_1RW.txt'), ('left', 's2_2LW.txt')]:
            selected_path = os.path.join(folder_path, fname)
            if not os.path.exists(selected_path):
                continue

            try:
                times_50, acc_50, y_50, acts_50 = _load_qsense_file(selected_path, folder)
                discontinuity_times = get_discontinuity_times(times_50)

                # Resample 50 Hz → 30 Hz
                n_new = int(len(acc_50) * (TARGET_FS / 50.0))
                acc_rs = signal.resample(acc_50, n_new)
                idx_map = np.linspace(0, len(y_50) - 1, n_new).astype(int)
                y_rs = y_50[idx_map]
                acts_rs = acts_50[idx_map]
                times_rs = np.linspace(times_50[0], times_50[-1], n_new) if n_new > 1 else times_50

                wins_np, y_true, win_times, win_activities = extract_windows(
                    times_rs, acc_rs, y_rs, acts_rs
                )
                if wins_np is None:
                    print(f"  Skipping {folder}/{fname}: no valid windows")
                    continue

                probs = run_inference(model, wins_np, device)
                y_pred = (probs > CONF_THRESH).astype(int)
                update_activity_confusions(by_act, y_true, y_pred, win_activities)

                prec, rec, f1, acc_score, cm = compute_metrics(y_true, y_pred)
                total_tn += cm[0, 0]; total_fp += cm[0, 1]
                total_fn += cm[1, 0]; total_tp += cm[1, 1]

                print(f"  {folder:<30} {wrist:<6} | Prec={prec:.3f}  Rec={rec:.3f}  "
                      f"F1={f1:.3f}  Acc={acc_score:.3f}  [gait={y_true.sum()}/{len(y_true)} windows]")

                results.append({
                    'subject': subject, 'dataset': dataset_name,
                    'activity': activity_type, 'wrist': wrist,
                    'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc_score,
                    'confusion_matrix': cm.tolist(),
                    'probs': probs, 'y_true': y_true, 'y_pred': y_pred,
                    'win_times': win_times, 'win_activities': win_activities,
                    'discontinuity_times': discontinuity_times,
                })

            except Exception as e:
                print(f"  Error in {folder}/{fname} ({wrist}): {e}")

    total = total_tp + total_tn + total_fp + total_fn
    g_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    g_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    g_f1   = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0
    g_acc  = (total_tp + total_tn) / total if total > 0 else 0
    print(f"\n{dataset_name} GLOBAL: Prec={g_prec:.3f} | Rec={g_rec:.3f} | "
          f"F1={g_f1:.3f} | Acc={g_acc:.3f}")
    if by_act:
        print_by_activity_table(by_act, dataset_name)
    return results, {'precision': g_prec, 'recall': g_rec, 'f1': g_f1, 'accuracy': g_acc}


def evaluate_free_living(model, device, dataset_path=FREE_LIVING_PATH):
    print("\n" + "=" * 60)
    print(f"EVALUATING: {FREE_LIVING_DATASET_NAME}")
    print("=" * 60)

    if not os.path.isdir(dataset_path):
        print(f"Path not found: {dataset_path}")
        return [], {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0}

    SOURCE_FS_FL = 50.0
    results = []
    total_tp = total_fp = total_fn = total_tn = 0
    by_act = defaultdict(lambda: [0, 0, 0, 0])

    for fname in sorted(os.listdir(dataset_path)):
        if not fname.endswith('_annotated.csv'):
            continue

        fpath = os.path.join(dataset_path, fname)
        parts = fname.replace('_annotated.csv', '').split('_')
        subject = parts[1] if len(parts) > 1 else fname

        try:
            raw = pd.read_csv(fpath)
            raw['datetime'] = pd.to_datetime(
                raw['time'], format='%m/%d/%Y %H:%M:%S.%f', errors='coerce'
            )
            raw = raw.dropna(subset=['datetime']).reset_index(drop=True)

            labels = pd.to_numeric(raw['Label'], errors='coerce').fillna(0).astype(int).values
            acc = np.column_stack([
                pd.to_numeric(raw['ax'], errors='coerce').values,
                pd.to_numeric(raw['ay'], errors='coerce').values,
                pd.to_numeric(raw['az'], errors='coerce').values,
            ])
            times_orig = (raw['datetime'] - raw['datetime'].iloc[0]).dt.total_seconds().values

            valid = np.isfinite(times_orig) & np.isfinite(acc).all(axis=1)
            times_orig = times_orig[valid]
            acc = acc[valid]
            labels = labels[valid]
            activities = np.where(labels == 1, 'Gait', 'NonGait').astype(object)

            # Resample 50 Hz → 30 Hz
            n_new = int(len(acc) * (TARGET_FS / SOURCE_FS_FL))
            acc_rs = signal.resample(acc, n_new)
            idx_map = np.linspace(0, len(labels) - 1, n_new).astype(int)
            labels_rs = labels[idx_map]
            activities_rs = activities[idx_map]
            times_rs = np.linspace(times_orig[0], times_orig[-1], n_new) if n_new > 1 else times_orig

            wins_np, y_true, win_times, win_activities = extract_windows(
                times_rs, acc_rs, labels_rs, activities_rs
            )
            if wins_np is None:
                print(f"  Skipping {fname}: no valid windows")
                continue

            probs = run_inference(model, wins_np, device)
            y_pred = (probs > CONF_THRESH).astype(int)
            update_activity_confusions(by_act, y_true, y_pred, win_activities)

            prec, rec, f1, acc_score, cm = compute_metrics(y_true, y_pred)
            total_tn += cm[0, 0]; total_fp += cm[0, 1]
            total_fn += cm[1, 0]; total_tp += cm[1, 1]

            print(f"  {fname:<35} | Prec={prec:.3f}  Rec={rec:.3f}  "
                  f"F1={f1:.3f}  Acc={acc_score:.3f}  [gait={y_true.sum()}/{len(y_true)} windows]")

            results.append({
                'subject': subject, 'dataset': FREE_LIVING_DATASET_NAME,
                'activity': subject, 'wrist': 'left',
                'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc_score,
                'confusion_matrix': cm.tolist(),
                'probs': probs, 'y_true': y_true, 'y_pred': y_pred,
                'win_times': win_times, 'win_activities': win_activities,
            })

        except Exception as e:
            print(f"  Error in {fname}: {e}")

    total = total_tp + total_tn + total_fp + total_fn
    g_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    g_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    g_f1   = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0
    g_acc  = (total_tp + total_tn) / total if total > 0 else 0
    print(f"\n{FREE_LIVING_DATASET_NAME} GLOBAL: Prec={g_prec:.3f} | Rec={g_rec:.3f} | "
          f"F1={g_f1:.3f} | Acc={g_acc:.3f}")
    if by_act:
        print_by_activity_table(by_act, FREE_LIVING_DATASET_NAME)
    return results, {'precision': g_prec, 'recall': g_rec, 'f1': g_f1, 'accuracy': g_acc}


# ============================================================
# TIMELINE PLOTTING (saves under …/{dataset}/eldernet/)
# ============================================================

def plot_subject_timeline(results, plots_root_dir):
    os.makedirs(plots_root_dir, exist_ok=True)
    name_counts = defaultdict(int)

    for r in results:
        dataset  = r['dataset']
        subject  = r['subject']
        wrist    = r.get('wrist', 'right')
        activity = str(r.get('activity', '')).strip()
        probs    = r['probs']
        y_pred   = r['y_pred']
        y_true   = r['y_true']

        if 'win_times' in r and r['win_times'] is not None:
            x      = r['win_times']
            xlabel = 'Time (s)'
        else:
            x      = np.arange(len(probs))
            xlabel = 'Window index'

        fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
        title = f"{dataset} — {subject} ({wrist})"
        if activity:
            title += f" | {activity}"
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # Activity transition markers
        if (
            dataset in {'BIOCLITE', 'HMP', 'WearGait', 'WISDM', FREE_LIVING_DATASET_NAME}
            or str(dataset).lower().startswith('qsense')
        ) and 'win_activities' in r and r['win_activities'] is not None:
            win_acts = np.asarray(r['win_activities'])
            if len(win_acts) == len(x) and len(win_acts) > 0:
                transition_idx = np.where(win_acts[1:] != win_acts[:-1])[0] + 1
                transition_idx = np.concatenate(([0], transition_idx))

                for idx in transition_idx:
                    xv = x[idx]
                    if dataset == 'BIOCLITE':
                        act_id = int(win_acts[idx])
                        act_name = BIOCLITE_LABEL_MAP.get(act_id, f'Activity {act_id}')
                        is_gait = (act_id == BIOCLITE_GAIT_LABEL)
                    elif dataset == 'HMP':
                        act_name = str(win_acts[idx])
                        is_gait = (act_name in HMP_GAIT_ACTIVITIES)
                    elif dataset == 'WISDM':
                        act_name = str(win_acts[idx])
                        is_gait = (act_name in WISDM_GAIT_ACTIVITIES)
                    elif str(dataset).lower().startswith('qsense'):
                        act_name = str(win_acts[idx])
                        is_gait = any(tag in act_name.lower() for tag in QSENSE_GAIT_ACTIVITIES)
                    elif str(dataset).lower() == FREE_LIVING_DATASET_NAME:
                        act_name = str(win_acts[idx])
                        is_gait = (act_name.lower() == 'gait')
                    else:
                        act_name = str(win_acts[idx])
                        is_gait = any(p in act_name.lower() for p in WEARGAIT_PATTERNS)

                    line_color = 'green' if is_gait else 'dimgray'
                    for ax in axes:
                        ax.axvline(xv, color=line_color, linestyle='--', linewidth=0.9, alpha=0.30)
                    axes[0].text(
                        xv, 0.98, act_name,
                        transform=axes[0].get_xaxis_transform(),
                        rotation=90, va='top', ha='left',
                        fontsize=7, color=line_color, alpha=0.9,
                    )

        # Build plotting arrays, breaking lines at discontinuities
        x_plot      = np.asarray(x, dtype=float)
        probs_plot  = np.asarray(probs, dtype=float)
        ytrue_plot  = np.asarray(y_true, dtype=float)
        ypred_plot  = np.asarray(y_pred, dtype=float)

        if 'discontinuity_times' in r and r['discontinuity_times'] is not None:
            x_plot, broken = insert_nan_breaks(
                x_plot, [probs_plot, ytrue_plot, ypred_plot], r['discontinuity_times']
            )
            probs_plot, ytrue_plot, ypred_plot = broken

        # Subplot 0: probability
        axes[0].plot(x_plot, probs_plot, color='steelblue', linewidth=1.5, label='Gait probability')
        axes[0].axhline(CONF_THRESH, color='black', linestyle='--', linewidth=1,
                        label=f'Threshold = {CONF_THRESH}')
        axes[0].fill_between(x_plot, 0, probs_plot, alpha=0.15, color='steelblue')
        axes[0].set_ylim(-0.05, 1.1)
        axes[0].set_ylabel('Probability', fontsize=11)
        axes[0].legend(fontsize=9, loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # Subplot 1: ground truth vs prediction
        axes[1].step(x_plot, ytrue_plot, where='post', color='green',
                     linewidth=2, alpha=0.7, label='Ground truth')
        axes[1].step(x_plot, ypred_plot + 0.05, where='post', color='crimson',
                     linewidth=1.5, linestyle='--', label='Prediction')
        axes[1].fill_between(x_plot, 0, ytrue_plot, step='post', alpha=0.15, color='green')
        axes[1].set_ylim(-0.15, 1.2)
        axes[1].set_ylabel('Gait (0/1)', fontsize=11)
        axes[1].legend(fontsize=9, loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # Subplot 2: classification breakdown
        tp_mask = ((ypred_plot == 1) & (ytrue_plot == 1)).astype(float)
        fp_mask = ((ypred_plot == 1) & (ytrue_plot == 0)).astype(float)
        fn_mask = ((ypred_plot == 0) & (ytrue_plot == 1)).astype(float)
        tn_mask = ((ypred_plot == 0) & (ytrue_plot == 0)).astype(float)

        axes[2].fill_between(x_plot, 0, tp_mask, step='post', color='green',     alpha=0.6, label='TP')
        axes[2].fill_between(x_plot, 0, tn_mask, step='post', color='lightgrey', alpha=0.6, label='TN')
        axes[2].fill_between(x_plot, 0, fp_mask, step='post', color='orange',    alpha=0.8, label='FP')
        axes[2].fill_between(x_plot, 0, fn_mask, step='post', color='crimson',   alpha=0.8, label='FN')
        axes[2].set_ylim(-0.1, 1.2)
        axes[2].set_ylabel('Classification', fontsize=11)
        axes[2].set_xlabel(xlabel, fontsize=11)
        axes[2].legend(fontsize=9, loc='upper right', ncol=4)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        safe_subject   = subject.replace('/', '_').replace(' ', '_')
        safe_activity  = activity.replace('/', '_').replace(' ', '_') if activity else ''
        dataset_safe   = str(dataset).replace('/', '_').replace(' ', '_')
        dataset_plot_dir = os.path.join(plots_root_dir, dataset_safe, 'eldernet')
        os.makedirs(dataset_plot_dir, exist_ok=True)

        parts = [dataset_safe]
        if safe_activity:
            parts.append(safe_activity)
        parts.extend([safe_subject, wrist])
        base_name = '_'.join(parts)

        name_counts[base_name] += 1
        suffix = '' if name_counts[base_name] == 1 else f"_{name_counts[base_name]}"
        save_path = os.path.join(dataset_plot_dir, f'{base_name}{suffix}.png')

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    model = load_eldernet_model().to(device)

    # wisdm_results,    wisdm_global    = evaluate_wisdm(model, device)
    # weargait_results, weargait_global = evaluate_weargait(model, device)
    # hmp_results,      hmp_global      = evaluate_hmp(model, device)
    # bioclite_results, bioclite_global = evaluate_bioclite(model, device)

    qsense_results_all = []
    qsense_globals = []
    for qsense_path in QSENSE_PATHS:
        qs_results, qs_global = evaluate_qsense_dataset(model, device, qsense_path)
        qsense_results_all.extend(qs_results)
        qsense_globals.append({
            'dataset': os.path.basename(os.path.normpath(qsense_path)),
            **qs_global,
        })

    # free_results, free_global = evaluate_free_living(model, device, FREE_LIVING_PATH)

    # all_results = (
    #     wisdm_results
    #     + weargait_results
    #     + hmp_results
    #     + bioclite_results
    #     + qsense_results_all
    #     + free_results
    # )

    # plot_subject_timeline(all_results, OUTPUT_PLOTS_DIR)
    # plot_ROC_PR.plot_roc_curves(all_results, OUTPUT_PLOTS_DIR, model_name='eldernet')
    # plot_ROC_PR.plot_pr_curves(all_results, OUTPUT_PLOTS_DIR, model_name='eldernet')

    # os.makedirs(RESULTS_DIR, exist_ok=True)

    # all_rows = [
    #     {
    #         'dataset':   r['dataset'],
    #         'subject':   r['subject'],
    #         'wrist':     r.get('wrist', 'N/A'),
    #         'precision': r['precision'],
    #         'recall':    r['recall'],
    #         'f1':        r['f1'],
    #         'accuracy':  r['accuracy'],
    #     }
    #     for r in all_results
    # ]
    # global_rows = [
    #     {'dataset': 'WISDM',    **wisdm_global},
    #     {'dataset': 'WearGait', **weargait_global},
    #     {'dataset': 'HMP',      **hmp_global},
    #     {'dataset': 'BIOCLITE', **bioclite_global},
    #     *qsense_globals,
    #     {'dataset': FREE_LIVING_DATASET_NAME, **free_global},
    # ]

    # per_subject_csv = os.path.join(RESULTS_DIR, 'eldernet_cross_dataset_per_subject.csv')
    # global_csv      = os.path.join(RESULTS_DIR, 'eldernet_cross_dataset_global.csv')
    # pd.DataFrame(all_rows).to_csv(per_subject_csv, index=False)
    # pd.DataFrame(global_rows).to_csv(global_csv, index=False)
    # print(f"\nSaved per-subject results : {per_subject_csv}")
    # print(f"Saved global summary      : {global_csv}")


if __name__ == '__main__':
    main()
