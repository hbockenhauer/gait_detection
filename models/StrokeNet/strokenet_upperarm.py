''' 
StrokeNet Upper Arm Evaluation Script
This script loads the pre-trained StrokeNet model and evaluates it on the upper arm sensor data from the Clinical dataset.
'''

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import signal
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.paths import QSENSE_CLINIC, STROKENET_WEIGHTS, PLOTS_DIR as OUTPUT_PLOTS_DIR, RESULTS_DIR
import utils.plot_ROC_PR as plot_ROC_PR
from strokenet_utils import (
    get_discontinuity_times,
    extract_windows_with_gaps_and_activity,
    plot_subject_timeline,
    run_inference,
    update_activity_confusions,
    compute_metrics,
    print_by_activity_table,
    QSENSE_GAIT_ACTIVITIES,
    _column_by_name_or_index,
    load_finetuned_model,
)

# --- CONFIGURATION ---
QSENSE_PATHS = [QSENSE_CLINIC]
WEIGHTS_PATH = STROKENET_WEIGHTS
STROKENET_RESULTS_DIR = os.path.join(RESULTS_DIR, 'StrokeNet')
REPO_NAME     = 'yonbrand/ElderNet'

WINDOW_SIZE   = 100    # 2s at 50Hz
STEP_SIZE     = 50     # 1s stride
GAP_THRESHOLD = 0.1
CONF_THRESH   = 0.5

def _load_qsense_file(filepath, folder_name):
    """Load one QSense wrist file and return sample-level times, acc, labels, and activities."""
    df = pd.read_csv(filepath, sep=None, engine='python').reset_index(drop=True)

    # Parse timestamp from first two columns (Date + Time format used in QSense exports).
    df['datetime'] = pd.to_datetime(
        df.iloc[:, 0].astype(str) + ' ' + df.iloc[:, 1].astype(str),
        errors='coerce'
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

    acc_x = _column_by_name_or_index(df, ['ax', 'accX', 'AccX'], 5)
    acc_y = _column_by_name_or_index(df, ['ay', 'accY', 'AccY'], 6)
    acc_z = _column_by_name_or_index(df, ['az', 'accZ', 'AccZ'], 7)
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
        folder_lower = folder_name.lower()
        is_gait_folder = any(tag in folder_lower for tag in QSENSE_GAIT_ACTIVITIES)
        label_series = np.ones(len(df), dtype=int) if is_gait_folder else np.zeros(len(df), dtype=int)

    times = df['time_sec'].values.astype(float)
    valid = np.isfinite(times) & np.isfinite(acc).all(axis=1)
    times = times[valid]
    acc = acc[valid]
    labels = label_series[valid]

    folder_activity = folder_name.split('_')[0]
    activities = np.array([folder_activity] * len(times), dtype=object)
    return times, acc, labels, activities


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

        wrist_candidates = {
            'upperarm': ['s0_Hub.txt'],
        }

        for wrist, file_candidates in wrist_candidates.items():
            selected_path = None
            for fname in file_candidates:
                p = os.path.join(folder_path, fname)
                if os.path.exists(p):
                    selected_path = p
                    break

            if selected_path is None:
                continue

            try:
                times, acc, y_binary, activities = _load_qsense_file(selected_path, folder)
                discontinuity_times = get_discontinuity_times(times)

                wins_np, y_true, win_times, win_activities = extract_windows_with_gaps_and_activity(
                    times, acc, y_binary, activities
                )
                if wins_np is None:
                    print(f"  Skipping {folder}/{os.path.basename(selected_path)}: no valid windows")
                    continue

                probs = run_inference(model, wins_np, device)
                y_pred = (probs > CONF_THRESH).astype(int)
                update_activity_confusions(by_act, y_true, y_pred, win_activities)

                prec, rec, f1, acc_score, cm = compute_metrics(y_true, y_pred)
                total_tn += cm[0, 0]
                total_fp += cm[0, 1]
                total_fn += cm[1, 0]
                total_tp += cm[1, 1]

                print(
                    f"  {folder:<30} {wrist:<6} | Prec={prec:.3f}  Rec={rec:.3f}  "
                    f"F1={f1:.3f}  Acc={acc_score:.3f}  [gait={y_true.sum()}/{len(y_true)} windows]"
                )

                results.append({
                    'subject': subject,
                    'dataset': dataset_name,
                    'activity': activity_type,
                    'wrist': wrist,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'accuracy': acc_score,
                    'confusion_matrix': cm.tolist(),
                    'probs': probs,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'win_times': win_times,
                    'win_activities': win_activities,
                    'discontinuity_times': discontinuity_times,
                })

            except Exception as e:
                print(f"  Error in {folder}/{os.path.basename(selected_path)} ({wrist}): {e}")

    total = total_tp + total_tn + total_fp + total_fn
    g_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    g_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    g_f1 = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0
    g_acc = (total_tp + total_tn) / total if total > 0 else 0
    print(f"\n{dataset_name} GLOBAL: Prec={g_prec:.3f} | Rec={g_rec:.3f} | F1={g_f1:.3f} | Acc={g_acc:.3f}")
    if len(by_act) > 0:
        print_by_activity_table(by_act, dataset_name)

    return results, {'precision': g_prec, 'recall': g_rec, 'f1': g_f1, 'accuracy': g_acc}

# only run on inference on QSense_clinic for now, so we can analyze the upper arm performance without needing to load all the other datasets
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_finetuned_model(WEIGHTS_PATH).to(device)
    qsense_results_all = []
    qsense_globals = []
    for path in QSENSE_PATHS:
        res, glob = evaluate_qsense_dataset(model, device, path)
        qsense_results_all.extend(res)
        qsense_globals.append({'precision': glob['precision'], 'recall': glob['recall'], 'f1': glob['f1'], 'accuracy': glob['accuracy']})

    all_results = qsense_results_all

    # Save per-subject results for QSense
    if qsense_results_all:
        qsense_rows = [{
            'subject': r['subject'], 'dataset': r['dataset'],
            'activity': r['activity'], 'wrist': r['wrist'],
            'precision': r['precision'], 'recall': r['recall'],
            'f1': r['f1'], 'accuracy': r['accuracy']
        } for r in qsense_results_all]
        qsense_df = pd.DataFrame(qsense_rows)
        os.makedirs(STROKENET_RESULTS_DIR, exist_ok=True)
        qsense_csv = os.path.join(STROKENET_RESULTS_DIR, 'strokenet_QSense_clinic_upperarm_metrics.csv')
        qsense_df.to_csv(qsense_csv, index=False)
        print(f"Saved QSense results: {qsense_csv}")


    plot_subject_timeline(all_results, OUTPUT_PLOTS_DIR)

if __name__ == "__main__":    main()
