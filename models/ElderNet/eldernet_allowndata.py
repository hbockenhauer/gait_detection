# Apply ElderNet gait detection to self-recorded data in Baseline, recorded at 50Hz
import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import matplotlib.pyplot as plt
import pickle
import random
from datetime import datetime
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score, confusion_matrix
from scipy.ndimage import median_filter
import matplotlib.colors as mcolors
import colorsys
from eldernet_owndata import prepare_windows_overlapping, resample_to_30hz, CONF_THRESH, MIN_ENERGY, MAX_ENERGY, MIN_FREQ, MAX_FREQ, apply_bout_constraints

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.paths import QSENSE_EDGE, QSENSE_DATA, QSENSE_MIXED, RESULTS_DIR
from utils.hub_utils import safe_hub_load

DATASET_PATHS = [
    QSENSE_EDGE,
    QSENSE_DATA,
    QSENSE_MIXED,
]

REPO_NAME = 'yonbrand/ElderNet'
WINDOW_SIZE = 300      
STEP_SIZE = 30
GAIT_CLASSES = {'Walking', 'Stairs'}
SAMPLE_RATE_QSENSE = 50.0 #Hz


# --- REPRODUCIBILITY ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- RUN ELDERNET AND OBTAIN PROBABILITIES ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = safe_hub_load(REPO_NAME, 'eldernet_ft', trust_repo=True).to(device)
    model.eval()

    results = []

    all_y_true = []
    all_y_pred = []

    for dataset_path in DATASET_PATHS:
        dataset_results = []
        for folder in os.listdir(dataset_path):
            if not os.path.isdir(os.path.join(dataset_path, folder)):
                continue

            files = [
                os.path.join(dataset_path, folder, 's1_1RW.txt'),  # Right wrist
                os.path.join(dataset_path, folder, 's2_2LW.txt')   # Left wrist 
            ]

            for file in files:
                if not os.path.exists(file):
                    continue
                wrist = "right" if "1RW" in file else "left"

                try:
                    df_30hz = resample_to_30hz(file)
                    wins, engs, frqs, acts, tmstps = prepare_windows_overlapping(df_30hz)

                    with torch.no_grad():
                        logits = model(wins.to(device))
                        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                        # Save per-window outputs
                        output_df = pd.DataFrame({
                            "timestamp": tmstps,
                            "probability": probs,
                            "energy": engs,
                            "frequency": frqs
                        })

                        save_path = os.path.join(dataset_path, folder, f"{wrist}_window_outputs.csv")
                        output_df.to_csv(save_path, index=False)

                    print(f"             Processed {os.path.basename(file)}: {len(probs)} windows")

                    # n_smooth = int(SMOOTHING_SEC / STEP_SEC)     # 10 windows
                    #n_bout   = int(MIN_BOUT_SEC  / STEP_SEC)     # 5 windows

                    #probs_smoothed = uniform_filter1d(probs, size=n_smooth)
                    # y_pred = (probs_smoothed > CONF_THRESH).astype(int)

                    #probs_sm = np.convolve(probs, np.ones(3)/3, mode='same')
                    y_pred = ((probs > 0.65)) #& (engs > MIN_ENERGY) & (engs < MAX_ENERGY) & (frqs > MIN_FREQ) & (frqs < MAX_FREQ)).astype(int)
                    y_pred = apply_bout_constraints(y_pred, min_bout_sec=5.0, max_gap_sec=2.0, step_sec=1.0)
                    #y_pred = median_filter(y_pred_raw, size=3)
                    #y_pred = apply_bout_filtering(y_pred, min_bout_length=MIN_BOUT_SEC) # Remove bouts shorter than MIN_BOUT_SEC windows           
                    y_true_full = df_30hz['gt'].values

                    # Convert sample-level GT to window-level GT
                    y_true = []
                    for i in range(0, len(y_true_full) - WINDOW_SIZE + 1, STEP_SIZE):
                        segment = y_true_full[i:i + WINDOW_SIZE]
                        y_true.append(int(np.mean(segment) > 0.5))

                    y_true = np.array(y_true)

                    # Metrics
                    if np.sum(y_true) == 0:
                        p, r, f1 = 0.0, 0.0, 0.0
                    else:
                        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average='binary', zero_division=0)
                    acc = accuracy_score(y_true, y_pred)
                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

                    print(
                        f"{folder} | {wrist.upper()} | "
                        f"Precision: {p:.3f} | Recall: {r:.3f} | "
                        f"F1: {f1:.3f} | Accuracy: {acc:.3f}"
                    )

                    results.append({
                        "activity": folder,
                        "wrist": wrist,
                        "precision": p,
                        "recall": r,
                        "f1": f1,
                        "accuracy": acc,
                        "num_windows": len(probs),
                        "confusion_matrix": cm.tolist()
                    })
                    dataset_results.append({
                        "activity": folder,
                        "wrist": wrist,
                        "precision": p,
                        "recall": r,
                        "f1": f1,
                        "accuracy": acc,
                        "num_windows": len(probs),
                        "confusion_matrix": cm.tolist()
                    })

                    all_y_true.extend(y_true.tolist())
                    all_y_pred.extend(y_pred.tolist())

                except Exception as e:
                    print(f"Error processing {os.path.basename(file)}: {e}")
                    continue
        results_df = pd.DataFrame(dataset_results)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        dataset_name = os.path.basename(dataset_path)
        summary_path = os.path.join(RESULTS_DIR, f"eldernet_{dataset_name}_metrics.csv")
        results_df.to_csv(summary_path, index=False)
        print(f"Saved metrics summary to: {summary_path}")

        if results_df.empty:
            print(f"\n=== FOLDER SUMMARY ({dataset_name}) ===")
            print("No valid windows found.")
            continue

        print("\n=== FOLDER SUMMARY ===")
        def pooled_metrics(rows):
            tn = fp = fn = tp = 0
            for row in rows:
                c = np.array(row.get('confusion_matrix', []))
                if c.shape != (2, 2):
                    continue
                tn += int(c[0, 0])
                fp += int(c[0, 1])
                fn += int(c[1, 0])
                tp += int(c[1, 1])

            total = tp + tn + fp + fn
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            acc = (tp + tn) / total if total > 0 else 0.0
            return {'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc}

        pooled_rows = []
        for wrist, grp in results_df.groupby('wrist'):
            m = pooled_metrics(grp.to_dict('records'))
            m['wrist'] = wrist
            pooled_rows.append(m)

        pooled_df = pd.DataFrame(pooled_rows).set_index('wrist')[['precision', 'recall', 'f1', 'accuracy']]
        print(pooled_df)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    p_g, r_g, f1_g, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, labels=[1], average='binary', zero_division=0
    )
    acc_g = accuracy_score(all_y_true, all_y_pred)

    print("\n=== GLOBAL PERFORMANCE (ALL DATASETS COMBINED) ===")
    print(f"Precision: {p_g:.3f}")
    print(f"Recall:    {r_g:.3f}")
    print(f"F1-score:  {f1_g:.3f}")
    print(f"Accuracy:  {acc_g:.3f}")

    pd.DataFrame([{
        'dataset': 'QSense_combined',
        'precision': p_g,
        'recall': r_g,
        'f1': f1_g,
        'accuracy': acc_g,
    }]).to_csv(os.path.join(RESULTS_DIR, 'eldernet_QSense_combined_global_metrics.csv'), index=False)

if __name__ == "__main__":main()