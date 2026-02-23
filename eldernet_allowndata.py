# Apply ElderNet gait detection to self-recorded data in QSense_data, recorded at 50Hz
import os
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
from eldernet_owndata import prepare_windows_overlapping, resample_to_30hz, CONF_THRESH, MIN_ENERGY, MAX_ENERGY, MIN_FREQ, MAX_FREQ

# --- CONFIGURATION ---
DATASET_PATHS = [
    r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data_edge',
    r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data',
    r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data_mixed'
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
    model = torch.hub.load(REPO_NAME, 'eldernet_ft', trust_repo=True).to(device)
    model.eval()

    results = []

    all_y_true = []
    all_y_pred = []

    for dataset_path in DATASET_PATHS:
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
                    y_pred = ((probs > CONF_THRESH) & (engs > MIN_ENERGY) & (engs < MAX_ENERGY) & (frqs > MIN_FREQ) & (frqs < MAX_FREQ)).astype(int)
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
                        "num_windows": len(probs)
                    })

                    all_y_true.extend(y_true.tolist())
                    all_y_pred.extend(y_pred.tolist())

                except Exception as e:
                    print(f"Error processing {os.path.basename(file)}: {e}")
                    continue
        results_df = pd.DataFrame(results)
        # summary_path = os.path.join(DATASET_PATH, "overall_wrist_summary.csv")
        # results_df.to_csv(summary_path, index=False)

        print("\n=== FOLDER SUMMARY ===")
        print(results_df.groupby("wrist")[["precision", "recall", "f1", "accuracy"]].mean())

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

if __name__ == "__main__":main()