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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from scipy.ndimage import median_filter
import matplotlib.colors as mcolors
import colorsys
from eldernet_owndata import prepare_windows_overlapping, apply_bout_filtering, generate_distinct_colors, get_wrist_variant, load_data, resample_to_30hz, obtain_ground_truth

# --- CONFIGURATION ---
DATASET_PATH = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data_edge'
DATASET_PATHS = [
    r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data_edge',
    r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data'
]

REPO_NAME = 'yonbrand/ElderNet'
WINDOW_SIZE = 300      
STEP_SIZE = 30
GAIT_CLASSES = {'Walking', 'Stairs'}
SAMPLE_RATE_QSENSE = 50.0 #Hz

CONF_THRESH = 0.6
MIN_ENERGY = 0.1
MAX_ENERGY = 2.0
MIN_FREQ = 0.0
MAX_FREQ = 3.0


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
    all_windows = []
    meta_info = []   # to track which window belongs to which file

    for dataset_root in DATASET_PATHS:
        for folder in os.listdir(dataset_root):
            folder_path = os.path.join(dataset_root, folder)
            if not os.path.isdir(folder_path):
                continue

            files = [
                os.path.join(dataset_root, folder, 's1_1RW.txt'),
                os.path.join(dataset_root, folder, 's2_2LW.txt')
            ]

            for file in files:
                if not os.path.exists(file):
                    continue

                wrist = "right" if "1RW" in file else "left"

                df_30hz = resample_to_30hz(file)
                wins, engs, frqs, acts, tmstps = prepare_windows_overlapping(df_30hz)

                all_windows.append(wins)
                meta_info.append({
                    "dataset_root": dataset_root,
                    "folder": folder,
                    "file": file,
                    "wrist": wrist,
                    "energies": engs,
                    "freqs": frqs,
                    "timestamps": tmstps,
                    "df_len": len(df_30hz)
                })

    # Concatenate all windows
    all_windows = torch.cat(all_windows, dim=0)

    with torch.no_grad():
        logits = model(all_windows.to(device))
        all_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    start_idx = 0

    for meta in meta_info:
        n = len(meta["timestamps"])
        probs = all_probs[start_idx:start_idx+n]
        start_idx += n

        folder = meta["folder"]
        wrist = meta["wrist"]
        engs = meta["energies"]
        frqs = meta["freqs"]
        tmstps = meta["timestamps"]

        output_df = pd.DataFrame({
            "timestamp": tmstps,
            "probability": probs,
            "energy": engs,
            "frequency": frqs
        })

        save_path = os.path.join(meta["dataset_root"], folder, f"{wrist}_window_outputs.csv")
        output_df.to_csv(save_path, index=False)

        # Apply filtering
        probs_sm = np.convolve(probs, np.ones(3)/3, mode='same')
        y_pred_raw = (probs_sm > CONF_THRESH)
        y_pred = median_filter(y_pred_raw, size=3)
        y_pred = apply_bout_filtering(y_pred, min_bout_length=5)

        # Ground truth
        df_len = meta["df_len"]
        y_true_full = np.ones(df_len, dtype=int) \
            if folder.split('_')[0] in GAIT_CLASSES else np.zeros(df_len, dtype=int)

        y_true = []
        for i in range(0, len(y_true_full) - WINDOW_SIZE + 1, STEP_SIZE):
            segment = y_true_full[i:i + WINDOW_SIZE]
            y_true.append(int(np.mean(segment) > 0.5))
        y_true = np.array(y_true)

        if np.sum(y_true) == 0:
            p, r, f1 = 0.0, 0.0, 0.0
        else:
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred,
                labels=[1],
                average='binary',
                zero_division=0
            )

        acc = accuracy_score(y_true, y_pred)

        results.append({
            "activity": folder,
            "wrist": wrist,
            "precision": p,
            "recall": r,
            "f1": f1,
            "accuracy": acc,
            "num_windows": n
        })



    results_df = pd.DataFrame(results)
    summary_path = os.path.join(DATASET_PATH, "overall_wrist_summary.csv")
    results_df.to_csv(summary_path, index=False)

    print("\n=== OVERALL SUMMARY ===")
    print(results_df.groupby("wrist")[["precision", "recall", "f1", "accuracy"]].mean())

    # Subjects to plot
    subjects = ['Hendrik', 'Tanya']
    wrists = ['right', 'left']
    metrics = ['probability', 'energy', 'frequency']

    for subject in subjects:
        fig = plt.figure(figsize=(20, 10))
        plt.suptitle(f"{subject} - ElderNet Window Metrics", fontsize=16)
        
        # Get all folders for this subject
        unique_folders = sorted([
            f for f in os.listdir(DATASET_PATH) 
            if subject.lower() in f.lower()
        ])
        num_acts = len(unique_folders)
    
        base_colors = generate_distinct_colors(num_acts)
        
        # Create axes
        axes = []
        for i, metric in enumerate(metrics, start=1):
            ax = plt.subplot(3, 1, i)
            axes.append(ax)

            # Plot each activity with its distinct color
            for idx, folder in enumerate(unique_folders):
                base_color = base_colors[idx]
                
                for wrist in wrists:
                    file_path = os.path.join(DATASET_PATH, folder, f"{wrist}_window_outputs.csv")
                    if not os.path.exists(file_path):
                        continue

                    # Get wrist-specific color variant
                    plot_color = get_wrist_variant(base_color, wrist)

                    df = pd.read_csv(file_path)
                    
                    ax.plot(
                        df['timestamp'], 
                        df[metric], 
                        label=f"{folder} | {wrist}", 
                        color=plot_color,
                        alpha=0.85
                    )

            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=10)

        axes[-1].set_xlabel("Time (seconds)", fontsize=12)
        
        handles, labels = axes[-1].get_legend_handles_labels()
        
        # Sort legend entries by activity (groups wrists together)
        sorted_pairs = sorted(zip(labels, handles), key=lambda x: x[0])
        sorted_labels, sorted_handles = zip(*sorted_pairs)
        
        fig.legend(sorted_handles, sorted_labels, loc='center left', bbox_to_anchor=(0.851, 0.5), fontsize=8, title='Activity | Wrist', 
            frameon=True, ncol=1, title_fontsize=9)

        plt.tight_layout(rect=[0, 0, 0.85, 0.96])
        
        # Save figure
        # save_path = os.path.join(DATASET_PATH, f"{subject}_eldernet_metrics.png")
        # plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # print(f"Saved plot: {save_path}")
        
        # plt.show()

if __name__ == "__main__":main()
