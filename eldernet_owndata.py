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

# --- CONFIGURATION ---
DATASET_PATH = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data_edge'
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

# --- DATA LOADING AND PREPROCESSING ---
def load_data(filepath):
    df = pd.read_csv(filepath, sep=r"\s+", engine="python")
    timestamps = pd.to_datetime(
        df['yyyy-MM-dd'] + ' ' + df['HH:mm:ss.fff'],
        format='%Y-%m-%d %H:%M:%S.%f'
    )
    time_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds()
    parent_folder = os.path.basename(os.path.dirname(filepath))
    activity_label = parent_folder.split('_')[0]
    data = pd.DataFrame({
        'datetime': timestamps,
        'time_sec': time_seconds,
        'accX': pd.to_numeric(df['accX'], errors='coerce'),
        'accY': pd.to_numeric(df['accY'], errors='coerce'),
        'accZ': pd.to_numeric(df['accZ'], errors='coerce'),
        'activity': activity_label
    })
    return data

# --- RESAMPLE DATA TO 30Hz ---
def resample_to_30hz(filepath, original_fs=SAMPLE_RATE_QSENSE):
    df = load_data(filepath)
    if abs(original_fs - 30.0) < 0.1: return df
    t = df['time_sec'].values
    new_time = np.linspace(t[0], t[-1], int((t[-1] - t[0]) * 30.0) + 1)
    resampled = pd.DataFrame({
        'time_sec': new_time,
        'accX': np.interp(new_time, t, df['accX'].values),
        'accY': np.interp(new_time, t, df['accY'].values),
        'accZ': np.interp(new_time, t, df['accZ'].values),
        'activity': df['activity'].iloc[0] 
    })
    return resampled

# --- OBTAIN GROUND TRUTH FROM DIRECTORY NAME ---   
def obtain_ground_truth(filepath):
    df = load_data(filepath)
    activity = df['activity'].iloc[0]
    if activity in GAIT_CLASSES:
        return np.ones(len(df), dtype=int)
    else:
        return np.zeros(len(df), dtype=int)
    
# --- PREPARE WINDOWS FOR ELDERNET ---
def prepare_windows_overlapping(df):
    acc_data = df[['accX', 'accY', 'accZ']].values
    activities_raw = df['activity'].values
    times = df['time_sec'].values
    windows, energies, freqs, activities, timestamps = [], [], [], [], []
    
    def get_dominant_freq(win, fs=30):
        mag = np.sqrt(np.sum(win**2, axis=0))
        mag = mag - np.mean(mag)
        freqs_fft = np.fft.rfftfreq(len(mag), d=1/fs)
        fft_vals = np.abs(np.fft.rfft(mag))
        return freqs_fft[np.argmax(fft_vals)]
    
    for i in range(0, len(acc_data) - WINDOW_SIZE + 1, STEP_SIZE):
        win = acc_data[i:i + WINDOW_SIZE]
        act_win = activities_raw[i:i + WINDOW_SIZE]
        unique, counts = np.unique(act_win, return_counts=True)
        
        windows.append(win.T)
        energies.append(np.std(np.sqrt(np.sum(win**2, axis=1))))
        freqs.append(get_dominant_freq(win.T))
        activities.append(unique[np.argmax(counts)])
        timestamps.append(times[i])

    return torch.FloatTensor(np.array(windows)), np.array(energies), np.array(freqs), activities, np.array(timestamps)

# --- FILTER PREDICTIONS TO REMOVE SHORT BOUTS ---
def apply_bout_filtering(predictions, min_bout_length=5):
    filtered = predictions.copy()
    
    # Find consecutive runs of 1s
    in_bout = False
    bout_start = 0
    
    for i in range(len(predictions)):
        if predictions[i] == 1 and not in_bout:
            # Start of potential bout
            bout_start = i
            in_bout = True
        elif predictions[i] == 0 and in_bout:
            # End of bout
            bout_length = i - bout_start
            if bout_length < min_bout_length:
                # Bout too short - remove it
                filtered[bout_start:i] = 0
            in_bout = False
    
    # Handle bout extending to end
    if in_bout:
        bout_length = len(predictions) - bout_start
        if bout_length < min_bout_length:
            filtered[bout_start:] = 0
    
    return filtered

def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        # Spread hues evenly around the color wheel
        hue = i / n
        # Use high saturation (0.8) and value (0.9) for vivid colors
        saturation = 0.8
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors

def get_wrist_variant(base_rgb, wrist):
    h, l, s = colorsys.rgb_to_hls(*base_rgb)
    
    if wrist.lower() == 'right':
        # Right wrist: use base color as-is (vivid)
        return base_rgb
    else:
        # Left wrist: lighter and less saturated
        # Increase lightness by 10%, reduce saturation by 40%
        new_l = min(0.85, l+0.3)  # Cap to avoid white
        new_s = max(0.3, s)    # Keep some saturation
        return colorsys.hls_to_rgb(h, new_l, new_s)

# --- RUN ELDERNET AND OBTAIN PROBABILITIES ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(REPO_NAME, 'eldernet_ft', trust_repo=True).to(device)
    model.eval()

    results = []

    for folder in os.listdir(DATASET_PATH):
        if not os.path.isdir(os.path.join(DATASET_PATH, folder)):
            continue

        files = [
            os.path.join(DATASET_PATH, folder, 's1_1RW.txt'),  # Right wrist
            #os.path.join(DATASET_PATH, folder, 's3_3ST.txt')   # Left wrist for QSense_data
            os.path.join(DATASET_PATH, folder, 's2_2LW.txt')   # Left wrist for QSense_data_edge
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

                    save_path = os.path.join(DATASET_PATH, folder, f"{wrist}_window_outputs.csv")
                    output_df.to_csv(save_path, index=False)

                print(f"             Processed {os.path.basename(file)}: {len(probs)} windows")

                probs_sm = np.convolve(probs, np.ones(3)/3, mode='same')
                y_pred_raw = (probs_sm > CONF_THRESH) #& (engs > MIN_ENERGY) & (engs < MAX_ENERGY) & (frqs > MIN_FREQ) & (frqs < MAX_FREQ)).astype(int)
                y_pred = median_filter(y_pred_raw, size=3)
                y_pred = apply_bout_filtering(y_pred, min_bout_length=5) # Remove bouts shorter than 5 windows (1s)

                y_true_full = np.ones(len(df_30hz), dtype=int) \
                    if df_30hz['activity'].iloc[0] in GAIT_CLASSES \
                    else np.zeros(len(df_30hz), dtype=int)

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
            except Exception as e:
                print(f"Error processing {os.path.basename(file)}: {e}")
                continue
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
            if subject.lower() in f.lower() and "free_hendrik" not in f.lower()
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
        save_path = os.path.join(DATASET_PATH, f"{subject}_eldernet_metrics.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {save_path}")
        
        plt.show()

if __name__ == "__main__":main()
