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
from scipy.ndimage import median_filter, uniform_filter1d
import matplotlib.colors as mcolors
import colorsys

# --- CONFIGURATION ---
DATASET_PATH = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data_mixed'
REPO_NAME = 'yonbrand/ElderNet'
WINDOW_SIZE = 300      #10s at 30Hz
STEP_SIZE = 30          #1s at 30Hz
GAIT_CLASSES = {'Walking', 'Stairs'}
SAMPLE_RATE_QSENSE = 50.0 #Hz
# SMOOTHING_SEC = 10.0
STEP_SEC = STEP_SIZE / 30.0
# N_SMOOTH = int(SMOOTHING_SEC / STEP_SEC)
MIN_BOUT_SEC = 10.0

CONF_THRESH = 0.1
MIN_ENERGY = 0.07
MAX_ENERGY = 0.4
MIN_FREQ = 0.5
MAX_FREQ = 3.5


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

def load_data(filepath):
    df = pd.read_csv(filepath, sep=None, engine="python")
    df = df.reset_index(drop=True)

    time_seconds = np.arange(len(df)) / SAMPLE_RATE_QSENSE

    parent_folder = os.path.basename(os.path.dirname(filepath))

    # -------- CASE 1: Sample-level label exists --------
    if 'label' in df.columns or 'Label' in df.columns:
        label_col = 'label' if 'label' in df.columns else 'Label'
        sample_gt = pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(int)

    # -------- CASE 2: Folder-level activity --------
    else:
        activity_name = parent_folder.split('_')[0]
        sample_gt = np.ones(len(df), dtype=int) if activity_name in GAIT_CLASSES else np.zeros(len(df), dtype=int)

    data = pd.DataFrame({
        'time_sec': time_seconds,
        'accX': pd.to_numeric(df['accX'], errors='coerce'),
        'accY': pd.to_numeric(df['accY'], errors='coerce'),
        'accZ': pd.to_numeric(df['accZ'], errors='coerce'),
        'gt': sample_gt
    })

    data = data[data['time_sec'] >= 10.0].reset_index(drop=True)

    return data

# --- RESAMPLE DATA TO 30Hz ----
def resample_to_30hz(filepath, original_fs=SAMPLE_RATE_QSENSE):
    df = load_data(filepath)

    if abs(original_fs - 30.0) < 0.1:
        return df

    t = df['time_sec'].values
    new_time = np.linspace(t[0], t[-1], int((t[-1] - t[0]) * 30.0) + 1)

    resampled = pd.DataFrame({
        'time_sec': new_time,
        'accX': np.interp(new_time, t, df['accX'].values),
        'accY': np.interp(new_time, t, df['accY'].values),
        'accZ': np.interp(new_time, t, df['accZ'].values),
        'gt': np.round(np.interp(new_time, t, df['gt'].values)).astype(int)
    })

    return resampled
    
# --- PREPARE WINDOWS FOR ELDERNET ---
def prepare_windows_overlapping(df):
    acc_data = df[['accX', 'accY', 'accZ']].values
    gt_raw = df['gt'].values

    #activities_raw = df['activity'].values
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
        #act_win = activities_raw[i:i + WINDOW_SIZE]
        act_win = gt_raw[i:i + WINDOW_SIZE]
        #unique, counts = np.unique(act_win, return_counts=True)
        
        windows.append(win.T)
        energies.append(np.std(np.sqrt(np.sum(win**2, axis=1))))
        freqs.append(get_dominant_freq(win.T))
        activities.append(int(np.mean(act_win) > 0.5))

        #activities.append(unique[np.argmax(counts)])
        timestamps.append(times[i])

    return torch.FloatTensor(np.array(windows)), np.array(energies), np.array(freqs), activities, np.array(timestamps)

# # --- SMOOTHING AND TEMPORALLY-AWARE THRESHOLDING ---
# def smooth_and_threshold(probs, timestamps,
#                           smoothing_window=10,    # seconds of smoothing
#                           conf_thresh=0.6,
#                           min_bout_sec=5.0,       # minimum bout duration
#                           step_size_sec=1.0):     # your step size
    
#     # Step 1: Smooth probabilities over a longer window
#     # This collapses short spikes but preserves sustained high values
#     n_windows_smooth = int(smoothing_window / step_size_sec)
#     probs_smoothed = uniform_filter1d(probs, size=n_windows_smooth)
    
#     # Step 2: Threshold on the SMOOTHED signal
#     # Now a spike needs to be sustained for `smoothing_window` seconds
#     # to cross the threshold - short spikes get averaged down
#     y_pred = (probs_smoothed > conf_thresh).astype(int)
    
#     # Step 3: Bout filter on top of smoothed predictions
#     min_bout_windows = int(min_bout_sec / step_size_sec)
#     y_pred = apply_bout_filtering(y_pred, min_bout_length=min_bout_windows)
    
#     return y_pred, probs_smoothed

# # --- FILTER PREDICTIONS TO REMOVE SHORT BOUTS ---
# def apply_bout_filtering(predictions, min_bout_length=MIN_BOUT_SEC):
#     """Remove bouts shorter than min_bout_length windows"""
#     filtered = predictions.copy()
#     in_bout = False
#     bout_start = 0
    
#     for i in range(len(predictions)):
#         if predictions[i] == 1 and not in_bout:
#             bout_start = i
#             in_bout = True
#         elif predictions[i] == 0 and in_bout:
#             if i - bout_start < min_bout_length:
#                 filtered[bout_start:i] = 0
#             in_bout = False
    
#     if in_bout and len(predictions) - bout_start < min_bout_length:
#         filtered[bout_start:] = 0
    
#     return filtered

# --- PLOTTING FUNCTION ---
def plot_per_activity(results_list, subjects, metrics):
    """Plot metrics and GT/predictions using pre-calculated results"""
    
    # --- GROUP RESULTS BY ACTIVITY TYPE ---
    activities_by_type = {}
    for result in results_list:
        activity_type = result['activity_type']
        if activity_type not in activities_by_type:
            activities_by_type[activity_type] = []
        activities_by_type[activity_type].append(result)
    
    activity_types = sorted(activities_by_type.keys())
    
    print(f"\nFound {len(activity_types)} unique activities: {activity_types}")
    
    # --- COLOR SCHEME ---
    wrist_colors = {
        'right': '#1f77b4',   # Blue
        'left':  '#ff7f0e',   # Orange
    }

    # --- ONE PLOT PER ACTIVITY ---
    for activity_type, activity_results in activities_by_type.items():
        
        # Create figure: metrics + GT/Pred row
        n_rows = len(metrics) + 1  # +1 for GT/Pred row
        fig, axes = plt.subplots(n_rows, 1, 
                                  figsize=(16, 4 * n_rows), 
                                  sharex=False)
        
        if n_rows == 1:
            axes = [axes]  # Ensure iterable
        
        fig.suptitle(f"Activity: {activity_type}", fontsize=16, fontweight='bold')
        
        has_data = False
        
        for ax, metric in zip(axes[:-1], metrics):
            
            # Plot each subject
            for subject in subjects:
                
                # Filter results for this subject and metric
                subject_results = [r for r in activity_results if r['subject'] == subject]
                
                if not subject_results:
                    print(f"  No data for {subject} in {activity_type}")
                    continue
                
                # Plot each wrist
                for result in subject_results:
                    if metric not in ['probability', 'energy', 'frequency']:
                        continue
                    
                    has_data = True
                    timestamps = result['timestamps']
                    
                    if metric == 'probability':
                        values = result['probability']
                    elif metric == 'energy':
                        values = result['energy']
                    elif metric == 'frequency':
                        values = result['frequency']
                    
                    wrist = result['wrist']
                    color = wrist_colors[wrist]

                    label_raw = f"{wrist} | {subject}"

                    ax.plot(timestamps, values,
                            color=color,
                            linewidth=1.5,
                            alpha=0.95,
                            label=label_raw)
            
            # --- Threshold lines ---
            if metric == 'probability':
                ax.axhline(CONF_THRESH,
                        color='black',
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.8,
                        label=f'Prob thresh = {CONF_THRESH}')
                ax.set_ylim(-0.05, 1.1)

            elif metric == 'energy':
                ax.axhline(MIN_ENERGY,
                        color='black',
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.8,
                        label=f'Min energy = {MIN_ENERGY}')
                ax.axhline(MAX_ENERGY,
                        color='black',
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.8,
                        label=f'Max energy = {MAX_ENERGY}')

            elif metric == 'frequency':
                ax.axhline(MIN_FREQ,
                        color='black',
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.8,
                        label=f'Min freq = {MIN_FREQ}')
                ax.axhline(MAX_FREQ,
                        color='black',
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.8,
                        label=f'Max freq = {MAX_FREQ}')
                        
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=10)
        
        # --- GROUND TRUTH AND PREDICTIONS ROW ---
        ax_gt_pred = axes[-1]

        for subject in subjects:
            subject_results = [r for r in activity_results if r['subject'] == subject]
            if not subject_results:
                continue

            for result in subject_results:
                timestamps = result['timestamps']
                y_true = result['y_true']
                y_pred = result['y_pred']
                wrist = result['wrist']

                # --- Color encodes wrist ---
                if wrist == 'right':
                    base_color = '#1f77b4' 
                else:
                    base_color = '#ff7f0e'  

                # --- Ground Truth (thick solid band) ---
                ax_gt_pred.fill_between(
                    timestamps,
                    0,
                    y_true,
                    step='post',
                    alpha=0.25,
                    color=base_color,
                    label=f'{wrist} | {subject} | GT'
                )

                # --- Prediction (sharp line, slightly offset) ---
                ax_gt_pred.step(
                    timestamps,
                    y_pred + 0.05,
                    where='post',
                    color=base_color,
                    linewidth=2.5,
                    label=f'{wrist} | {subject} | Pred'
                )

        ax_gt_pred.set_ylabel("GT / Prediction", fontsize=12)
        ax_gt_pred.set_ylim(-0.1, 1.15)
        ax_gt_pred.grid(True, alpha=0.3)
        ax_gt_pred.set_xlabel("Time (seconds)", fontsize=12)
                
        if not has_data:
            print(f"  Skipping {activity_type}: no data found")
            plt.close(fig)
            continue
        
        # --- LEGEND ---
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
                
        seen = set()
        unique_handles, unique_labels = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)
        
        fig.legend(
            unique_handles, unique_labels,
            loc='center left',
            bbox_to_anchor=(0.87, 0.5),
            fontsize=10,
            title='Subject | Wrist',
            title_fontsize=11,
            frameon=True,
            ncol=1
        )
        
        plt.tight_layout(rect=[0, 0, 0.86, 0.95])
        
        # Save - use DATASET_PATH from config
        plots_dir = os.path.join(DATASET_PATH, "Plots")
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, f"activity_{activity_type}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')        
        plt.show()

# --- RUN ELDERNET AND OBTAIN PROBABILITIES ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(REPO_NAME, 'eldernet_ft', trust_repo=True).to(device)
    model.eval()

    results = []

    for folder in os.listdir(DATASET_PATH):
        if not os.path.isdir(os.path.join(DATASET_PATH, folder)):
            continue

        # Extract subject name and activity type from folder name
        parts = folder.split('_')
        activity_type = '_'.join(parts[:-1]) if len(parts) > 1 else folder
        subject = parts[-1] if len(parts) > 1 else 'Unknown'

        files = [
            os.path.join(DATASET_PATH, folder, 's1_1RW.txt'),  # Right wrist
            os.path.join(DATASET_PATH, folder, 's2_2LW.txt')   # Left wrist 
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

                # Compute predictions
                y_pred = ((probs > CONF_THRESH) & (engs > MIN_ENERGY) & (engs < MAX_ENERGY) & 
                         (frqs > MIN_FREQ) & (frqs < MAX_FREQ)).astype(int)
                
                y_true_full = df_30hz['gt'].values

                # Convert sample-level GT to window-level GT
                y_true = []
                timestamps = []
                for i in range(0, len(y_true_full) - WINDOW_SIZE + 1, STEP_SIZE):
                    segment = y_true_full[i:i + WINDOW_SIZE]
                    y_true.append(int(np.mean(segment) > 0.5))
                    timestamps.append(df_30hz['time_sec'].values[i])

                y_true = np.array(y_true)
                timestamps = np.array(timestamps)

                # Compute metrics
                if np.sum(y_true) == 0:
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0
                else:
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, labels=[1], average='binary', zero_division=0
                    )
                accuracy = accuracy_score(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

                print(
                    f"{folder} | {wrist.upper()} | "
                    f"Precision: {precision:.3f} | Recall: {recall:.3f} | "
                    f"F1: {f1:.3f} | Accuracy: {accuracy:.3f}"
                )

                # Create a comprehensive results dictionary
                results.append({
                    'subject': subject,
                    'folder': folder,
                    'activity_type': activity_type,
                    'wrist': wrist,
                    'file_path': file,
                    # Raw data for plotting
                    'timestamps': timestamps,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'probability': probs,
                    'energy': engs,
                    'frequency': frqs,
                    # Performance metrics
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'confusion_matrix': cm.tolist()
                })
            except Exception as e:
                print(f"Error processing {os.path.basename(file)}: {e}")
                continue
    
    # --- CREATE METRICS SUMMARY ---
    metrics_data = []
    for result in results:
        metrics_data.append({
            'subject': result['subject'],
            'activity': result['activity_type'],
            'wrist': result['wrist'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'accuracy': result['accuracy']
        })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Save metrics summary
    plots_dir = os.path.join(DATASET_PATH, "Plots")
    os.makedirs(plots_dir, exist_ok=True)
    metrics_path = os.path.join(plots_dir, 'performance_metrics.csv')
    df_metrics.to_csv(metrics_path, index=False)
    
    print("\n=== OVERALL SUMMARY ===")
    print("\nBy Wrist:")
    print(df_metrics.groupby("wrist")[["precision", "recall", "f1", "accuracy"]].mean())
    print("\nBy Activity:")
    print(df_metrics.groupby("activity")[["precision", "recall", "f1", "accuracy"]].mean())
    print("\nBy Subject:")
    print(df_metrics.groupby("subject")[["precision", "recall", "f1", "accuracy"]].mean())

    # --- PLOTTING: one plot per activity ---
    subjects  = ['Hendrik', 'Tanya']
    metrics   = ['probability', 'energy', 'frequency']

    plot_per_activity(results, subjects, metrics)

if __name__ == "__main__":main()
