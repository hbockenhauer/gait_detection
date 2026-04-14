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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from scipy.ndimage import median_filter, uniform_filter1d
import matplotlib.colors as mcolors
import colorsys
import time
import psutil

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.paths import QSENSE_MIXED, PLOTS_DIR, RESULTS_DIR, QSENSE_DATA, QSENSE_EDGE
from utils.hub_utils import safe_hub_load

DATASET_PATH = QSENSE_EDGE
PLOT_DATASET_NAME = os.path.basename(DATASET_PATH)
REPO_NAME = 'yonbrand/ElderNet'
WINDOW_SIZE = 300      #10s at 30Hz
STEP_SIZE = 30          #1s at 30Hz
GAIT_CLASSES = {'Walking', 'Stairs'}
SAMPLE_RATE_QSENSE = 50.0 #Hz
# SMOOTHING_SEC = 10.0
STEP_SEC = STEP_SIZE / 30.0
# N_SMOOTH = int(SMOOTHING_SEC / STEP_SEC)
MIN_BOUT_SEC = 5.0
RESULTS_DIR = os.path.join(RESULTS_DIR, 'ElderNet')

CONF_THRESH = 0.65
MIN_ENERGY = 0.07
MAX_ENERGY = 0.4
MIN_FREQ = 0.5
MAX_FREQ = 3.5


def normalize_subject_name(name):
    return str(name).strip().title()


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

    parent_folder = os.path.basename(os.path.dirname(filepath))

    # --- CREATE DATETIME COLUMN ---
    df['datetime'] = pd.to_datetime(
        df.iloc[:, 0].astype(str) + ' ' + df.iloc[:, 1].astype(str),
        errors='coerce'
    )

    # Step 0: Remove backwards-jump blocks
    timestamps = df['datetime']
    running_max = timestamps.iloc[0]
    keep_mask = []
    for t in timestamps:
        if t < running_max:
            keep_mask.append(False)
        else:
            keep_mask.append(True)
            running_max = t
    df = df[keep_mask].reset_index(drop=True)

    # Step 1: Fix time travelers (>100 day jumps)
    dt = df['datetime'].diff()
    jump_idx = dt[abs(dt) > pd.Timedelta(days=100)].index
    for idx in jump_idx:
        false_gap = dt[idx] - pd.Timedelta(seconds=1/50)
        df.loc[idx:, 'datetime'] = df.loc[idx:, 'datetime'] - false_gap
        dt = df['datetime'].diff()

    # Step 2: Sort
    df = df.sort_values('datetime').reset_index(drop=True)

    # Step 3: Remove duplicates
    df = df.drop_duplicates(subset='datetime', keep='first').reset_index(drop=True)

    time_seconds = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds().values

    # -------- CASE 1: Sample-level label --------
    if 'label' in df.columns or 'Label' in df.columns:
        label_col = 'label' if 'label' in df.columns else 'Label'
        sample_gt = pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(int)

    # -------- CASE 2: Folder-level activity --------
    else:
        activity_name = parent_folder.split('_')[0]
        # Use len(df) AFTER filtering, not before
        sample_gt = np.ones(len(df), dtype=int) if activity_name in GAIT_CLASSES else np.zeros(len(df), dtype=int)

    data = pd.DataFrame({
        'time_sec': time_seconds,
        'accX': pd.to_numeric(df['accX'], errors='coerce'),
        'accY': pd.to_numeric(df['accY'], errors='coerce'),
        'accZ': pd.to_numeric(df['accZ'], errors='coerce'),
        'gt': sample_gt,
        'energy': pd.to_numeric(df['Energy'], errors='coerce')
    })

    # Remove first 10s of data to avoid initial noise/artifacts
    #data = data[data['time_sec'] >= 10.0].reset_index(drop=True)

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
        'gt': np.round(np.interp(new_time, t, df['gt'].values)).astype(int),
        'energy': np.interp(new_time, t, df['energy'].values)
    })

    return resampled
    
# --- PREPARE WINDOWS FOR ELDERNET ---
def prepare_windows_overlapping(df, window_size, step_size, nfft=512):
    acc_data = df[['accX', 'accY', 'accZ']].values
    gt_raw = df['gt'].values
    times = df['time_sec'].values
    q_energies = df['energy'].values
    
    windows, energies, freqs, activities, timestamps, Q_energies = [], [], [], [], [], []

    def get_dominant_freq(win, fs=30, nfft_size=512):
        mag = np.sqrt(np.sum(win**2, axis=0))
        mag = mag - np.mean(mag)
        hann_win = np.hanning(len(mag))
        mag_windowed = mag * hann_win
        freqs_fft = np.fft.rfftfreq(nfft_size, d=1/fs)
        power_spectrum = np.abs(np.fft.rfft(mag_windowed, n=nfft_size))**2
        return freqs_fft[np.argmax(power_spectrum)]

    # --- Find continuous segments (skip across gaps) ---
    dt = np.diff(times)
    gap_threshold = 0.1  # seconds
    gap_indices = np.where(dt > gap_threshold)[0] + 1
    segment_boundaries = np.concatenate([[0], gap_indices, [len(acc_data)]])
    segments = []
    for k in range(len(segment_boundaries) - 1):
        start = segment_boundaries[k]
        end   = segment_boundaries[k + 1]
        if end - start >= window_size:
            segments.append((start, end))

    # --- Window within each segment only ---
    for (seg_start, seg_end) in segments:
        for i in range(seg_start + window_size, seg_end, step_size):
            win     = acc_data[i - window_size:i]
            act_win = gt_raw[i - window_size:i]

            windows.append(win.T)
            energies.append(np.std(np.sqrt(np.sum((win)**2, axis=1)))*9.81**3)
            freqs.append(get_dominant_freq(win.T, fs=30, nfft_size=nfft))
            activities.append(int(np.mean(act_win) > 0.5))
            timestamps.append(times[i-1])
            Q_energies.append(np.mean(q_energies[i - window_size:i]))

    return (
        torch.FloatTensor(np.array(windows)),
        np.array(energies),
        np.array(freqs),
        activities,
        np.array(timestamps),
        np.array(Q_energies)
    )

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
    color_palette = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    subject_colors = {
        subject: color_palette[idx % len(color_palette)]
        for idx, subject in enumerate(subjects)
    }
    wrist_styles = {'right': '-', 'left': '--'}

    # --- ONE PLOT PER ACTIVITY ---
    for activity_type, activity_results in activities_by_type.items():
        
        # Create figure: metrics + GT/Pred row
        n_rows = len(metrics) + 1  # +1 for GT/Pred row
        fig, axes = plt.subplots(n_rows, 1, 
                                  figsize=(16, 4 * n_rows), 
                                  sharex=True)

        x_min = None
        x_max = None
        for result in activity_results:
            for key in ['raw_timestamps', 'timestamps']:
                t = np.asarray(result.get(key, []), dtype=float)
                t = t[np.isfinite(t)]
                if len(t) == 0:
                    continue
                t_min = float(np.min(t))
                t_max = float(np.max(t))
                x_min = t_min if x_min is None else min(x_min, t_min)
                x_max = t_max if x_max is None else max(x_max, t_max)
        
        if n_rows == 1:
            axes = [axes]  # Ensure iterable
        
        fig.suptitle(f"Activity: {activity_type}", fontsize=16, fontweight='bold')
        
        has_data = False
        
        for ax, metric in zip(axes[:-1], metrics):
            
            # Plot each subject
            for subject in subjects:
                
                # Filter results for this subject and metric
                subject_results = [
                    r for r in activity_results
                    if normalize_subject_name(r['subject']) == normalize_subject_name(subject)
                ]
                
                if not subject_results:
                    print(f"  No data for {subject} in {activity_type}")
                    continue
                
                # Plot each wrist
                for result in subject_results:
                    if metric not in ['probability', 'energy', 'Q_energies', 'frequency']:
                        continue
                    
                    has_data = True
                    timestamps = result['timestamps']
                    
                    if metric == 'probability':
                        values = result['probability']
                        x = result['timestamps']        # window-level
                    elif metric == 'energy':
                        values = result['energy']
                        x = result['timestamps']        # window-level
                    # elif metric == 'Q_energy':
                    #     values = result['Q_energy']
                    #     x = result['raw_timestamps']    # sample-level — matches Q_energy length
                    elif metric == 'Q_energies':
                        values = result['Q_energies']
                        x = result['timestamps']        # window-level

                    elif metric == 'frequency':
                        values = result['frequency']
                        x = result['timestamps']        # window-level

                    wrist = result['wrist']
                    color = subject_colors[normalize_subject_name(subject)]
                    style = wrist_styles.get(wrist, '-')
                    label_raw = f"{normalize_subject_name(subject)} | {wrist}"

                    ax.plot(x, values,
                            color=color,
                            linestyle=style,
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
                
            elif metric == 'Q_energy':  #Raw Q-energy at sample level
                ax.axhline(MIN_ENERGY,
                        color='black',
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.8,
                        label=f'Min Q-energy = {MIN_ENERGY}')
                ax.axhline(MAX_ENERGY,
                        color='black',
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.8,
                        label=f'Max Q-energy = {MAX_ENERGY}')

            elif metric == 'Q_energies':  #Window-level Q-energy
                ax.axhline(MIN_ENERGY,
                        color='black',
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.8,
                        label=f'Min Q-energy = {MIN_ENERGY}')
                ax.axhline(MAX_ENERGY,
                        color='black',
                        linestyle='--',
                        linewidth=1.5,
                        alpha=0.8,
                        label=f'Max Q-energy = {MAX_ENERGY}')

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
            subject_results = [
                r for r in activity_results
                if normalize_subject_name(r['subject']) == normalize_subject_name(subject)
            ]
            if not subject_results:
                continue

            for result in subject_results:
                timestamps = result['timestamps']
                timestamps_raw = result['raw_timestamps']
                y_true_raw = result['raw_gt']
                y_pred = result['y_pred']
                wrist = result['wrist']

                subj_label = normalize_subject_name(subject)
                base_color = subject_colors[subj_label]
                line_style = wrist_styles.get(wrist, '-')

                # --- Ground Truth (thick solid band) ---
                ax_gt_pred.fill_between(
                    timestamps_raw,
                    0,
                    y_true_raw,
                    step='post',
                    alpha=0.25,
                    color=base_color,
                    label=f'{subj_label} | {wrist} | GT'
                )

                # --- Prediction (sharp line, slightly offset) ---
                ax_gt_pred.step(
                    timestamps,
                    y_pred + 0.05,
                    where='post',
                    color=base_color,
                    linestyle=line_style,
                    linewidth=2.5,
                    label=f'{subj_label} | {wrist} | Pred'
                )

        ax_gt_pred.set_ylabel("GT / Prediction", fontsize=12)
        ax_gt_pred.set_ylim(-0.1, 1.15)
        ax_gt_pred.grid(True, alpha=0.3)
        ax_gt_pred.set_xlabel("Time (seconds)", fontsize=12)

        if x_min is not None and x_max is not None and x_max > x_min:
            for ax in axes:
                ax.set_xlim(x_min, x_max)
                
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
        plots_dir = os.path.join(PLOTS_DIR, PLOT_DATASET_NAME, 'eldernet')
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, f"activity_{activity_type}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')        
        plt.show()

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size (actual RAM used)
    return mem_bytes


# --- RUN ELDERNET AND OBTAIN PROBABILITIES ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = safe_hub_load(REPO_NAME, 'eldernet_ft', trust_repo=True).to(device)
    model.eval()

    results = []

    for folder in os.listdir(DATASET_PATH):
        if not os.path.isdir(os.path.join(DATASET_PATH, folder)):
            continue

        # Extract subject name and activity type from folder name
        parts = folder.split('_')
        activity_type = '_'.join(parts[:-1]) if len(parts) > 1 else folder
        subject = normalize_subject_name(parts[-1] if len(parts) > 1 else 'Unknown')

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
                Q_energy = df_30hz['energy'].values
                wins, engs, frqs, acts, tmstps, Q_energies = prepare_windows_overlapping(df_30hz, window_size=WINDOW_SIZE, step_size=STEP_SIZE)

                with torch.no_grad():
                    logits = model(wins.to(device))
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                    # Save per-window outputs
                    output_df = pd.DataFrame({
                        "timestamp": tmstps,
                        "probability": probs,
                        "energy": engs,
                        "frequency": frqs,
                        "Q_energies": Q_energies
                    })

                    save_path = os.path.join(DATASET_PATH, folder, f"{wrist}_window_outputs.csv")
                    output_df.to_csv(save_path, index=False)

                print(f"             Processed {os.path.basename(file)}: {len(probs)} windows")

                y_pred = (probs > CONF_THRESH).astype(int)

                # --- Robust computation of window-level GT to match number of windows ---
                n_windows = len(probs)  # number of predicted windows
                y_true_full = df_30hz['gt'].values

                # Compute start indices of each window
                start_idxs = np.arange(0, n_windows * STEP_SIZE, STEP_SIZE)
                # Ensure we don't go beyond the signal
                start_idxs = start_idxs[start_idxs + WINDOW_SIZE <= len(y_true_full)]

                y_true = np.array([
                    int(np.mean(y_true_full[i:i + WINDOW_SIZE]) > 0.5)
                    for i in start_idxs
                ])
                timestamps = df_30hz['time_sec'].values[start_idxs]

                # If y_true ends up shorter than y_pred due to truncation at end, pad last value
                if len(y_true) < len(y_pred):
                    n_pad = len(y_pred) - len(y_true)
                    y_true = np.pad(y_true, (0, n_pad), mode='edge')
                    timestamps = np.pad(timestamps, (0, n_pad), mode='edge')


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
                    'raw_timestamps': df_30hz['time_sec'].values,
                    'raw_gt': df_30hz['gt'].values,
                    'Q_energy': Q_energy,
                    'timestamps': timestamps,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'probability': probs,
                    'energy': engs,
                    'frequency': frqs,
                    'Q_energies': Q_energies,
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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_path = os.path.join(RESULTS_DIR, f'eldernet_{PLOT_DATASET_NAME}_metrics.csv')
    df_metrics.to_csv(metrics_path, index=False)
    print(f"Saved metrics summary to: {metrics_path}")

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
        return {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'accuracy': acc,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
        }

    def pooled_by_group(rows, key):
        groups = {}
        for row in rows:
            g = row.get(key)
            if g is None:
                continue
            groups.setdefault(g, []).append(row)

        out = []
        for g, g_rows in sorted(groups.items()):
            m = pooled_metrics(g_rows)
            m[key] = g
            out.append(m)

        cols = [key, 'precision', 'recall', 'f1', 'accuracy', 'tn', 'fp', 'fn', 'tp']
        return pd.DataFrame(out, columns=cols)
    
    print("\n=== OVERALL SUMMARY ===")
    print("\nBy Wrist:")
    print(pooled_by_group(results, 'wrist').set_index('wrist')[['precision', 'recall', 'f1', 'accuracy']])
    print("\nBy Activity:")
    print(pooled_by_group(results, 'activity_type').set_index('activity_type')[['precision', 'recall', 'f1', 'accuracy']])
    print("\nBy Subject:")
    print(pooled_by_group(results, 'subject').set_index('subject')[['precision', 'recall', 'f1', 'accuracy']])
    print("\nGlobal (pooled counts):")
    gm = pooled_metrics(results)
    print(pd.Series({
        'precision': gm['precision'],
        'recall': gm['recall'],
        'f1': gm['f1'],
        'accuracy': gm['accuracy'],
    }))

    # --- PLOTTING: one plot per activity ---
    subjects = sorted({normalize_subject_name(r['subject']) for r in results})
    metrics   = ['probability', 'energy', 'Q_energies', 'frequency']

    plot_per_activity(results, subjects, metrics)

if __name__ == "__main__":main()