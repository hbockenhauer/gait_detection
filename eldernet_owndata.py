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
import time
import psutil

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
MIN_BOUT_SEC = 5.0

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

    parent_folder = os.path.basename(os.path.dirname(filepath))

    # --- CREATE DATETIME COLUMN ---
    df['datetime'] = pd.to_datetime(
        df.iloc[:, 0].astype(str) + ' ' + df.iloc[:, 1].astype(str),
        errors='coerce'
    )

    # Remove rows with invalid timestamps
    df = df.dropna(subset=['datetime'])

    # Sort chronologically (fix jump-backs)
    df = df.sort_values('datetime')

    # Remove duplicate timestamps (keep first instance only)
    df = df.drop_duplicates(subset='datetime', keep='first')

    df = df.reset_index(drop=True)

    # Convert to seconds relative to start
    time_seconds = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds().values  

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
        'gt': np.round(np.interp(new_time, t, df['gt'].values)).astype(int)
    })

    return resampled
    
# --- PREPARE WINDOWS FOR ELDERNET ---
def prepare_windows_overlapping(df):
    acc_data = df[['accX', 'accY', 'accZ']].values
    gt_raw = df['gt'].values
    times = df['time_sec'].values
    windows, energies, freqs, activities, timestamps = [], [], [], [], []
    
    def get_dominant_freq(win, fs=30):
        mag = np.sqrt(np.sum(win**2, axis=0))
        mag = mag - np.mean(mag)
        freqs_fft = np.fft.rfftfreq(len(mag), d=1/fs)
        fft_vals = np.abs(np.fft.rfft(mag))
        return freqs_fft[np.argmax(fft_vals)]

    # Create overlapping windows by looking at previous 10s of data for each timestamp   
    for i in range(WINDOW_SIZE, len(acc_data), STEP_SIZE):
        win = acc_data[i - WINDOW_SIZE:i]
        act_win = gt_raw[i - WINDOW_SIZE:i]
        
        windows.append(win.T)
        energies.append(np.std(np.sqrt(np.sum(win**2, axis=1))))
        freqs.append(get_dominant_freq(win.T))
        activities.append(int(np.mean(act_win) > 0.5))

        timestamps.append(times[i-1])

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
# def apply_bout_filtering(predictions, min_bout_length):
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

def segment_walking(probs, step_sec=1.0, smoothing_window=8, high_thresh=0.65, low_thresh=0.45, min_bout_sec=5, max_gap_sec=2):
    """
    Robust walking segmentation using:
    - Moving average smoothing
    - Hysteresis thresholding
    - Gap merging
    - Minimum bout filtering
    """

    import numpy as np
    from scipy.ndimage import uniform_filter1d

    # --------------------------------------------------
    # 1) Smooth probabilities
    # --------------------------------------------------
    probs_smooth = uniform_filter1d(probs, size=smoothing_window)

    # --------------------------------------------------
    # 2) Hysteresis thresholding
    # --------------------------------------------------
    binary = np.zeros_like(probs_smooth, dtype=int)

    walking = False
    for i in range(len(probs_smooth)):
        if not walking and probs_smooth[i] >= high_thresh:
            walking = True
        elif walking and probs_smooth[i] < low_thresh:
            walking = False

        binary[i] = int(walking)

    # --------------------------------------------------
    # 3) Merge small gaps
    # --------------------------------------------------
    max_gap_windows = int(max_gap_sec / step_sec)

    i = 0
    while i < len(binary):
        if binary[i] == 0:
            start = i
            while i < len(binary) and binary[i] == 0:
                i += 1
            gap_length = i - start

            # If surrounded by walking and gap is short → fill it
            if (
                start > 0 and
                i < len(binary) and
                gap_length <= max_gap_windows and
                binary[start - 1] == 1 and
                binary[i] == 1
            ):
                binary[start:i] = 1
        else:
            i += 1

    # --------------------------------------------------
    # 4) Remove short bouts
    # --------------------------------------------------
    min_bout_windows = int(min_bout_sec / step_sec)

    i = 0
    while i < len(binary):
        if binary[i] == 1:
            start = i
            while i < len(binary) and binary[i] == 1:
                i += 1
            bout_length = i - start

            if bout_length < min_bout_windows:
                binary[start:i] = 0
        else:
            i += 1

    return binary, probs_smooth

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
                timestamps_raw = result['raw_timestamps']
                y_true_raw = result['raw_gt']
                y_pred = result['y_pred']
                wrist = result['wrist']

                # --- Color encodes wrist ---
                if wrist == 'right':
                    base_color = '#1f77b4' 
                else:
                    base_color = '#ff7f0e'  

                # --- Ground Truth (thick solid band) ---
                ax_gt_pred.fill_between(
                    timestamps_raw,
                    0,
                    y_true_raw,
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

def viterbi_smooth_gait(probs, energy, freq, 
                        transition_cost=0.3,
                        min_gait_prob=0.5,
                        energy_weight=0.2,
                        freq_weight=0.1):
    """
    Viterbi-like algorithm for gait detection with state persistence.
    
    Key insight: Transitions between gait/non-gait states are "expensive"
    → model prefers to stay in current state unless strong evidence to switch
    
    Parameters:
    -----------
    probs : array
        ElderNet probabilities
    energy, freq : array
        Additional features
    transition_cost : float
        Cost of changing state (higher = more stable/persistent states)
        0.0 = no persistence (just threshold)
        0.5 = very sticky states
    min_gait_prob : float
        Base threshold for gait
    energy_weight, freq_weight : float
        How much to weight these features
    
    Returns:
    --------
    states : array
        Binary gait predictions (0/1)
    """
    n = len(probs)
    
    # Normalize features to [0, 1]
    energy_norm = np.clip((energy - 0.07) / (0.4 - 0.07), 0, 1)
    freq_score = np.exp(-0.5 * ((freq - 2.0) / 0.8)**2)  # Gaussian centered at 2 Hz
    
    # Combined evidence for "gait-ness"
    gait_evidence = (
        probs + 
        energy_weight * energy_norm + 
        freq_weight * freq_score
    ) / (1 + energy_weight + freq_weight)
    
    # Viterbi forward pass
    states = np.zeros(n, dtype=int)
    current_state = 0  # Start in non-gait
    
    for t in range(n):
        evidence_gait = gait_evidence[t]
        evidence_nongait = 1 - gait_evidence[t]
        
        if current_state == 0:  # Currently non-gait
            # Cost to stay non-gait
            cost_stay = evidence_nongait
            # Cost to switch to gait (pay transition cost)
            cost_switch = evidence_gait - transition_cost
            
            if cost_switch > cost_stay:
                current_state = 1
                
        else:  # Currently gait
            # Cost to stay gait
            cost_stay = evidence_gait
            # Cost to switch to non-gait
            cost_switch = evidence_nongait - transition_cost
            
            if cost_switch > cost_stay:
                current_state = 0
        
        states[t] = current_state
    
    return states

# --- APPLY WITH BOUT FILTERING ---
def apply_bout_constraints(states, min_bout_sec=3.0, max_gap_sec=2.0, step_sec=1.0):
    """
    Post-process states:
    1. Remove bouts shorter than min_bout_sec
    2. Merge gaps shorter than max_gap_sec
    """
    min_bout_windows = int(min_bout_sec / step_sec)
    max_gap_windows = int(max_gap_sec / step_sec)
    
    states = states.copy()
    
    # --- MERGE SHORT GAPS ---
    i = 0
    while i < len(states):
        if states[i] == 0:
            start = i
            while i < len(states) and states[i] == 0:
                i += 1
            gap_length = i - start
            
            # Fill short gaps between gait bouts
            if (start > 0 and i < len(states) and 
                gap_length <= max_gap_windows and
                states[start-1] == 1 and states[i] == 1):
                states[start:i] = 1
        else:
            i += 1
    
    # --- REMOVE SHORT BOUTS ---
    i = 0
    while i < len(states):
        if states[i] == 1:
            start = i
            while i < len(states) and states[i] == 1:
                i += 1
            bout_length = i - start
            
            if bout_length < min_bout_windows:
                states[start:i] = 0
        else:
            i += 1
    
    return states

def ensemble_eldernet_signal_processing(
    eldernet_prob,
    sp_predictions,  # From your signal processing method
    energy,
    freq,
    step_sec=1.0
):
    """
    Adaptive ensemble: trust ElderNet when signal is clean/regular,
    trust signal processing when signal is noisy/irregular.
    
    Key insight: ElderNet works best on regular gait.
    Signal processing works best on irregular/variable gait.
    """
    
    # --- COMPUTE SIGNAL QUALITY METRICS ---
    
    # 1. Stability: How consistent is the probability?
    prob_smooth = uniform_filter1d(eldernet_prob, size=10)
    prob_std = uniform_filter1d((eldernet_prob - prob_smooth)**2, size=10)**0.5
    stability = np.exp(-5 * prob_std)  # High when stable (regular gait)
    
    # 2. Regularity: Is frequency consistent and in gait range?
    freq_smooth = uniform_filter1d(freq, size=10)
    freq_std = uniform_filter1d((freq - freq_smooth)**2, size=10)**0.5
    freq_in_range = ((freq > 0.8) & (freq < 2.5)).astype(float)
    freq_regularity = np.exp(-3 * freq_std) * freq_in_range
    
    # 3. Energy consistency
    energy_smooth = uniform_filter1d(energy, size=10)
    energy_std = uniform_filter1d((energy - energy_smooth)**2, size=10)**0.5
    energy_stability = np.exp(-10 * energy_std)
    
    # Combined signal quality score
    signal_quality = (
        0.4 * stability + 
        0.4 * freq_regularity + 
        0.2 * energy_stability
    )
    
    # --- ADAPTIVE WEIGHTING ---
    # When signal quality is HIGH → trust ElderNet (learned features)
    # When signal quality is LOW → trust signal processing (explicit rules)
    
    eldernet_weight = signal_quality
    sp_weight = 1 - signal_quality
    
    # Normalize weights
    total_weight = eldernet_weight + sp_weight
    eldernet_weight /= total_weight
    sp_weight /= total_weight
    
    # --- ENSEMBLE DECISION ---
    # Weighted vote
    ensemble_score = (
        eldernet_weight * eldernet_prob + 
        sp_weight * sp_predictions
    )
    
    # Threshold
    y_pred = (ensemble_score > 0.5).astype(int)
    
    return y_pred, ensemble_score, signal_quality, eldernet_weight

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size (actual RAM used)
    return mem_bytes


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

                # start_time = time.perf_counter()

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

                # end_time = time.perf_counter()
                # total_latency = end_time - start_time
                # avg_per_window = total_latency / len(wins)

                # print(f"Total Inference Time: {total_latency:.4f}s")
                # print(f"Time per 1s window: {avg_per_window * 1000:.2f}ms")

                # bytes_used = get_memory_usage()
                # print(f"System RAM: {bytes_used} bytes")
                # print(f"System RAM: {bytes_used / (1024**2):.2f} MB")

        
                # Compute predictions
                # # Apply smoothing and temporal thresholding to get final predictions
                # _, probs_smoothed = segment_walking(probs, step_sec=STEP_SEC, smoothing_window=8, high_thresh=0.75, low_thresh=0.55, min_bout_sec=5, max_gap_sec=2)
                # rolling_std = uniform_filter1d((probs - probs_smoothed)**2, size=5)**0.5
                # high_conf = probs_smoothed > 0.75

                # mid_conf = (
                #     (probs_smoothed > 0.1) &
                #     (engs > MIN_ENERGY) &
                #     (frqs > MIN_FREQ) &
                #     (frqs < MAX_FREQ)
                # )

                # stable = rolling_std < 0.1

                # y_pred = ((high_conf & stable) | (mid_conf & stable)).astype(int)

                # y_pred = ((engs > MIN_ENERGY) & (engs < MAX_ENERGY) & 
                #        (frqs > MIN_FREQ) & (frqs < MAX_FREQ)).astype(int)                                          

                # y_pred = ((probs > CONF_THRESH) & (engs > MIN_ENERGY) & (engs < MAX_ENERGY) & 
                #        (frqs > MIN_FREQ) & (frqs < MAX_FREQ)).astype(int)

                y_pred = (probs > 0.65).astype(int)

                # probs_smoothed = uniform_filter1d(probs, size=8)
                # rolling_std = uniform_filter1d((probs - probs_smoothed)**2, size=5)**0.5

                # # ---- Normalize features ----
                # eng_norm  = (engs - MIN_ENERGY) / (np.percentile(engs, 95) - MIN_ENERGY)
                # eng_norm  = np.clip(eng_norm, 0, 1)

                # freq_center = 2.0  # typical walking cadence
                # freq_width  = 0.8
                # freq_score  = np.exp(-0.5 * ((frqs - freq_center)/freq_width)**2)

                # stability_score = np.exp(-5 * rolling_std)   # penalize instability

                # # ---- Normalize features ----
                # eng_norm  = (engs - MIN_ENERGY) / (np.percentile(engs, 95) - MIN_ENERGY)
                # eng_norm  = np.clip(eng_norm, 0, 1)

                # freq_center = 2.0  # typical walking cadence
                # freq_width  = 0.8
                # freq_score  = np.exp(-0.5 * ((frqs - freq_center)/freq_width)**2)

                # stability_score = np.exp(-5 * rolling_std)   # penalize instability

                # # ---- Weighted fusion ----
                # w_prob = 0.6
                # w_eng  = 0.15
                # w_freq = 0.15
                # w_stab = 0.10

                # fused_prob = (
                #     w_prob * probs_smoothed +
                #     w_eng  * eng_norm +
                #     w_freq * freq_score +
                #     w_stab * stability_score
                # )

                # y_pred, fused_smooth = segment_walking(
                #     fused_prob,
                #     step_sec=STEP_SEC,
                #     smoothing_window=5,
                #     high_thresh=0.6,
                #     low_thresh=0.45,
                #     min_bout_sec=5,
                #     max_gap_sec=2
                # )

                # --- METHOD 1: Viterbi-like smoothing (no training needed) ---
                # y_pred_raw = viterbi_smooth_gait(
                #     probs, 
                #     engs, 
                #     frqs,
                #     transition_cost=0.15,     # Tune this: higher = stickier states
                #     min_gait_prob=0.75,
                #     energy_weight=0.2,
                #     freq_weight=0.1
                # )

                # # Apply bout constraints
                # y_pred = apply_bout_constraints(
                #     y_pred_raw,
                #     min_bout_sec=3.0,   # Minimum 3-second bouts
                #     max_gap_sec=2.0,    # Merge gaps < 2 seconds
                #     step_sec=STEP_SEC
                # )

                # sp_pred = ((engs > MIN_ENERGY) & (engs < MAX_ENERGY) & (frqs > MIN_FREQ) & (frqs < MAX_FREQ)).astype(int)

                # y_pred, ensemble_score, signal_quality, eldernet_weight = ensemble_eldernet_signal_processing(
                #     eldernet_prob=probs,
                #     sp_predictions=sp_pred,  # From signal processing
                #     energy=engs,
                #     freq=frqs,
                #     step_sec=STEP_SEC
                # )

                # Optional: Apply bout filtering on ensemble output
                # y_pred = apply_bout_constraints(
                #     y_pred,
                #     min_bout_sec=3.0,
                #     max_gap_sec=2.0,
                #     step_sec=STEP_SEC
                # )

                
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
