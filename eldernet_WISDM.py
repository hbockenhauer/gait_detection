import torch
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import median_filter
import os
import pickle
import random
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt

# --- 0. REPRODUCIBILITY LOCKDOWN ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# --- 1. CORE FUNCTIONS ---
def load_wisdm_cleaned(path):
    df = pd.read_csv(path, header=None, names=['user', 'activity', 'ts', 'x', 'y', 'z'])
    df['z'] = df['z'].astype(str).str.replace(';', '').astype(float)
    return df

def get_dominant_freq(win, fs=30):
    mag = np.sqrt(np.sum(win**2, axis=0)) 
    mag = mag - np.mean(mag)
    freqs = np.fft.rfftfreq(len(mag), d=1/fs)
    fft_vals = np.abs(np.fft.rfft(mag))
    return freqs[np.argmax(fft_vals)]

def prepare_batch_windows_overlapping(df, window_size=300, step_size=30):
    """
    Prepare overlapping windows from WISDM data
    
    Parameters:
    -----------
    df : DataFrame
        WISDM data with columns ['user', 'activity', 'ts', 'x', 'y', 'z']
    window_size : int
        Window size in samples (default 300 = 10 seconds at 30Hz)
    step_size : int
        Step size in samples (default 30 = 1 second at 30Hz)
    
    Returns:
    --------
    X_windows, energy_values, freq_values, y_true, codes, timestamps
    """
    X_windows, energy_values, freq_values, y_true, codes, timestamps = [], [], [], [], [], []
    GAIT_CODES = {'A', 'C'}  # Walk, Stairs
    
    raw_xyz = df[['x', 'y', 'z']].values / 9.81
    raw_labels = df['activity'].values
    
    # Resample 20Hz -> 30Hz
    new_samples = int(len(raw_xyz) * (30 / 20))
    resampled_data = signal.resample(raw_xyz, new_samples)
    indices = np.linspace(0, len(raw_labels) - 1, new_samples).astype(int)
    resampled_labels = raw_labels[indices]
    
    # Create OVERLAPPING windows
    for i in range(0, len(resampled_data) - window_size + 1, step_size):
        win = resampled_data[i : i + window_size]
        label_win = resampled_labels[i : i + window_size]
        unique, counts = np.unique(label_win, return_counts=True)
        majority_label = unique[np.argmax(counts)]
        
        y_true.append(1 if majority_label in GAIT_CODES else 0)
        codes.append(majority_label)
        energy_values.append(np.std(np.sqrt(np.sum(win**2, axis=1))))
        freq_values.append(get_dominant_freq(win.T, fs=30))
        X_windows.append(win.T)
        timestamps.append(i / 30.0)  # Time in seconds
        
    return (torch.FloatTensor(np.array(X_windows)), 
            np.array(energy_values), 
            np.array(freq_values), 
            np.array(y_true), 
            codes,
            np.array(timestamps))

# --- 2. CONFIGURATION ---
FOLDER_PATH = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\wisdm-dataset\raw\watch\accel'
WINDOW_SIZE = 300
STEP_SIZE = 30  # 1 second steps = 90% overlap
ACTIVITY_MAP = {
    'A':'Walk', 'B':'Jog', 'C':'Stairs', 'D':'Sit', 'E':'Stand', 
    'F':'Type', 'G':'Teeth', 'H':'Soup', 'I':'Chips', 'J':'Pasta', 
    'K':'Drink', 'L':'Sandwich', 'M':'Kicking', 'O':'Catch', 
    'P':'Dribbling', 'Q':'Writing', 'R':'Clapping', 'S':'Folding'
}

# Thresholds
CONF_THRESH = 0.4
ENERGY_THRESH_MIN = 0.18
ENERGY_THRESH_MAX = 0.65
MIN_FREQ = 0.5
MAX_FREQ = 3.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('yonbrand/ElderNet', 'eldernet_ft').to(device)
model.eval()

print("="*80)
print("WISDM DATASET - ELDERNET WITH OVERLAPPING WINDOWS")
print("="*80)
print(f"Window size: {WINDOW_SIZE} samples ({WINDOW_SIZE/30:.1f}s)")
print(f"Step size: {STEP_SIZE} samples ({STEP_SIZE/30:.1f}s)")
print(f"Overlap: {(1 - STEP_SIZE/WINDOW_SIZE)*100:.1f}%")
print("="*80 + "\n")

results_list = []
files = sorted([f for f in os.listdir(FOLDER_PATH) if f.endswith('.txt')])
print(f"Processing {len(files)} subjects with overlapping windows...")

# --- 3. EVALUATION LOOP ---
try:
    for filename in files:
        try:
            subject_id = filename.split('_')[1]
            df = load_wisdm_cleaned(os.path.join(FOLDER_PATH, filename))
            
            # Get overlapping windows
            windows, energies, freqs, y_true, codes, timestamps = prepare_batch_windows_overlapping(
                df, window_size=WINDOW_SIZE, step_size=STEP_SIZE
            )
            
            if len(windows) == 0:
                print(f"⚠️  Subject {subject_id}: No windows generated")
                continue
            
            with torch.no_grad():
                probs = torch.softmax(model(windows.to(device)), dim=1)[:, 1].cpu().numpy()

            # Apply thresholds
            probs_smoothed = np.convolve(probs, np.ones(3)/3, mode='same')
            y_pred_raw = ((probs_smoothed > CONF_THRESH) & 
                         (energies > ENERGY_THRESH_MIN) & 
                         (energies < ENERGY_THRESH_MAX) & 
                         (freqs > MIN_FREQ) & 
                         (freqs < MAX_FREQ)).astype(int)
            y_pred = median_filter(y_pred_raw, size=3)

            # Per-activity breakdown
            activity_stats = []
            activity_points = []
            for code in np.unique(codes):
                idx = [i for i, c in enumerate(codes) if c == code]
                p_a, r_a, f_a, _ = precision_recall_fscore_support(
                    np.array(y_true)[idx], y_pred[idx], 
                    labels=[1], average='binary', zero_division=0
                )
                activity_stats.append({
                    'Activity': ACTIVITY_MAP.get(code, code), 
                    'Precision': p_a, 
                    'Recall': r_a, 
                    'F1': f_a,
                    'Windows': len(idx)
                })

                mean_freq = float(np.mean(freqs[idx])) if len(idx) > 0 else float('nan')
                mean_energy = float(np.mean(energies[idx])) if len(idx) > 0 else float('nan')
                mean_conf = float(np.mean(probs_smoothed[idx])) if len(idx) > 0 else float('nan')
                activity_points.append({
                    'Activity': ACTIVITY_MAP.get(code, code),
                    'Code': code,
                    'MeanFreq': mean_freq,
                    'MeanEnergy': mean_energy,
                    'MeanConf': mean_conf
                })

            # Subject-wide metrics
            p, r, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=[1], average='binary', zero_division=0
            )
            
            results_list.append({
                'Subject': subject_id,
                'NumWindows': len(windows),
                'Duration': timestamps[-1] if len(timestamps) > 0 else 0,
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': p, 
                'Recall': r, 
                'F1': f1,
                'ActivityDetails': activity_stats,
                'ActivityPoints': activity_points,
                'Timestamps': timestamps.tolist(),
                'EnergiesSeq': energies.tolist(),
                'FreqsSeq': freqs.tolist(),
                'ConfSeq': probs_smoothed.tolist(),
                'CodesSeq': list(codes),
                'PredSeq': y_pred.tolist(),
                'TrueSeq': y_true.tolist()
            })

            # Save intermediate results
            try:
                with open('wisdm_results_overlapping.pkl', 'wb') as fh:
                    pickle.dump(results_list, fh)
            except Exception:
                pass

            print(f"✅ Subject {subject_id}: {len(windows)} windows, P={p:.2f}, R={r:.2f}, F1={f1:.2f}")

        except Exception as e:
            print(f"❌ Error {filename}: {e}")
            
except KeyboardInterrupt:
    print('\nInterrupted by user; proceeding to analysis...')

# --- 4. GLOBAL SUMMARY ---
print("\n" + "="*80)
print("GLOBAL PERFORMANCE (WITH OVERLAPPING WINDOWS)")
print("="*80)

results_df = pd.DataFrame(results_list)
if not results_df.empty:
    print(f"Total subjects: {len(results_df)}")
    print(f"Total windows: {results_df['NumWindows'].sum()}")
    print(f"\nMean performance across subjects:")
    print(results_df[['Accuracy', 'Precision', 'Recall', 'F1']].mean())
    print(f"\nStd deviation:")
    print(results_df[['Accuracy', 'Precision', 'Recall', 'F1']].std())
else:
    print('No subject results collected.')

# --- 5. ACTIVITY BREAKDOWN ---
print("\n" + "="*80)
print("PER-ACTIVITY PERFORMANCE (AGGREGATED)")
print("="*80)

all_act_data = []
for row in results_list:
    for item in row.get('ActivityDetails', []):
        all_act_data.append(item)

if len(all_act_data) > 0:
    act_df = pd.DataFrame(all_act_data)
    summary_act = act_df.groupby('Activity').agg({
        'Precision': 'mean',
        'Recall': 'mean',
        'F1': 'mean',
        'Windows': 'sum'
    }).round(3)
    print(summary_act.to_string())
else:
    print('No activity details collected.')

# --- 6. VISUALIZATION ---
PLOT_DIR = 'ElderNet_WISDM_Plots_Overlapping'
os.makedirs(PLOT_DIR, exist_ok=True)

if len(results_list) == 0:
    try:
        with open('wisdm_results_overlapping.pkl', 'rb') as fh:
            results_list = pickle.load(fh)
    except Exception:
        pass

if len(results_list) > 0:
    print("\n" + "="*80)
    print("AVAILABLE SUBJECTS FOR PLOTTING:")
    print("="*80)
    for i, subj in enumerate(results_list):
        subj_id = subj.get('Subject')
        n_windows = subj.get('NumWindows', 0)
        duration = subj.get('Duration', 0)
        print(f"  {i}: Subject {subj_id} ({n_windows} windows, {duration:.1f}s)")
    
    def parse_subject_selection(input_str, max_idx):
        indices = set()
        for part in input_str.split(','):
            part = part.strip()
            if '-' in part:
                try:
                    start, end = part.split('-')
                    start, end = int(start.strip()), int(end.strip())
                    for i in range(start, end + 1):
                        if 0 <= i < max_idx:
                            indices.add(i)
                except ValueError:
                    return None
            else:
                try:
                    i = int(part)
                    if 0 <= i < max_idx:
                        indices.add(i)
                except ValueError:
                    return None
        return sorted(list(indices))
    
    while True:
        choice = input("\nEnter subject index to plot (e.g., '0' or '0,2-5' or 'quit'): ").strip()
        if choice.lower() == 'quit':
            print("Exiting.")
            break
        selected_indices = parse_subject_selection(choice, len(results_list))
        if selected_indices:
            break
        else:
            print(f"Invalid input. Enter indices between 0 and {len(results_list)-1}.")
    
    if choice.lower() != 'quit':
        for idx in selected_indices:
            selected_subj = results_list[idx]
            subj_id = selected_subj.get('Subject')
            
            # Extract sequences
            timestamps = np.array(selected_subj.get('Timestamps', []))
            codes = selected_subj.get('CodesSeq', [])
            conf_seq = np.array(selected_subj.get('ConfSeq', []))
            energy_seq = np.array(selected_subj.get('EnergiesSeq', []))
            freq_seq = np.array(selected_subj.get('FreqsSeq', []))
            y_pred = np.array(selected_subj.get('PredSeq', []))
            y_true = np.array(selected_subj.get('TrueSeq', []))
            
            # Create figure with 4 subplots
            fig, axes = plt.subplots(4, 1, figsize=(16, 12))
            
            # Calculate gait detection mask
            gait_mask = ((conf_seq > CONF_THRESH) & 
                        (energy_seq > ENERGY_THRESH_MIN) & 
                        (energy_seq < ENERGY_THRESH_MAX) & 
                        (freq_seq > MIN_FREQ) & 
                        (freq_seq < MAX_FREQ))
            
            # Find contiguous gait regions
            gait_regions = []
            in_gait = False
            start_idx = 0
            for i, is_gait in enumerate(gait_mask):
                if is_gait and not in_gait:
                    start_idx = i
                    in_gait = True
                elif not is_gait and in_gait:
                    gait_regions.append((start_idx, i - 1))
                    in_gait = False
            if in_gait:
                gait_regions.append((start_idx, len(gait_mask) - 1))
            
            # Activity transitions
            activity_transitions = []
            last_code = None
            for xi, code in enumerate(codes):
                if code != last_code:
                    activity_transitions.append((xi, code))
                    last_code = code
            
            # Plot 1: Confidence
            ax = axes[0]
            for start, end in gait_regions:
                ax.axvspan(timestamps[start], timestamps[end], alpha=0.15, color='gold')
            ax.plot(timestamps, conf_seq, color='steelblue', linewidth=1.5)
            ax.axhline(CONF_THRESH, color='gold', linestyle='--', linewidth=1.5, 
                      alpha=0.7, label=f'Threshold={CONF_THRESH}')
            ax.set_ylabel('Confidence')
            ax.set_title(f'Subject {subj_id} - Gait Detection (Overlapping Windows)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Energy
            ax = axes[1]
            for start, end in gait_regions:
                ax.axvspan(timestamps[start], timestamps[end], alpha=0.15, color='gold')
            ax.plot(timestamps, energy_seq, color='steelblue', linewidth=1.5)
            ax.axhline(ENERGY_THRESH_MIN, color='gold', linestyle='--', linewidth=1.5,
                      alpha=0.7, label=f'Min={ENERGY_THRESH_MIN}')
            ax.axhline(ENERGY_THRESH_MAX, color='gold', linestyle='--', linewidth=1.5,
                      alpha=0.7, label=f'Max={ENERGY_THRESH_MAX}')
            ax.set_ylabel('Energy (std)')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Frequency
            ax = axes[2]
            for start, end in gait_regions:
                ax.axvspan(timestamps[start], timestamps[end], alpha=0.15, color='gold')
            ax.plot(timestamps, freq_seq, color='steelblue', linewidth=1.5)
            ax.axhline(MIN_FREQ, color='gold', linestyle='--', linewidth=1.0, alpha=0.6)
            ax.axhline(MAX_FREQ, color='gold', linestyle='--', linewidth=1.0, alpha=0.6)
            ax.set_ylabel('Dominant Freq (Hz)')
            ax.grid(True, alpha=0.3)
            
            # Plot 4: Predictions vs Ground Truth
            ax = axes[3]
            ax.fill_between(timestamps, 0, y_pred, alpha=0.3, color='blue', 
                           label='Predicted Gait', step='mid')
            ax.fill_between(timestamps, 0, y_true, alpha=0.2, color='green', 
                           label='True Gait', step='mid')
            
            # Mark activity transitions
            for xi, code in activity_transitions:
                if xi < len(timestamps):
                    act_name = ACTIVITY_MAP.get(code, code)
                    line_color = 'green' if code in {'A', 'C'} else 'red'
                    ax.axvline(timestamps[xi], color=line_color, linestyle='--', 
                             alpha=0.4, linewidth=1)
                    ax.text(timestamps[xi], 0.5, act_name, rotation=90, 
                           fontsize=7, color=line_color, va='bottom')
            
            ax.set_ylabel('Gait (0/1)')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylim([-0.1, 1.1])
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            fig.suptitle(f"Subject {subj_id}: Overlapping Window Gait Detection", 
                        fontsize=14, fontweight='bold')
            fig.tight_layout()
            
            out_file = os.path.join(PLOT_DIR, f'subject_{subj_id}_overlapping.png')
            plt.savefig(out_file, dpi=150)
            print(f"Plot saved: {out_file}")
            plt.show()
else:
    print('No results to plot.')

print("\n✅ Analysis complete!")