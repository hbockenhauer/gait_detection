import torch
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import median_filter
import os
import pickle
import random
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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
    mag = mag - np.mean(mag) # Detrend
    freqs = np.fft.rfftfreq(len(mag), d=1/fs)
    fft_vals = np.abs(np.fft.rfft(mag))
    return freqs[np.argmax(fft_vals)]

def prepare_batch_windows(df, window_size=300):
    X_windows, energy_values, freq_values, y_true, codes = [], [], [], [], []
    GAIT_CODES = {'A', 'C'} # Walk, Stairs
    
    raw_xyz = df[['x', 'y', 'z']].values / 9.81
    raw_labels = df['activity'].values
    
    # Resample 20Hz -> 30Hz
    new_samples = int(len(raw_xyz) * (30 / 20))
    resampled_data = signal.resample(raw_xyz, new_samples)
    indices = np.linspace(0, len(raw_labels) - 1, new_samples).astype(int)
    resampled_labels = raw_labels[indices]
    
    for i in range(0, len(resampled_data) - window_size + 1, window_size):
        win = resampled_data[i : i + window_size]
        label_win = resampled_labels[i : i + window_size]
        unique, counts = np.unique(label_win, return_counts=True)
        majority_label = unique[np.argmax(counts)]
        
        y_true.append(1 if majority_label in GAIT_CODES else 0)
        codes.append(majority_label)
        energy_values.append(np.std(np.sqrt(np.sum(win**2, axis=1))))
        freq_values.append(get_dominant_freq(win.T, fs=30))
        X_windows.append(win.T)
        
    return torch.FloatTensor(np.array(X_windows)), np.array(energy_values), np.array(freq_values), np.array(y_true), codes

# --- 2. BATCH CONFIG ---
FOLDER_PATH = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\wisdm-dataset\raw\watch\accel'
ACTIVITY_MAP = {'A':'Walk', 'B':'Jog', 'C':'Stairs', 'D':'Sit', 'E':'Stand', 'F':'Type', 'G':'Teeth', 'H':'Soup', 'I':'Chips', 'J':'Pasta', 'K':'Drink', 'L':'Sandwich', 'M':'Kicking', 'O':'Catch', 'P':'Dribbling', 'Q':'Writing', 'R':'Clapping', 'S':'Folding'}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('yonbrand/ElderNet', 'eldernet_ft').to(device)
model.eval()

results_list = []
files = sorted([f for f in os.listdir(FOLDER_PATH) if f.endswith('.txt')])
print(f"Locked & Seeded. Processing {len(files)} subjects...")

# --- 3. EVALUATION LOOP ---
try:
    for filename in files:
        try:
            subject_id = filename.split('_')[1]
            df = load_wisdm_cleaned(os.path.join(FOLDER_PATH, filename))
            windows, energies, freqs, y_true, codes = prepare_batch_windows(df)
            
            with torch.no_grad():
                probs = torch.softmax(model(windows.to(device)), dim=1)[:, 1].cpu().numpy()

            # Logic Gates
            CONF_THRESH = 0.4
            ENERGY_THRESH_MIN = 0.18
            ENERGY_THRESH_MAX = 0.65
            min_freq = 0.5
            max_freq = 3.0
            probs_smoothed = np.convolve(probs, np.ones(3)/3, mode='same')
            y_pred_raw = ((probs_smoothed > CONF_THRESH) & (energies > ENERGY_THRESH_MIN) & (energies < ENERGY_THRESH_MAX) & (freqs > min_freq) & (freqs < max_freq)).astype(int)
            y_pred = median_filter(y_pred_raw, size=3)

            # Break down by activity
            activity_stats = []
            activity_points = []
            for code in np.unique(codes):
                idx = [i for i, c in enumerate(codes) if c == code]
                # Precision/Recall for THIS specific activity
                p_a, r_a, f_a, _ = precision_recall_fscore_support(np.array(y_true)[idx], y_pred[idx], labels=[1], average='binary', zero_division=0)
                activity_stats.append({'Activity': ACTIVITY_MAP.get(code, code), 'Precision': p_a, 'Recall': r_a, 'F1': f_a})

                # Mean descriptors for plotting
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

            # Subject-wide Metrics
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average='binary', zero_division=0)
            results_list.append({
                'Subject': subject_id,
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': p, 'Recall': r, 'F1': f1,
                'ActivityDetails': activity_stats,
                'ActivityPoints': activity_points,
                'EnergiesSeq': energies.tolist() if hasattr(energies, 'tolist') else list(energies),
                'FreqsSeq': freqs.tolist() if hasattr(freqs, 'tolist') else list(freqs),
                'ConfSeq': probs_smoothed.tolist() if hasattr(probs_smoothed, 'tolist') else list(probs_smoothed),
                'CodesSeq': list(codes)
            })

            # save intermediate results
            try:
                with open('results_partial.pkl', 'wb') as fh:
                    pickle.dump(results_list, fh)
            except Exception:
                pass

            print(f"✅ ID {subject_id}: Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}")

        except Exception as e:
            print(f"❌ Error {filename}: {e}")
except KeyboardInterrupt:
    print('\nInterrupted by user; proceeding to plotting with collected results...')

# --- 4. GLOBAL SUMMARY ---
results_df = pd.DataFrame(results_list)
print("\n" + "="*50)
print("OVERALL METRICS (MEAN ACROSS SUBJECTS)")
if not results_df.empty:
    print(results_df[['Accuracy', 'Precision', 'Recall', 'F1']].mean())
else:
    print('No subject results collected.')

# --- 5. ACTIVITY BREAKDOWN ---
all_act_data = []
for row in results_list:
    for item in row.get('ActivityDetails', []):
        all_act_data.append(item)

if len(all_act_data) > 0:
    summary_act = pd.DataFrame(all_act_data).groupby('Activity').mean()
    print("\n" + "="*50)
    print("PER-ACTIVITY PERFORMANCE")
    print(summary_act)
else:
    print('No activity details collected.')

# --- 6. Visualization: three subplots (confidence, energy, frequency) for selected subject(s) ---
import matplotlib.pyplot as plt

# output folder for plots
PLOT_DIR = 'ElderNet_WISDM_Plots'
os.makedirs(PLOT_DIR, exist_ok=True)

if len(results_list) == 0:
    # Try to load partial results if they exist
    try:
        with open('results_partial.pkl', 'rb') as fh:
            results_list = pickle.load(fh)
    except Exception:
        pass

if len(results_list) > 0:
    # List available subjects
    print("\n" + "="*50)
    print("AVAILABLE SUBJECTS:")
    for i, subj in enumerate(results_list):
        print(f"  {i}: Subject {subj.get('Subject')}")
    
    # Prompt user to select subject(s)
    def parse_subject_selection(input_str, max_idx):
        """Parse comma-separated or range input (e.g., '0,2,5' or '0-3' or '1')"""
        indices = set()
        for part in input_str.split(','):
            part = part.strip()
            if '-' in part:
                # Range
                try:
                    start, end = part.split('-')
                    start, end = int(start.strip()), int(end.strip())
                    for i in range(start, end + 1):
                        if 0 <= i < max_idx:
                            indices.add(i)
                except ValueError:
                    return None
            else:
                # Single index
                try:
                    i = int(part)
                    if 0 <= i < max_idx:
                        indices.add(i)
                except ValueError:
                    return None
        return sorted(list(indices))
    
    while True:
        choice = input("\nEnter subject index/indices to plot (e.g., '0' or '0,2,5' or '0-3' or 'quit'): ").strip()
        if choice.lower() == 'quit':
            print("Exiting.")
            exit()
        selected_indices = parse_subject_selection(choice, len(results_list))
        if selected_indices:
            break
        else:
            print(f"Invalid input. Please enter valid indices between 0 and {len(results_list)-1}.")
    
    # Create one figure per selected subject
    for idx in selected_indices:
        selected_subj = results_list[idx]
        subj_id = selected_subj.get('Subject')
        
        # Create one figure with three subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        plots_info = [
            ('Confidence', 'ConfSeq', 'Confidence (probability)', axes[0]),
            ('Energy', 'EnergiesSeq', 'Energy (std)', axes[1]),
            ('Frequency', 'FreqsSeq', 'Mean Frequency (Hz)', axes[2])
        ]
        
        codes = selected_subj.get('CodesSeq', [])
        
        for metric_name, key, ylabel, ax in plots_info:
            seq = selected_subj.get(key, [])
            if seq is None or len(seq) == 0:
                ax.text(0.5, 0.5, f'No data for {metric_name}', ha='center', va='center', transform=ax.transAxes)
                continue
            
            x = np.arange(len(seq))
            ax.plot(x, seq, color='steelblue', linewidth=1.5, label=metric_name)
            
            # Mark activity begins with vertical lines and labels
            last = None
            for seq_idx, code in enumerate(codes):
                if seq_idx == 0 or code != last:
                    act_label = ACTIVITY_MAP.get(code, code)
                    ax.axvline(seq_idx, color='red', linestyle='--', alpha=0.4, linewidth=1)
                    yval = seq[seq_idx] if seq_idx < len(seq) else None
                    if yval is not None:
                        ax.text(seq_idx + 0.5, yval, act_label, rotation=90, fontsize=7, color='red', va='bottom')
                last = code
            
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Window index')
            ax.set_title(f"Subject {subj_id} - {metric_name}")
            ax.grid(True, alpha=0.3)
        
        fig.suptitle(f"Subject {subj_id}: Gait Detection Metrics with Activity Annotations", fontsize=14, fontweight='bold')
        fig.tight_layout()
        out_file = os.path.join(PLOT_DIR, f'subject_{subj_id}_gait_detection.png')
        plt.savefig(out_file, dpi=150)
        print(f"Plot saved as '{out_file}'")
        plt.show()
else:
    print('No results to plot.')
