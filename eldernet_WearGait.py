import os
import glob
import numpy as np
import pandas as pd
import torch
import pickle
from scipy import signal
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import random

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

# --- CONFIGURATION ---
DATA_PATH = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\WearGait-PD'
WINDOW_SIZE = 300
STEP_SIZE = 30
REPO_NAME = 'yonbrand/ElderNet'

# Thresholds
CONF_THRESH = 0.5
ENERGY_THRESH = 0.7
MIN_FREQ = 1.0
MAX_FREQ = 2.5

def parse_time_value(t):
    if pd.isna(t): return np.nan
    if isinstance(t, (int, float)): return float(t)
    if isinstance(t, str):
        t_clean = t.replace('sec', '').strip()
        try: return float(t_clean)
        except ValueError: return np.nan
    return np.nan

def inspect_csv_structure(filepath, max_rows=5):
    for sep in ['\t', ',', ';', ' ']:
        try:
            df = pd.read_csv(filepath, sep=sep, nrows=max_rows)
            if len(df.columns) > 10: return sep, df.columns.tolist()
        except: continue
    return None, None

def load_weargait_data(filepath, wrist='right'):
    sep, columns = inspect_csv_structure(filepath)
    if sep is None: raise ValueError(f"Could not parse CSV file: {filepath}")
    
    df = pd.read_csv(filepath, sep=sep, low_memory=False)
    wrist_prefix = 'R_Wrist' if wrist == 'right' else 'L_Wrist'
    
    def find_column(patterns):
        for pattern in patterns:
            for col in df.columns:
                if pattern.lower() in col.lower(): return col
        return None
    
    time_col = find_column(['Time', 'time', 'timestamp'])
    activity_col = find_column(['GeneralEvent', 'Event', 'Activity', 'Label'])
    acc_x_col = find_column([f'{wrist_prefix}_Acc_X', f'{wrist_prefix}_AccX'])
    acc_y_col = find_column([f'{wrist_prefix}_Acc_Y', f'{wrist_prefix}_AccY'])
    acc_z_col = find_column([f'{wrist_prefix}_Acc_Z', f'{wrist_prefix}_AccZ'])
    
    if not all([acc_x_col, acc_y_col, acc_z_col]):
        raise ValueError(f"Missing {wrist} sensor columns")
    
    data = pd.DataFrame({
        'time_raw': df[time_col],
        'acc_x': pd.to_numeric(df[acc_x_col], errors='coerce'),
        'acc_y': pd.to_numeric(df[acc_y_col], errors='coerce'),
        'acc_z': pd.to_numeric(df[acc_z_col], errors='coerce'),
        'activity': df[activity_col].astype(str) if activity_col else 'Walking'
    })
    
    data['time'] = data['time_raw'].apply(parse_time_value)
    data = data.dropna(subset=['time', 'acc_x', 'acc_y', 'acc_z'])
    data['time'] = data['time'] - data['time'].min()
    return data.drop(columns=['time_raw'])

def detect_sampling_rate(time_series):
    return 1.0 / np.median(np.diff(time_series[:1000]))

def resample_to_30hz(df, original_fs):
    if abs(original_fs - 30.0) < 0.1: return df
    t = df['time'].values
    new_time = np.linspace(t[0], t[-1], int((t[-1] - t[0]) * 30.0) + 1)
    resampled = pd.DataFrame({
        'time': new_time,
        'acc_x': np.interp(new_time, t, df['acc_x'].values),
        'acc_y': np.interp(new_time, t, df['acc_y'].values),
        'acc_z': np.interp(new_time, t, df['acc_z'].values)
    })
    return pd.merge_asof(resampled, df[['time', 'activity']], on='time', direction='nearest')

def prepare_windows_overlapping(df):
    acc_data = df[['acc_x', 'acc_y', 'acc_z']].values
    activities_raw = df['activity'].values
    times = df['time'].values
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
        
        windows.append(win.T / 9.81)
        energies.append(np.std(np.sqrt(np.sum(win**2, axis=1))))
        freqs.append(get_dominant_freq(win.T))
        activities.append(unique[np.argmax(counts)])
        timestamps.append(times[i])
    
    return torch.FloatTensor(np.array(windows)), np.array(energies), np.array(freqs), activities, np.array(timestamps)

def create_ground_truth(activities):
    patterns = ['walk', 'jog', 'run', 'stair', 'climb', 'freewalk', 'gait']
    return np.array([1 if any(p in str(a).lower() for p in patterns) else 0 for a in activities])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load(REPO_NAME, 'eldernet_ft', trust_repo=True).to(device)
    model.eval()
    
    csv_files = sorted(glob.glob(os.path.join(DATA_PATH, '*.csv')))
    if not csv_files: return

    all_subject_summary = []
    all_plot_data = []

    for idx, filepath in enumerate(csv_files):
        subject_id = os.path.basename(filepath).replace('.csv', '')
        print(f"\n[{idx+1}/{len(csv_files)}] Processing: {subject_id}")
        
        subject_results = {'Subject': subject_id, 'Wrists': {}}
        
        for wrist in ['right', 'left']:
            try:
                df = load_weargait_data(filepath, wrist=wrist)
                fs = detect_sampling_rate(df['time'].values)
                df_30hz = resample_to_30hz(df, fs)
                wins, engs, frqs, acts, tmstps = prepare_windows_overlapping(df_30hz)
                
                with torch.no_grad():
                    logits = model(wins.to(device))
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                
                probs_sm = np.convolve(probs, np.ones(3)/3, mode='same')
                y_pred_raw = ((probs_sm > CONF_THRESH) & (engs > ENERGY_THRESH) & (frqs > MIN_FREQ) & (frqs < MAX_FREQ)).astype(int)
                y_pred = median_filter(y_pred_raw, size=3)
                y_true = create_ground_truth(acts)
                
                # Metrics
                if np.sum(y_true) == 0:
                    p, r, f1 = 0.0, 0.0, 0.0
                else:
                    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average='binary', zero_division=0)
                acc = accuracy_score(y_true, y_pred)

                # Store for summary table
                all_subject_summary.append({
                    'Subject': subject_id, 'Wrist': wrist,
                    'Duration_sec': tmstps[-1] if len(tmstps) > 0 else 0,
                    'NumWindows': len(wins), 'Accuracy': acc, 'Precision': p, 'Recall': r, 'F1': f1
                })

                # Store sequences for dual plotting
                subject_results['Wrists'][wrist] = {
                    'ConfSeq': probs_sm, 'EnergiesSeq': engs, 'FreqsSeq': frqs, 'CodesSeq': acts
                }
            except Exception as e:
                print(f"  ✗ {wrist.capitalize()} wrist failed: {str(e)[:50]}")

        if subject_results['Wrists']:
            all_plot_data.append(subject_results)

    # --- SUMMARY DISPLAY ---
    if all_subject_summary:
        print("\n" + "="*80 + "\nOVERALL SUMMARY\n" + "="*80)
        results_df = pd.DataFrame(all_subject_summary)
        
        print(f"\nProcessed {len(results_df)} wrist-data entries successfully")
        print(f"\nMean Performance:")
        print(results_df[['Accuracy', 'Precision', 'Recall', 'F1']].mean())
        print(f"\n{results_df.to_string()}")
        
        results_df.to_csv('weargait_eldernet_dual_results.csv', index=False)
        with open('weargait_detailed_dual_results.pkl', 'wb') as f:
            pickle.dump(all_plot_data, f)
            
        plot_weargait_results_dual(all_plot_data)

def plot_weargait_results_dual(results_list):
    ACTIVITY_MAP = {'Chair': 'Chair', 'Stairs': 'Stairs', 'Standing': 'Standing', 'Walk': 'Walking'}
    GAIT_ACTIVITIES = {'Walk', 'Stairs'}
    PLOT_DIR = 'ElderNet_DualWrist_Plots'
    os.makedirs(PLOT_DIR, exist_ok=True)

    for subj in results_list:
        subj_id = subj['Subject']
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        colors = {'right': 'crimson', 'left': 'steelblue'}
        
        # Determine metadata from available wrist
        ref_side = list(subj['Wrists'].keys())[0]
        ref_data = subj['Wrists'][ref_side]

        plots_info = [
            ('Confidence', 'ConfSeq', 'Prob', axes[0], CONF_THRESH),
            ('Energy', 'EnergiesSeq', 'Std Dev', axes[1], ENERGY_THRESH),
            ('Frequency', 'FreqsSeq', 'Hz', axes[2], MIN_FREQ)
        ]

        for name, key, ylabel, ax, thresh in plots_info:
            for side, data in subj['Wrists'].items():
                ax.plot(data[key], color=colors[side], label=f'{side.capitalize()} Wrist', alpha=0.7, linewidth=1.5)
            
            ax.axhline(thresh, color='gold', linestyle='--', alpha=0.7, label='Threshold')
            if name == 'Frequency':
                ax.axhline(MAX_FREQ, color='gold', linestyle=':', alpha=0.7)
            
            # Gold Shading (using combined mask)
            masks = []
            for side, data in subj['Wrists'].items():
                m = (data['ConfSeq'] > CONF_THRESH) & (data['EnergiesSeq'] > ENERGY_THRESH) & \
                    (data['FreqsSeq'] > MIN_FREQ) & (data['FreqsSeq'] < MAX_FREQ)
                masks.append(m)
            combined_mask = np.any(masks, axis=0) if masks else []
            
            for i, is_gait in enumerate(combined_mask):
                if is_gait: ax.axvspan(i, i+1, color='gold', alpha=0.05)

            # Activity Annotations
            last = None
            for seq_idx, code in enumerate(ref_data['CodesSeq']):
                if seq_idx == 0 or code != last:
                    line_color = 'green' if any(p in str(code).lower() for p in ['walk', 'stair', 'gait']) else 'red'
                    ax.axvline(seq_idx, color=line_color, linestyle='--', alpha=0.3)
                    ax.text(seq_idx + 0.5, ax.get_ylim()[1]*0.8, ACTIVITY_MAP.get(code, code), 
                            rotation=90, fontsize=8, color=line_color, fontweight='bold')
                last = code
            
            ax.set_ylabel(ylabel)
            ax.set_title(f"{name}")
            ax.legend(loc='upper right', fontsize=8)

        plt.suptitle(f"Subject: {subj_id} - Dual Wrist Comparison", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(PLOT_DIR, f'{subj_id}_dual_plot.png'))
        plt.close()
    print(f"✅ Dual wrist plots saved to {PLOT_DIR}")

if __name__ == "__main__":
    main()