import os
import glob
import numpy as np
import pandas as pd
import torch
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
    """Parse time value from various formats"""
    if pd.isna(t):
        return np.nan
    
    if isinstance(t, (int, float)):
        return float(t)
    
    if isinstance(t, str):
        # Remove 'sec' and any whitespace
        t_clean = t.replace('sec', '').strip()
        try:
            return float(t_clean)
        except ValueError:
            return np.nan
    
    return np.nan

def inspect_csv_structure(filepath, max_rows=5):
    """Inspect CSV structure to determine separator and columns"""
    for sep in ['\t', ',', ';', ' ']:
        try:
            df = pd.read_csv(filepath, sep=sep, nrows=max_rows)
            if len(df.columns) > 10:
                return sep, df.columns.tolist()
        except:
            continue
    return None, None

def load_weargait_data(filepath, wrist='right'):
    """
    Load WearGait-PD CSV with flexible column detection
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    wrist : str
        'right' or 'left'
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with time, acc_x, acc_y, acc_z, activity
    """
    # Detect separator
    sep, columns = inspect_csv_structure(filepath)
    
    if sep is None:
        raise ValueError(f"Could not parse CSV file: {filepath}")
    
    # Read full file with low_memory=False to avoid dtype warnings
    df = pd.read_csv(filepath, sep=sep, low_memory=False)
    
    print(f"  File has {len(df.columns)} columns, {len(df)} rows")
    
    # Determine wrist column prefix
    wrist_prefix = 'R_Wrist' if wrist == 'right' else 'L_Wrist'
    
    # Find columns
    def find_column(patterns):
        """Find first matching column name"""
        for pattern in patterns:
            for col in df.columns:
                if pattern.lower() in col.lower():
                    return col
        return None
    
    # Find time column
    time_col = find_column(['Time', 'time', 'timestamp'])
    if time_col is None:
        raise ValueError("Could not find time column")
    
    # Find activity column
    activity_col = find_column(['GeneralEvent', 'Event', 'Activity', 'Label'])
    if activity_col is None:
        print("  WARNING: No activity column found, using 'Walking' for all")
        df['Activity'] = 'Walking'
        activity_col = 'Activity'
    
    # Find wrist accelerometer columns
    acc_x_col = find_column([f'{wrist_prefix}_Acc_X', f'{wrist_prefix}_AccX'])
    acc_y_col = find_column([f'{wrist_prefix}_Acc_Y', f'{wrist_prefix}_AccY'])
    acc_z_col = find_column([f'{wrist_prefix}_Acc_Z', f'{wrist_prefix}_AccZ'])
    
    if not all([acc_x_col, acc_y_col, acc_z_col]):
        raise ValueError(f"Could not find all {wrist} wrist accelerometer columns")
    
    print(f"  Found columns: Time={time_col}, Activity={activity_col}")
    print(f"  Acc: X={acc_x_col}, Y={acc_y_col}, Z={acc_z_col}")
    
    # Create simplified dataframe
    data = pd.DataFrame({
        'time_raw': df[time_col],
        'acc_x': pd.to_numeric(df[acc_x_col], errors='coerce'),
        'acc_y': pd.to_numeric(df[acc_y_col], errors='coerce'),
        'acc_z': pd.to_numeric(df[acc_z_col], errors='coerce'),
        'activity': df[activity_col].astype(str)
    })
    
    # Parse time column
    print(f"  Parsing time column (first value: {data['time_raw'].iloc[0]})...")
    data['time'] = data['time_raw'].apply(parse_time_value)
    
    # Remove rows with invalid time or NaN accelerometer data
    data = data.dropna(subset=['time', 'acc_x', 'acc_y', 'acc_z'])
    
    if len(data) == 0:
        raise ValueError("No valid data after removing NaN values")
    
    # Reset time to start from 0
    data['time'] = data['time'] - data['time'].min()
    
    # Drop the raw time column
    data = data.drop(columns=['time_raw'])
    
    print(f"  Valid data: {len(data)} samples, duration: {data['time'].iloc[-1]:.1f} sec")
    
    return data

def detect_sampling_rate(time_series):
    """Detect sampling rate from time series"""
    time_diffs = np.diff(time_series[:1000])  # Use first 1000 samples
    median_diff = np.median(time_diffs)
    fs = 1.0 / median_diff
    return fs

def resample_to_30hz(df, original_fs):
    """Resample data to 30 Hz"""
    target_fs = 30.0
    
    if abs(original_fs - target_fs) < 0.1:
        return df
    
    t = df['time'].values
    duration = t[-1] - t[0]
    n_samples = int(duration * target_fs) + 1
    new_time = np.linspace(t[0], t[-1], n_samples)
    
    resampled = {
        'time': new_time,
        'acc_x': np.interp(new_time, t, df['acc_x'].values),
        'acc_y': np.interp(new_time, t, df['acc_y'].values),
        'acc_z': np.interp(new_time, t, df['acc_z'].values)
    }
    
    resampled_df = pd.DataFrame(resampled)
    
    # Map activities
    resampled_df = pd.merge_asof(
        resampled_df,
        df[['time', 'activity']],
        on='time',
        direction='nearest'
    )
    
    return resampled_df

def prepare_windows_overlapping(df, window_size=300, step_size=30):
    """Create overlapping windows for ElderNet"""
    
    acc_data = df[['acc_x', 'acc_y', 'acc_z']].values
    activities_raw = df['activity'].values
    times = df['time'].values
    
    windows = []
    energies = []
    freqs = []
    activities = []
    timestamps = []
    
    def get_dominant_freq(win, fs=30):
        mag = np.sqrt(np.sum(win**2, axis=0))
        mag = mag - np.mean(mag)
        freqs_fft = np.fft.rfftfreq(len(mag), d=1/fs)
        fft_vals = np.abs(np.fft.rfft(mag))
        return freqs_fft[np.argmax(fft_vals)]
    
    for i in range(0, len(acc_data) - window_size + 1, step_size):
        win = acc_data[i:i + window_size]
        act_win = activities_raw[i:i + window_size]
        time_start = times[i]
        
        unique, counts = np.unique(act_win, return_counts=True)
        majority_activity = unique[np.argmax(counts)]
        
        # Calculate features from original data
        energy = np.std(np.sqrt(np.sum(win**2, axis=1)))
        freq = get_dominant_freq(win.T, fs=30)
        
        # Normalize for model (divide by 9.81)
        win_normalized = win.T / 9.81
        
        windows.append(win_normalized)
        energies.append(energy)
        freqs.append(freq)
        activities.append(majority_activity)
        timestamps.append(time_start)
    
    windows = torch.FloatTensor(np.array(windows))
    energies = np.array(energies)
    freqs = np.array(freqs)
    timestamps = np.array(timestamps)
    
    return windows, energies, freqs, activities, timestamps

def create_ground_truth(activities):
    """Create ground truth labels"""
    # Patterns that indicate gait/walking
    GAIT_PATTERNS = [
        'walk', 'jog', 'run', 'stair', 'climb',
        'freewalk', 'free walk', 'gait'
    ]
    
    y_true = []
    for act in activities:
        act_lower = str(act).lower()
        is_gait = any(pattern in act_lower for pattern in GAIT_PATTERNS)
        y_true.append(1 if is_gait else 0)
    
    return np.array(y_true)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = torch.hub.load(REPO_NAME, 'eldernet_ft', trust_repo=True).to(device)
    model.eval()
    print("ElderNet model loaded successfully\n")
    
    csv_files = sorted(glob.glob(os.path.join(DATA_PATH, '*.csv')))
    
    if len(csv_files) == 0:
        print(f"ERROR: No CSV files found in {DATA_PATH}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    print("="*80)
    
    all_results = []
    
    for idx, filepath in enumerate(csv_files):
        subject_id = os.path.basename(filepath).replace('.csv', '')
        
        print(f"\n[{idx+1}/{len(csv_files)}] Processing: {subject_id}")
        print("-"*80)
        
        try:
            # Try right wrist first, then left
            wrist_used = None
            df = None
            
            for wrist in ['left', 'right']:
                try:
                    df = load_weargait_data(filepath, wrist=wrist)
                    wrist_used = wrist
                    print(f"  ✓ Using {wrist.upper()} wrist sensor")
                    break
                except Exception as e:
                    print(f"  ✗ {wrist.capitalize()} wrist failed: {str(e)[:50]}")
                    continue
            
            if df is None or wrist_used is None:
                raise ValueError("Could not load data from either wrist")
            
            # Detect sampling rate
            fs = detect_sampling_rate(df['time'].values)
            print(f"  Detected sampling rate: {fs:.1f} Hz")
            
            # Resample to 30 Hz
            df_30hz = resample_to_30hz(df, fs)
            print(f"  Resampled to 30 Hz ({len(df_30hz)} samples)")
            
            # Prepare windows
            windows, energies, freqs, activities, timestamps = prepare_windows_overlapping(
                df_30hz, window_size=WINDOW_SIZE, step_size=STEP_SIZE
            )
            
            print(f"  Created {len(windows)} overlapping windows")
            
            # Run ElderNet
            with torch.no_grad():
                logits = model(windows.to(device))
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            # Apply thresholds
            probs_smoothed = np.convolve(probs, np.ones(3)/3, mode='same')
            y_pred_raw = ((probs_smoothed > CONF_THRESH) & 
                         (energies > ENERGY_THRESH) & 
                         (freqs > MIN_FREQ) & 
                         (freqs < MAX_FREQ)).astype(int)
            y_pred = median_filter(y_pred_raw, size=3)
            
            # Create ground truth
            y_true = create_ground_truth(activities)
            
            # Print unique activities
            unique_acts = np.unique(activities)
            print(f"  Activities found: {', '.join(unique_acts[:5])}")
            if len(unique_acts) > 5:
                print(f"                    ... and {len(unique_acts)-5} more")
            
            # Calculate metrics
            gait_windows = np.sum(y_true)
            if gait_windows == 0:
                print("  ⚠️  WARNING: No gait activities in ground truth")
                p, r, f1 = 0.0, 0.0, 0.0
            else:
                p, r, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=[1], average='binary', zero_division=0
                )
            
            acc = accuracy_score(y_true, y_pred)
            
            print(f"\n  📊 Results:")
            print(f"     Accuracy:  {acc:.3f}")
            print(f"     Precision: {p:.3f}")
            print(f"     Recall:    {r:.3f}")
            print(f"     F1-Score:  {f1:.3f}")
            print(f"     Ground truth gait: {gait_windows}/{len(y_true)} ({100*gait_windows/len(y_true):.1f}%)")
            print(f"     Predicted gait:    {np.sum(y_pred)}/{len(y_pred)} ({100*np.sum(y_pred)/len(y_pred):.1f}%)")
            
            # Add these specific keys to your existing append block
            all_results.append({
                'Subject': subject_id,
                'Wrist': wrist_used,
                'NumWindows': len(windows),
                'Duration_sec': timestamps[-1] if len(timestamps) > 0 else 0,
                'Accuracy': acc,
                'Precision': p,
                'Recall': r,
                'F1': f1,
                'Activities': activities,
                'Predictions': y_pred,
                'GroundTruth': y_true,
                'Timestamps': timestamps,
                # --- ADD THESE FOR THE PLOTTER ---
                'ConfSeq': probs_smoothed,
                'EnergiesSeq': energies,
                'FreqsSeq': freqs,
                'CodesSeq': activities
            })
            
        except Exception as e:
            print(f"  ❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    if len(all_results) > 0:
        print("\n" + "="*80)
        print("OVERALL SUMMARY")
        print("="*80)
        
        results_df = pd.DataFrame([{
            'Subject': r['Subject'],
            'Wrist': r['Wrist'],
            'Duration_sec': r['Duration_sec'],
            'NumWindows': r['NumWindows'],
            'Accuracy': r['Accuracy'],
            'Precision': r['Precision'],
            'Recall': r['Recall'],
            'F1': r['F1']
        } for r in all_results])
        
        print(f"\nProcessed {len(results_df)} subjects successfully")
        print(f"\nMean Performance:")
        print(results_df[['Accuracy', 'Precision', 'Recall', 'F1']].mean())
        print(f"\nStd Performance:")
        print(results_df[['Accuracy', 'Precision', 'Recall', 'F1']].std())
        
        print(f"\n{results_df.to_string()}")
        
        results_df.to_csv('weargait_eldernet_results_left_wrist.csv', index=False)
        print(f"\n✅ Results saved to 'weargait_eldernet_results_left_wrist.csv'")
        
        import pickle
        with open('weargait_detailed_results_left_wrist.pkl', 'wb') as f:
            pickle.dump(all_results, f)
        print(f"✅ Detailed results saved to 'weargait_detailed_results_left_wrist.pkl'")

        plot_weargait_results(all_results)
    
    else:
        print("\n❌ No subjects processed successfully")

def plot_weargait_results(results_list):
    import matplotlib.pyplot as plt
    
    ACTIVITY_MAP = {'Chair': 'Chair', 'Stairs': 'Stairs', 'Standing': 'Standing', 'Walk': 'Walking'}
    GAIT_ACTIVITIES = {'Walk', 'Stairs'}
    PLOT_DIR = 'ElderNet_WearGait_Plots'
    os.makedirs(PLOT_DIR, exist_ok=True)

    for selected_subj in results_list:
        subj_id = selected_subj.get('Subject')
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        conf_seq = selected_subj['ConfSeq']
        energy_seq = selected_subj['EnergiesSeq']
        freq_seq = selected_subj['FreqsSeq']
        codes = selected_subj['CodesSeq']

        gait_mask = (conf_seq > CONF_THRESH) & (energy_seq > ENERGY_THRESH) & \
                    (freq_seq > MIN_FREQ) & (freq_seq < MAX_FREQ)
        
        plots_info = [
            ('Confidence', conf_seq, 'Prob', axes[0], CONF_THRESH),
            ('Energy', energy_seq, 'Std Dev', axes[1], ENERGY_THRESH),
            ('Frequency', freq_seq, 'Hz', axes[2], MIN_FREQ)
        ]

        for name, seq, ylabel, ax, thresh in plots_info:
            ax.plot(seq, color='steelblue', linewidth=1.5)
            
            # Draw the standard/minimum threshold line
            ax.axhline(thresh, color='gold', linestyle='--', alpha=0.7, label='Threshold')
            
            # --- NEW: ADD MAX FREQUENCY LINE ---
            if name == 'Frequency':
                ax.axhline(MAX_FREQ, color='gold', linestyle=':', alpha=0.7, label='Max Frequency')
                ax.legend(loc='upper right', fontsize=8)
            # -----------------------------------
            
            # Add Gold Shading where ElderNet predicts gait
            for i, is_gait in enumerate(gait_mask):
                if is_gait:
                    ax.axvspan(i, i+1, color='gold', alpha=0.1)

            # Activity Annotations
            last = None
            for seq_idx, code in enumerate(codes):
                if seq_idx == 0 or code != last:
                    line_color = 'green' if code in GAIT_ACTIVITIES else 'red'
                    ax.axvline(seq_idx, color=line_color, linestyle='--', alpha=0.3)
                    ax.text(seq_idx + 0.5, ax.get_ylim()[1]*0.8, ACTIVITY_MAP.get(code, code), 
                            rotation=90, fontsize=8, color=line_color, fontweight='bold')
                last = code
            ax.set_ylabel(ylabel)
            ax.set_title(f"{name}")

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f'{subj_id}_plot.png'))
        plt.close()
    print(f"✅ Plots saved to {PLOT_DIR}")

if __name__ == "__main__":
    main()