import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gaitmap.gait_detection import UllrichGaitSequenceDetection
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from gaitmap.gait_detection import UllrichGaitSequenceDetection
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# --- 1. SETTINGS & MAPPING ---
ACTIVITY_MAP = {
    'A': 'Walking', 'B': 'Jogging', 'C': 'Stairs', 'D': 'Sitting',
    'E': 'Standing', 'F': 'Typing', 'G': 'Brushing Teeth', 'H': 'Eating (Soup)',
    'I': 'Eating (Chips)', 'J': 'Eating (Pasta)', 'K': 'Drinking',
    'L': 'Eating (Sandwich)', 'M': 'Kicking', 'O': 'Catching', 'P': 'Dribbling', 
    'Q': 'Writing', 'R': 'Clapping', 'S': 'Folding Clothes'
}

# Optimized Ground Truth: Include Walking, Jogging, and Stairs
WALKING_ACTIVITIES = {'A', 'B', 'C'} 

# --- 2. SIGNAL FILTERING ---
def lowpass_filter(data, cutoff=3.0, fs=100.0, order=4):
    """
    Applies a Butterworth lowpass filter to remove hand jitter.
    3Hz is usually enough to capture the gait harmonics while removing noise.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply to each column except timestamp and activity
    filtered_df = data.copy()
    cols_to_filter = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
    for col in cols_to_filter:
        filtered_df[col] = filtfilt(b, a, data[col])
    return filtered_df

def run_ullrich_detection(data_df, sampling_rate_hz=100.0):
    # Apply lowpass filter before detection to clean the PSD
    clean_df = lowpass_filter(data_df, cutoff=3.5, fs=sampling_rate_hz)
    
    detector = UllrichGaitSequenceDetection(
        sensor_channel_config="acc", 
        peak_prominence=1.0, 
        active_signal_threshold=0.15, # Slightly raised to ignore fidgeting
        locomotion_band=(0.5, 3.0)
    )
    
    mapped_df = clean_df.rename(columns={
        'acc_x': 'acc_pa', 'acc_y': 'acc_ml', 'acc_z': 'acc_si',
        'gyr_x': 'gyr_pa', 'gyr_y': 'gyr_ml', 'gyr_z': 'gyr_si'
    })
    
    detector = detector.detect(mapped_df, sampling_rate_hz=sampling_rate_hz)
    gs = detector.gait_sequences_
    
    # --- POST-PROCESSING: Minimum Duration Filter ---
    # Real walking bouts are rarely shorter than 3 seconds.
    if not gs.empty:
        gs['duration'] = (gs['end'] - gs['start']) / sampling_rate_hz
        gs = gs[gs['duration'] >= 3.0].copy()
        
    return gs
    return detector.gait_sequences_

# --- 2. DATA LOADING (FIXED COLUMN HANDLING) ---
def load_wisdm_full_6axis(acc_path):
    gyro_path = acc_path.replace("accel", "gyro")
    
    def read_file(p):
        rows = []
        if not os.path.exists(p): return pd.DataFrame()
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip().rstrip(';')
                if not line: continue
                parts = line.split(',')
                if len(parts) < 6: continue
                try:
                    # subject, activity, timestamp, x, y, z
                    rows.append((parts[1], float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])))
                except: continue
        return pd.DataFrame(rows, columns=["activity", "timestamp", "x", "y", "z"])

    df_acc = read_file(acc_path).rename(columns={'x':'acc_x', 'y':'acc_y', 'z':'acc_z'})
    df_gyr = read_file(gyro_path).rename(columns={'x':'gyr_x', 'y':'gyr_y', 'z':'gyr_z'})

    if df_acc.empty: return pd.DataFrame()
    
    df_acc = df_acc.sort_values("timestamp")
    if not df_gyr.empty:
        df_gyr = df_gyr.sort_values("timestamp")
        # We drop 'activity' from gyro before merging so we only have one 'activity' column
        df_gyr = df_gyr.drop(columns=['activity'])
        df = pd.merge_asof(df_acc, df_gyr, on="timestamp", direction="nearest")
    else:
        df = df_acc
        for c in ['gyr_x', 'gyr_y', 'gyr_z']: df[c] = 0.0

    df["timestamp"] = (df["timestamp"] - df["timestamp"].min()) / 1e9
    return df

# --- 3. RESAMPLING (FIXED CATEGORICAL HANDLING) ---
def resample_to_100hz(df):
    t = df["timestamp"].to_numpy()
    if len(t) < 2: return pd.DataFrame()
    
    duration = t[-1] - t[0]
    num_samples = int(np.round(duration * 100.0)) + 1
    new_t = np.linspace(t[0], t[-1], num_samples)
    
    resampled_data = {"timestamp": new_t}
    for col in ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']:
        resampled_data[col] = np.interp(new_t, t, df[col].to_numpy())
    
    resampled_df = pd.DataFrame(resampled_data)
    
    # Efficiently map activity labels to the new time grid
    # We use merge_asof to find the closest activity for every new timestamp
    resampled_df = pd.merge_asof(
        resampled_df, 
        df[['timestamp', 'activity']], 
        on='timestamp', 
        direction='nearest'
    )
    
    return resampled_df

# --- 4. VISUALIZATION ---
def save_full_activity_plot(df, gs, subj_id, output_folder):
    plt.figure(figsize=(20, 8))
    acc_norm = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    time_axis = df['timestamp']
    
    plt.plot(time_axis, acc_norm, color='black', alpha=0.2, linewidth=0.5, label='Signal Norm')
    
    # Draw Background Activities
    change_idx = np.where(df['activity'] != df['activity'].shift())[0]
    change_idx = np.append(change_idx, len(df)-1)
    
    for i in range(len(change_idx)-1):
        start_idx = change_idx[i]
        end_idx = change_idx[i+1]
        act_code = df['activity'].iloc[start_idx]
        label_text = ACTIVITY_MAP.get(act_code, act_code)
        
        # Cycle through colors
        color = plt.cm.get_cmap('tab20')(i % 20)
        plt.axvspan(time_axis.iloc[start_idx], time_axis.iloc[end_idx], alpha=0.15, color=color)
        
        # Label each segment
        plt.text((time_axis.iloc[start_idx] + time_axis.iloc[end_idx])/2, 
                 plt.gca().get_ylim()[1]*0.85, label_text, 
                 horizontalalignment='center', rotation=45, fontsize=7, fontweight='bold')

    # Plot Detection Results
    if not gs.empty:
        for i, row in gs.iterrows():
            plt.hlines(y=-1, xmin=row['start']/100, xmax=row['end']/100, 
                       color='green', linewidth=6, label='Gait Detected' if i == 0 else "")

    plt.title(f"Subject {subj_id}: Activity Timeline vs. Ullrich Gait Detection", fontsize=14)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Acceleration Norm (m/s²)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"subject_{subj_id}_full.png"), dpi=200)
    plt.close()

# --- 5. PERFORMANCE METRICS ---
def get_ground_truth_labels(df_resampled):
    """
    Create binary ground truth: 1 if activity is a walking activity, 0 otherwise.
    Returns a numpy array of labels (same length as df_resampled).
    """
    return (df_resampled['activity'].isin(WALKING_ACTIVITIES)).astype(int).to_numpy()

def get_predicted_labels(df_resampled, gs):
    """
    Create binary predictions: 1 if sample is within a detected gait segment, 0 otherwise.
    gs: gait sequences dataframe with 'start' and 'end' indices (in 100Hz samples).
    """
    predictions = np.zeros(len(df_resampled), dtype=int)
    
    if not gs.empty:
        for _, row in gs.iterrows():
            start_idx = int(row['start'])
            end_idx = int(row['end'])
            predictions[start_idx:end_idx+1] = 1
    
    return predictions

def compute_metrics(y_true, y_pred):
    """
    Compute precision, recall, accuracy, F1 score, and confusion matrix.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'sensitivity': recall_score(y_true, y_pred, zero_division=0),  # Alias for recall
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
    }
    return metrics

def print_metrics(subject_id, metrics):
    """
    Pretty print performance metrics for a subject.
    """
    print(f"\n  ┌─ Subject {subject_id} Performance Metrics ─────────────────")
    print(f"  │ Precision:  {metrics['precision']:.4f}  (True Positives / All Positives)")
    print(f"  │ Recall:     {metrics['recall']:.4f}  (True Positives / All Walking)")
    print(f"  │ Accuracy:   {metrics['accuracy']:.4f}  (Correct / Total)")
    print(f"  │ F1 Score:   {metrics['f1']:.4f}  (Harmonic mean of P & R)")
    print(f"  │ Specificity:{metrics['specificity']:.4f}  (True Negatives / All Non-Walking)")
    print(f"  │ TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}")
    print(f"  └" + "─" * 50)

# --- 5. MAIN ---
def main():
    BASE_PATH = r"C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\wisdm-dataset\raw\watch"
    ACC_FOLDER = os.path.join(BASE_PATH, "accel")
    PLOT_FOLDER = "full_activity_plots"
    if not os.path.exists(PLOT_FOLDER): os.makedirs(PLOT_FOLDER)

    files = sorted(glob.glob(os.path.join(ACC_FOLDER, "*.txt")))
    
    all_metrics = []
    all_y_true = []
    all_y_pred = []
    
    print("\n" + "="*60)
    print("GAIT DETECTION WITH PERFORMANCE EVALUATION")
    print("="*60)
    
    for file_path in files:
        subj_id = os.path.basename(file_path).split('_')[1]
        print(f"\nProcessing Subject {subj_id}...", end=" ")
        
        try:
            df_raw = load_wisdm_full_6axis(file_path)
            if df_raw.empty:
                print("(No data)")
                continue
            
            df_100hz = resample_to_100hz(df_raw)
            gs = run_ullrich_detection(df_100hz)
            
            # Get ground truth and predictions
            y_true = get_ground_truth_labels(df_100hz)
            y_pred = get_predicted_labels(df_100hz, gs)
            
            # Compute metrics
            metrics = compute_metrics(y_true, y_pred)
            metrics['subject'] = subj_id
            metrics['gait_bouts'] = len(gs)
            
            all_metrics.append(metrics)
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            
            # Print individual metrics
            print_metrics(subj_id, metrics)
            
            save_full_activity_plot(df_100hz, gs, subj_id, PLOT_FOLDER)
            
        except Exception as e:
            print(f"(Error: {e})")

    # Compute and print overall metrics
    if all_y_true:
        overall_metrics = compute_metrics(np.array(all_y_true), np.array(all_y_pred))
        
        print("\n" + "="*60)
        print("OVERALL PERFORMANCE (All Subjects Combined)")
        print("="*60)
        print(f"Precision:  {overall_metrics['precision']:.4f}")
        print(f"Recall:     {overall_metrics['recall']:.4f}")
        print(f"Accuracy:   {overall_metrics['accuracy']:.4f}")
        print(f"F1 Score:   {overall_metrics['f1']:.4f}")
        print(f"Specificity:{overall_metrics['specificity']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TP: {overall_metrics['tp']}, FP: {overall_metrics['fp']}")
        print(f"  TN: {overall_metrics['tn']}, FN: {overall_metrics['fn']}")
        print("="*60 + "\n")
    
    # Save metrics to CSV
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv("gait_detection_metrics.csv", index=False)
        print(f"\n✓ Metrics saved to 'gait_detection_metrics.csv'")
        print(f"✓ Plots saved to '{PLOT_FOLDER}/'")
        print(f"\nProcessed {len(all_metrics)} subjects total.")


if __name__ == "__main__":
    main()