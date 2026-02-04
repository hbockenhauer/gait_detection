import torch
import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import median_filter
import os
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
for filename in files:
    try:
        subject_id = filename.split('_')[1]
        df = load_wisdm_cleaned(os.path.join(FOLDER_PATH, filename))
        windows, energies, freqs, y_true, codes = prepare_batch_windows(df)
        
        with torch.no_grad():
            probs = torch.softmax(model(windows.to(device)), dim=1)[:, 1].cpu().numpy()

        # Logic Gates
        CONF_THRESH = 0.45
        ENERGY_THRESH = 0.07
        min_freq = 0.5
        max_freq = 3.0
        probs_smoothed = np.convolve(probs, np.ones(3)/3, mode='same')
        y_pred_raw = ((probs_smoothed > CONF_THRESH) & (energies > ENERGY_THRESH) & (freqs > min_freq) & (freqs < max_freq)).astype(int)
        y_pred = median_filter(y_pred_raw, size=3)

        # Break down by activity
        activity_stats = []
        for code in np.unique(codes):
            idx = [i for i, c in enumerate(codes) if c == code]
            # Precision/Recall for THIS specific activity
            p_a, r_a, f_a, _ = precision_recall_fscore_support(np.array(y_true)[idx], y_pred[idx], labels=[1], average='binary', zero_division=0)
            activity_stats.append({'Activity': ACTIVITY_MAP.get(code, code), 'Precision': p_a, 'Recall': r_a, 'F1': f_a})

        # Subject-wide Metrics
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average='binary', zero_division=0)
        results_list.append({
            'Subject': subject_id, 
            'Accuracy': accuracy_score(y_true, y_pred), 
            'Precision': p, 'Recall': r, 'F1': f1,
            'ActivityDetails': activity_stats
        })
        print(f"✅ ID {subject_id}: Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}")

    except Exception as e:
        print(f"❌ Error {filename}: {e}")

# --- 4. GLOBAL SUMMARY ---
results_df = pd.DataFrame(results_list)
print("\n" + "="*50)
print("OVERALL METRICS (MEAN ACROSS SUBJECTS)")
print(results_df[['Accuracy', 'Precision', 'Recall', 'F1']].mean())

# --- 5. ACTIVITY BREAKDOWN ---
all_act_data = []
for row in results_list:
    for item in row['ActivityDetails']:
        all_act_data.append(item)

summary_act = pd.DataFrame(all_act_data).groupby('Activity').mean()
print("\n" + "="*50)
print("PER-ACTIVITY PERFORMANCE")
print(summary_act)