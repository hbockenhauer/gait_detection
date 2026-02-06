import os
import glob
import pandas as pd
import numpy as np
import actipy
from scipy.signal import medfilt
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# --- 1. SETTINGS ---
DATA_DIR = r"C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\wisdm-dataset\raw\watch\accel"
FS = 20
GAIT_LABELS = ['Walk', 'Jog', 'Stairs']
ACTIVITY_MAP = {'A':'Walk', 'B':'Jog', 'C':'Stairs', 'D':'Sit', 'E':'Stand', 'F':'Type', 
                'G':'Teeth', 'H':'Soup', 'I':'Chips', 'J':'Pasta', 'K':'Drink', 
                'L':'Sandwich', 'M':'Kicking', 'O':'Catch', 'P':'Dribbling', 
                'Q':'Writing', 'R':'Clapping', 'S':'Folding'}

def extract_subject_features(file_path):
    print(f"Processing: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, header=None, engine='python', skipinitialspace=True,
                     names=["subject_id", "activity_code", "timestamp", "x", "y", "z"])
    
    # 1. Cleaning
    df['z'] = df['z'].astype(str).str.rstrip(';').astype(float)
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']] / 9.81
    df['time'] = pd.to_datetime(df['timestamp'], unit='ns')
    df = df.set_index('time').sort_index()

    # 2. ActiPy Processing (Calibration OFF for WISDM to avoid warnings/errors)
    data_proc, _ = actipy.process(df[['x', 'y', 'z']], sample_rate=FS, 
                                  lowpass_hz=3.5, calibrate_gravity=False, resample_hz=FS)
    
    # 3. Resample Activity Codes to match data_proc frequency
    # We take the most frequent activity in each 2.5s window
    step = '2.5s'
    window_seconds = 5
    win_len = int(window_seconds * FS)
    
    vm = np.sqrt(data_proc['x']**2 + data_proc['y']**2 + data_proc['z']**2)
    enmo_raw = np.maximum(0, vm - 1.0)
    
    # Extract windowed features safely
    feats = []
    # Using a range that stops safely before the end of the array
    for i in range(0, len(vm) - win_len, int(2.5 * FS)):
        win = vm.iloc[i : i + win_len]
        
        # FFT
        yf = np.abs(rfft(win.values - np.mean(win.values)))
        xf = rfftfreq(win_len, 1/FS)
        mask = (xf >= 0.5) & (xf <= 3.5)
        
        # Find peak
        peak_idx = np.argmax(yf[mask])
        
        # Get label from original df using the timestamp of the window center
        center_time = win.index[len(win)//2]
        # Find nearest label in time
        label_idx = df.index.get_indexer([center_time], method='nearest')[0]
        label_code = df['activity_code'].iloc[label_idx]

        feats.append({
            'enmo': enmo_raw.iloc[i : i + win_len].mean(),
            'cadence_hz': xf[mask][peak_idx],
            'purity': yf[mask][peak_idx] / (np.sum(yf[mask]) + 1e-6),
            'activity_name': ACTIVITY_MAP.get(label_code, 'Unknown')
        })
        
    return pd.DataFrame(feats)

# --- 2. MULTI-SUBJECT LOADING ---
all_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))

# Let's train on subjects 0-14 and test on 15
train_files = all_files[0:15]
test_file = all_files[15]

train_list = []
for f in train_files:
    try:
        train_list.append(extract_subject_features(f))
    except Exception as e:
        print(f"Skipping {f} due to error: {e}")

train_data = pd.concat(train_list, ignore_index=True)
train_data['is_walking'] = train_data['activity_name'].isin(GAIT_LABELS).astype(int)

# --- 3. TEST ON UNSEEN SUBJECT ---
test_data = extract_subject_features(test_file)
test_data['is_walking'] = test_data['activity_name'].isin(GAIT_LABELS).astype(int)

# --- 4. TRAIN ML MODEL ---
X_cols = ['enmo', 'cadence_hz', 'purity']
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(train_data[X_cols], train_data['is_walking'])

# Predict
test_data['pred'] = clf.predict(test_data[X_cols])
test_data['pred_smooth'] = medfilt(test_data['pred'], kernel_size=5)

print("\n" + "="*40)
print(f"TEST RESULTS (Unseen Subject: {os.path.basename(test_file)})")
print("="*40)
print(classification_report(test_data['is_walking'], test_data['pred_smooth']))