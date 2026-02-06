import os
import glob
import pandas as pd
import numpy as np
import actipy
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. SETTINGS ---
DATA_DIR = r"C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\wisdm-dataset\raw\watch\accel"
FS = 20
GAIT_LABELS = ['Walk', 'Stairs']
ACTIVITY_MAP = {'A':'Walk', 'B':'Jog', 'C':'Stairs', 'D':'Sit', 'E':'Stand', 'F':'Type', 
                'G':'Teeth', 'H':'Soup', 'I':'Chips', 'J':'Pasta', 'K':'Drink', 
                'L':'Sandwich', 'M':'Kicking', 'O':'Catch', 'P':'Dribbling', 
                'Q':'Writing', 'R':'Clapping', 'S':'Folding'}

def extract_subject_features(file_path):
    print(f"Processing: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path, header=None, engine='python', skipinitialspace=True,
                     names=["subject_id", "activity_code", "timestamp", "x", "y", "z"])
    
    df['z'] = df['z'].astype(str).str.rstrip(';').astype(float)
    df[['x', 'y', 'z']] = df[['x', 'y', 'z']] / 9.81
    df['time'] = pd.to_datetime(df['timestamp'], unit='ns')
    df = df.set_index('time').sort_index()

    # Process signal (Calibration off for speed in WISDM batch)
    data_proc, _ = actipy.process(df[['x', 'y', 'z']], sample_rate=FS, 
                                  lowpass_hz=3.5, calibrate_gravity=False, resample_hz=FS)
    
    window_seconds = 5
    win_len = int(window_seconds * FS)
    step_len = int(2.5 * FS)
    
    vm = np.sqrt(data_proc['x']**2 + data_proc['y']**2 + data_proc['z']**2)
    enmo_raw = np.maximum(0, vm - 1.0)
    
    feats = []
    for i in range(0, len(vm) - win_len, step_len):
        # Time-domain windows
        win_vm = vm.iloc[i : i + win_len]
        win_x = data_proc['x'].iloc[i : i + win_len]
        win_z = data_proc['z'].iloc[i : i + win_len]
        
        # 1. Frequency Features (FFT)
        yf = np.abs(rfft(win_vm.values - np.mean(win_vm.values)))
        xf = rfftfreq(win_len, 1/FS)
        mask = (xf >= 0.5) & (xf <= 3.5)
        peak_idx = np.argmax(yf[mask])
        
        # 2. Spatial Features (The "Anti-Folding" features)
        correlation = win_x.corr(win_z)
        if np.isnan(correlation): correlation = 0
        
        # 3. Label matching
        center_time = win_vm.index[len(win_vm)//2]
        label_idx = df.index.get_indexer([center_time], method='nearest')[0]
        label_code = df['activity_code'].iloc[label_idx]

        feats.append({
            'enmo': enmo_raw.iloc[i : i + win_len].mean(),
            'cadence_hz': xf[mask][peak_idx],
            'purity': yf[mask][peak_idx] / (np.sum(yf[mask]) + 1e-6),
            'std_z': win_z.std(),
            'corr_xz': correlation,
            'activity_name': ACTIVITY_MAP.get(label_code, 'Unknown')
        })
        
    return pd.DataFrame(feats)

# --- 2. DATA PREP ---
all_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
train_files = all_files[0:15]
test_file = all_files[15]

train_data = pd.concat([extract_subject_features(f) for f in train_files], ignore_index=True)
train_data['is_walking'] = train_data['activity_name'].isin(GAIT_LABELS).astype(int)

test_data = extract_subject_features(test_file)
test_data['is_walking'] = test_data['activity_name'].isin(GAIT_LABELS).astype(int)

# --- 3. TRAIN ML MODEL ---
# Expanded X_cols with our new spatial insights
X_cols = ['enmo', 'cadence_hz', 'purity', 'std_z', 'corr_xz']

clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
clf.fit(train_data[X_cols], train_data['is_walking'])

# --- 4. EVALUATION ---
test_data['pred_smooth'] = medfilt(clf.predict(test_data[X_cols]), kernel_size=5)

print("\n" + "="*40)
print(f"SPATIAL-AWARE RESULTS (Subject: {os.path.basename(test_file)})")
print("="*40)
print(classification_report(test_data['is_walking'], test_data['pred_smooth']))

importances = pd.Series(clf.feature_importances_, index=X_cols).sort_values(ascending=False)
print("\nFeature Importance:\n", importances)

# Check the new Trickster List
fps = test_data[(test_data['is_walking'] == 0) & (test_data['pred_smooth'] == 1)]
print("\nNew False Positive Breakdown:\n", fps['activity_name'].value_counts())

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_gait_diagnostic_timeline(test_data):
    # 1. Setup Figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True, 
                                        gridspec_kw={'height_ratios': [1, 2, 1]})
    
    # --- AXIS 1: Ground Truth Activities ---
    # Create a background color for every unique activity
    unique_acts = test_data['activity_name'].unique()
    colors = plt.cm.get_cmap('tab20', len(unique_acts))
    act_to_color = {act: colors(i) for i, act in enumerate(unique_acts)}
    
    for act in unique_acts:
        mask = test_data['activity_name'] == act
        ax1.fill_between(test_data.index, 0, 1, where=mask, color=act_to_color[act], 
                         alpha=0.4, label=act)
    
    ax1.set_yticks([])
    ax1.set_title("Subject Activity Context (Ground Truth Labels)")
    ax1.legend(loc='upper right', bbox_to_anchor=(1.12, 1), title="Activities", fontsize='small')

    # --- AXIS 2: Relevant Features (The 'Physics') ---
    ax2.plot(test_data.index, test_data['enmo'], color='black', alpha=0.6, label='Intensity (ENMO)')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(test_data.index, test_data['corr_xz'], color='blue', alpha=0.4, label='Spatial Coupling (corr_xz)')
    
    ax2.set_ylabel("ENMO (g)")
    ax2_twin.set_ylabel("Correlation (X-Z)")
    ax2.set_title("Key Features used by Random Forest")
    
    # Combine legends from twins
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # --- AXIS 3: RF Classifier Output ---
    # Actual Walking (Target)
    ax3.fill_between(test_data.index, 0, 1, where=test_data['is_walking']==1, 
                     color='green', alpha=0.15, label='Actual Gait')
    
    # Model Prediction (Result)
    ax3.step(test_data.index, test_data['pred_smooth'], where='post', 
             color='red', linewidth=1.5, label='Detected Gait (RF)')
    
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Idle', 'GAIT'])
    ax3.set_ylabel("Detection State")
    ax3.set_title("Classifier Outcome")
    ax3.legend(loc='upper left')

    plt.xlabel("Timeline")
    plt.tight_layout()
    plt.show()

# Run the plotting function
plot_gait_diagnostic_timeline(test_data)