import pandas as pd
import numpy as np
import actipy
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.fft import rfft, rfftfreq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---
FILE_PATH = r"C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\wisdm-dataset\raw\watch\accel\data_1600_accel_watch.txt"
FS = 20 

ACTIVITY_MAP = {
    'A':'Walk', 'B':'Jog', 'C':'Stairs', 'D':'Sit', 'E':'Stand', 'F':'Type', 
    'G':'Teeth', 'H':'Soup', 'I':'Chips', 'J':'Pasta', 'K':'Drink', 
    'L':'Sandwich', 'M':'Kicking', 'O':'Catch', 'P':'Dribbling', 
    'Q':'Writing', 'R':'Clapping', 'S':'Folding'
}
GAIT_LABELS = ['Walk', 'Stairs']

# --- 2. LOAD & CLEAN ---
df = pd.read_csv(FILE_PATH, header=None, engine='python', skipinitialspace=True,
                 names=["subject_id", "activity_code", "timestamp", "x", "y", "z"])
df['z'] = df['z'].astype(str).str.rstrip(';').astype(float)
for axis in ['x', 'y', 'z']:
    df[axis] = df[axis] / 9.81
df['time'] = pd.to_datetime(df['timestamp'], unit='ns')
df = df.set_index('time').sort_index()

# --- 3. SIGNAL PROCESSING ---
calib_args = {'calib_min_samples': 10, 'stdtol': 0.05}
data_processed, _ = actipy.process(
    df[['x', 'y', 'z']], sample_rate=FS, lowpass_hz=3.5, 
    calibrate_gravity=True, calibrate_gravity_kwargs=calib_args,
    resample_hz=FS
)

# --- 4. ADVANCED FEATURE EXTRACTION ---
print("Extracting Machine Learning features...")
window_len = int(5 * FS)
step_len = int(2.5 * FS)
vm = np.sqrt(data_processed['x']**2 + data_processed['y']**2 + data_processed['z']**2)
enmo_raw = np.maximum(0, vm - 1.0)

def get_freq_features(signal, fs):
    yf = np.abs(rfft(signal - np.mean(signal)))
    xf = rfftfreq(len(signal), 1/fs)
    mask = (xf >= 0.5) & (xf <= 3.5)
    if not any(mask): return 0, 0
    
    # Dom Freq
    peak_idx = np.argmax(yf[mask])
    dom_freq = xf[mask][peak_idx]
    
    # Purity (Peak Power / Total Power)
    purity = yf[mask][peak_idx] / np.sum(yf[mask])
    return dom_freq, purity

features_list = []
for i in range(0, len(vm) - window_len, step_len):
    win = vm.iloc[i : i + window_len]
    enmo_win = enmo_raw.iloc[i : i + window_len]
    freq, purity = get_freq_features(win.values, FS)
    
    features_list.append({
        'time': win.index[0],
        'enmo': enmo_win.mean(),
        'std_enmo': enmo_win.std(),
        'cadence_hz': freq,
        'purity': purity,
        'active_range': np.ptp(win.values) # Peak-to-peak amplitude
    })

features = pd.DataFrame(features_list).set_index('time')
activity_codes = df['activity_code'].resample('2.5s').apply(lambda x: x.mode()[0] if not x.empty else np.nan)
features['activity'] = activity_codes.reindex(features.index, method='nearest').map(ACTIVITY_MAP)
features['is_walking_actual'] = features['activity'].isin(GAIT_LABELS).astype(int)
features = features.dropna()

# --- 5. RANDOM FOREST CLASSIFICATION ---
print("Training Random Forest Classifier...")
X = features[['enmo', 'std_enmo', 'cadence_hz', 'purity', 'active_range']]
y = features['is_walking_actual']

# Using a split to validate performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Predict on full dataset
features['pred_ml'] = clf.predict(X)
# Smooth with a larger kernel to enforce behavioral continuity
features['pred_smooth'] = medfilt(features['pred_ml'], kernel_size=7)

# --- 6. RESULTS ---
print(f"\n--- ML CLASSIFICATION REPORT ---")
print(classification_report(features['is_walking_actual'], features['pred_smooth'], 
                            target_names=['Stationary/Other', 'Gait']))

# Feature Importance (Great for your internship presentation!)
importances = pd.Series(clf.feature_importances_, index=X.columns)
print("\nFeature Importance (Contribution to decision):")
print(importances.sort_values(ascending=False))

# --- 7. VISUALIZATION ---
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(features.index, features['enmo'], color='black', alpha=0.2, label='ENMO')
ax.fill_between(features.index, 0, 1, where=features['is_walking_actual']==1, color='green', alpha=0.1, label='Actual Gait')
ax.step(features.index, features['pred_smooth'], where='post', color='blue', linewidth=2, label='ML Prediction')

plt.title('Improved ML Gait Detection - Subject 1600')
plt.legend()
plt.tight_layout()
plt.show()