from own_signal_processing import load_wisdm_6axis, resample_to_100hz, WristGaitDetector
import numpy as np
import os

# Choose a single accel file
path = os.path.join('wisdm-dataset','raw','watch','accel','data_1601_accel_watch.txt')
print('Testing file:', path)

df_raw = load_wisdm_6axis(path)
print('Raw rows:', len(df_raw))
if df_raw.empty:
    print('No raw data loaded')
    exit()

# Resample
df100 = resample_to_100hz(df_raw)
print('Resampled rows:', len(df100))
if not df_raw.empty:
    t0 = df_raw['timestamp'].iloc[0]
    t1 = df_raw['timestamp'].iloc[-1]
    print('Raw time span (s):', t1 - t0)
    print('Raw time span (min):', (t1 - t0)/60.0)
if df100.empty:
    print('Resample returned empty')
    exit()

# Detect
det = WristGaitDetector()
# For quick testing, limit to first 300 seconds
max_seconds = 300
max_samples = int(max_seconds * 100)
df100_short = df100.iloc[:max_samples].reset_index(drop=True)
print('Using short segment rows:', len(df100_short))
preds, ts, confs = det.detect(df100_short)
print('Windows:', len(preds))
print('Predicted gait windows:', np.sum(preds))

# Map to sample-level (for the short segment)
y_pred = np.zeros(len(df100_short), dtype=int)
for i,p in enumerate(preds):
    s = i*det.step_size
    e = min(s+det.window_size, len(y_pred))
    if p==1:
        y_pred[s:e]=1

# Ground truth
GAIT_ACTIVITIES = {'A','C'}
y_true = df100_short['activity'].isin(GAIT_ACTIVITIES).astype(int).values
print('True gait samples:', np.sum(y_true))

# Simple metrics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
if np.sum(y_true)==0:
    print('No gait in ground truth')
else:
    p,r,f1,_ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average='binary', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    print(f'Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}, Acc: {acc:.3f}')
