import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from own_signal_processing import load_wisdm_6axis, resample_to_100hz, WristGaitDetector

# Files
BASE = os.path.join('wisdm-dataset','raw','watch','accel')
files = sorted(glob.glob(os.path.join(BASE, '*.txt')))[:5]

def evaluate_combo(params, cached_segments):
    """Evaluate params on precomputed short resampled segments.
    cached_segments: list of DataFrames (short resampled)"""
    f1s = []
    for df100 in cached_segments:
        if df100 is None or df100.empty:
            continue
        detector = WristGaitDetector(
            sampling_rate=100.0,
            window_size_sec=3.0,
            step_size_sec=0.5,
            gait_freq_range=(0.8,2.5),
            min_harmonic_ratio=params['min_harmonic_ratio'],
            min_amplitude=params['min_amplitude'],
            min_autocorr_peak=params['min_autocorr_peak'],
            use_gyro=True,
            required_rules=params['required_rules']
        )
        preds, ts, confs = detector.detect(df100)
        # map to sample-level
        y_pred = np.zeros(len(df100), dtype=int)
        for i,p in enumerate(preds):
            s = i*detector.step_size
            e = min(s+detector.window_size, len(y_pred))
            if p==1:
                y_pred[s:e]=1
        GAIT_ACTIVITIES = {'A','C'}
        y_true = df100['activity'].isin(GAIT_ACTIVITIES).astype(int).values
        if np.sum(y_true)==0:
            continue
        p,r,f1,_ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average='binary', zero_division=0)
        f1s.append(f1)
    if len(f1s)==0:
        return None
    return np.mean(f1s)

# Parameter grid
min_amplitudes = [0.1, 0.3, 0.6, 1.5]
min_harmonic_ratios = [0.1, 0.2, 0.3]
min_autocorr_peaks = [0.2, 0.3, 0.4]
required_rules_list = [2,3,4]

# Precompute short resampled segments to avoid repeated costly resampling
cached_segments = []
for file_path in files:
    df_raw = load_wisdm_6axis(file_path)
    if df_raw.empty:
        cached_segments.append(None)
        continue
    df100 = resample_to_100hz(df_raw)
    if df100.empty:
        cached_segments.append(None)
        continue
    # keep short segment (default 120s)
    df_short = df100.iloc[:int(120*100)].reset_index(drop=True)
    cached_segments.append(df_short)

results = []
for ma in min_amplitudes:
    for mh in min_harmonic_ratios:
        for mauto in min_autocorr_peaks:
            for rr in required_rules_list:
                params = {'min_amplitude': ma,
                          'min_harmonic_ratio': mh,
                          'min_autocorr_peak': mauto,
                          'required_rules': rr}
                mean_f1 = evaluate_combo(params, cached_segments)
                results.append((mean_f1, params))
                print(f"Tested {params} -> mean_f1={mean_f1}")

# Sort and print top results
results_sorted = sorted([r for r in results if r[0] is not None], key=lambda x: -x[0])
print('\nTop results:')
for score, params in results_sorted[:10]:
    print(f"F1={score:.3f} -> {params}")

if len(results_sorted)==0:
    print('No valid results (no gait labels or files).')
