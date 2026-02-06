import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class WristGaitDetector:
    """
    Signal processing-based gait detection for wrist-worn IMUs
    
    Based on the principle that gait produces:
    1. Periodic arm swing (1-2 Hz dominant frequency)
    2. Sufficient movement amplitude
    3. Harmonic structure in frequency domain
    4. Regularity in time domain
    """
    
    def __init__(self, 
                 sampling_rate=100.0,
                 window_size_sec=3.0,
                 step_size_sec=0.5,
                 # Frequency domain parameters
                 gait_freq_range=(0.8, 2.5),  # Expected arm swing frequency
                 min_harmonic_ratio=0.2,      # Ratio of 2nd harmonic to fundamental
                 # Time domain parameters
                 min_amplitude=0.3,           # Minimum movement amplitude (m/s²)
                 max_amplitude=15.0,          # Maximum (filters out impacts)
                 # Regularity parameters
                 min_autocorr_peak=0.3,       # Minimum autocorrelation peak
                 # Filtering
                 lowpass_cutoff=10.0,
                 # Other
                 use_gyro=True,
                 required_rules=3):           # number of rules that must be satisfied
        
        self.fs = sampling_rate
        self.window_size = int(window_size_sec * sampling_rate)
        self.step_size = int(step_size_sec * sampling_rate)
        self.gait_freq_range = gait_freq_range
        self.min_harmonic_ratio = min_harmonic_ratio
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude
        self.min_autocorr_peak = min_autocorr_peak
        self.lowpass_cutoff = lowpass_cutoff
        self.use_gyro = use_gyro
        self.required_rules = required_rules
        
    def _lowpass_filter(self, data, cutoff=None):
        """Apply Butterworth lowpass filter to remove high-frequency noise"""
        if cutoff is None:
            cutoff = self.lowpass_cutoff
            
        nyquist = 0.5 * self.fs
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='low')
        
        # Filter each column
        filtered = data.copy()
        for col in data.columns:
            if col in ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']:
                filtered[col] = signal.filtfilt(b, a, data[col])
        
        return filtered
    
    def _compute_magnitude(self, data, sensor_type='acc'):
        """Compute magnitude of accelerometer or gyroscope"""
        if sensor_type == 'acc':
            cols = ['acc_x', 'acc_y', 'acc_z']
        else:  # gyr
            cols = ['gyr_x', 'gyr_y', 'gyr_z']
        # Handle missing columns gracefully by substituting zeros
        n = len(data)
        vals = np.zeros((n, 3), dtype=float)
        for i, c in enumerate(cols):
            if c in data.columns:
                vals[:, i] = data[c].values
            else:
                vals[:, i] = 0.0

        return np.sqrt(np.sum(vals**2, axis=1))
    
    def _get_dominant_frequency(self, signal_window):
        """
        Find dominant frequency using FFT
        
        Returns:
        --------
        dominant_freq : float
            Frequency with maximum power in gait range
        spectral_power : float
            Power at dominant frequency
        harmonic_ratio : float
            Ratio of 2nd harmonic power to fundamental
        """
        # Remove DC component
        signal_window = signal_window - np.mean(signal_window)
        
        # Compute FFT
        fft_vals = np.abs(rfft(signal_window))
        freqs = rfftfreq(len(signal_window), d=1/self.fs)
        
        # Find dominant frequency in gait range
        gait_mask = (freqs >= self.gait_freq_range[0]) & (freqs <= self.gait_freq_range[1])
        
        if not np.any(gait_mask):
            return 0.0, 0.0, 0.0
        
        # Get dominant frequency
        gait_freqs = freqs[gait_mask]
        gait_power = fft_vals[gait_mask]
        
        if len(gait_power) == 0:
            return 0.0, 0.0, 0.0
        
        max_idx = np.argmax(gait_power)
        dominant_freq = gait_freqs[max_idx]
        spectral_power = gait_power[max_idx]
        
        # Check for harmonic structure (2nd harmonic)
        harmonic_freq = dominant_freq * 2
        harmonic_tolerance = 0.3  # Hz
        
        harmonic_mask = (freqs >= harmonic_freq - harmonic_tolerance) & \
                       (freqs <= harmonic_freq + harmonic_tolerance)
        
        if np.any(harmonic_mask):
            harmonic_power = np.max(fft_vals[harmonic_mask])
            harmonic_ratio = harmonic_power / (spectral_power + 1e-10)
        else:
            harmonic_ratio = 0.0
        
        return dominant_freq, spectral_power, harmonic_ratio
    
    def _compute_autocorrelation(self, signal_window):
        """
        Compute autocorrelation to measure periodicity
        
        Returns:
        --------
        max_autocorr : float
            Maximum autocorrelation (excluding lag 0)
        regularity_score : float
            Measure of signal regularity
        """
        # Normalize signal
        signal_norm = signal_window - np.mean(signal_window)
        signal_std = np.std(signal_norm)
        
        if signal_std < 1e-10:
            return 0.0, 0.0
        
        signal_norm = signal_norm / signal_std
        
        # Compute autocorrelation
        autocorr = np.correlate(signal_norm, signal_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find peaks (excluding lag 0)
        min_lag = int(0.4 * self.fs)  # Minimum 0.4 sec period (2.5 Hz)
        max_lag = int(1.25 * self.fs)  # Maximum 1.25 sec period (0.8 Hz)
        
        if max_lag >= len(autocorr):
            max_lag = len(autocorr) - 1
        
        search_region = autocorr[min_lag:max_lag]
        
        if len(search_region) == 0:
            return 0.0, 0.0
        
        max_autocorr = np.max(search_region)
        
        # Regularity: ratio of max peak to mean
        regularity_score = max_autocorr / (np.mean(np.abs(autocorr[1:])) + 1e-10)
        
        return max_autocorr, regularity_score
    
    def _compute_jerk(self, acc_magnitude):
        """
        Compute jerk (rate of change of acceleration)
        High jerk indicates rapid movements
        """
        jerk = np.diff(acc_magnitude)
        return np.std(jerk)
    
    def _extract_window_features(self, window_data):
        """
        Extract all features from a single window
        
        Returns:
        --------
        features : dict
            Dictionary of feature values
        """
        # Compute magnitudes
        acc_mag = self._compute_magnitude(window_data, 'acc')
        gyr_mag = self._compute_magnitude(window_data, 'gyr')
        
        # Detrend magnitude to remove gravity baseline for amplitude measures
        acc_mag_d = acc_mag - np.median(acc_mag)

        # Amplitude features (use detrended magnitude)
        acc_amplitude = np.std(acc_mag_d)
        acc_range = np.ptp(acc_mag_d)  # Peak-to-peak
        gyr_amplitude = np.std(gyr_mag)
        
        # Frequency domain features (acc + gyr)
        dom_freq, spectral_power, harmonic_ratio = self._get_dominant_frequency(acc_mag_d)
        gyr_dom_freq, gyr_spectral_power, gyr_harmonic_ratio = self._get_dominant_frequency(gyr_mag)
        
        # Autocorrelation features (use detrended)
        max_autocorr, regularity = self._compute_autocorrelation(acc_mag_d)

        # Jerk feature (use detrended)
        jerk = self._compute_jerk(acc_mag_d)

        # Signal energy (detrended)
        acc_energy = np.sum(acc_mag_d**2) / len(acc_mag_d)
        
        features = {
            'acc_amplitude': acc_amplitude,
            'acc_range': acc_range,
            'gyr_amplitude': gyr_amplitude,
            'dominant_freq': dom_freq,
            'spectral_power': spectral_power,
            'harmonic_ratio': harmonic_ratio,
            'gyr_dominant_freq': gyr_dom_freq,
            'gyr_spectral_power': gyr_spectral_power,
            'gyr_harmonic_ratio': gyr_harmonic_ratio,
            'max_autocorr': max_autocorr,
            'regularity': regularity,
            'jerk': jerk,
            'acc_energy': acc_energy
        }
        
        return features
    
    def _classify_window(self, features, effective_min_amplitude=None):
        """
        Rule-based classification using extracted features
        
        Returns:
        --------
        is_gait : bool
            Whether window contains gait
        confidence : float
            Confidence score (0-1)
        """
        # Rule 1: Frequency in gait range (accept acc OR gyro if available)
        acc_freq_ok = (features['dominant_freq'] >= self.gait_freq_range[0] and 
                       features['dominant_freq'] <= self.gait_freq_range[1] and
                       features['spectral_power'] > 1e-4)
        gyr_freq_ok = False
        if self.use_gyro and 'gyr_dominant_freq' in features:
            gyr_freq_ok = (features['gyr_dominant_freq'] >= self.gait_freq_range[0] and
                           features['gyr_dominant_freq'] <= self.gait_freq_range[1] and
                           features['gyr_spectral_power'] > 1e-4)

        freq_ok = acc_freq_ok or gyr_freq_ok
        
        # Rule 2: Sufficient amplitude
        # Adaptive amplitude: allow lowering threshold for low-motion recordings
        if effective_min_amplitude is None:
            amp_threshold = self.min_amplitude
        else:
            amp_threshold = effective_min_amplitude

        amp_ok = (features['acc_amplitude'] >= amp_threshold and 
                  features['acc_amplitude'] <= self.max_amplitude)
        
        # Rule 3: Harmonic structure (indicates periodic movement)
        # Accept harmonic from either acc or gyro
        harmonic_ok = (features['harmonic_ratio'] >= self.min_harmonic_ratio)
        if self.use_gyro and 'gyr_harmonic_ratio' in features:
            harmonic_ok = harmonic_ok or (features['gyr_harmonic_ratio'] >= self.min_harmonic_ratio)
        
        # Rule 4: Regularity (autocorrelation)
        regular_ok = features['max_autocorr'] >= self.min_autocorr_peak
        
        # Rule 5: Not too jerky (excludes impacts, falls)
        jerk_ok = features['jerk'] < 80.0  # allow slightly more movement for wrist
        
        # Count satisfied rules
        rules = [freq_ok, amp_ok, harmonic_ok, regular_ok, jerk_ok]
        satisfied = sum(rules)

        # Require configurable number of rules
        is_gait = satisfied >= self.required_rules

        # Confidence is fraction of satisfied rules
        confidence = satisfied / len(rules)
        
        return is_gait, confidence
    
    def detect(self, data, return_features=False):
        """
        Detect gait in continuous IMU data
        
        Parameters:
        -----------
        data : pd.DataFrame
            IMU data with columns: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z
        return_features : bool
            If True, also return extracted features
        
        Returns:
        --------
        predictions : np.array
            Binary predictions (0=not gait, 1=gait) per window
        timestamps : np.array
            Start time of each window
        confidences : np.array
            Confidence scores per window
        features_list : list (optional)
            List of feature dictionaries per window
        """
        # Apply lowpass filter
        filtered_data = self._lowpass_filter(data)
        
        predictions = []
        confidences = []
        timestamps = []
        features_list = []
        # Compute adaptive amplitude threshold from global data
        try:
            acc_mag_all = self._compute_magnitude(filtered_data, 'acc')
            global_std = np.std(acc_mag_all)
            effective_min_amp = min(self.min_amplitude, max(global_std * 0.5, 0.05))
        except Exception:
            effective_min_amp = self.min_amplitude

        # Sliding window processing
        for i in range(0, len(filtered_data) - self.window_size + 1, self.step_size):
            window = filtered_data.iloc[i:i + self.window_size]
            
            # Extract features
            features = self._extract_window_features(window)
            
            # Classify
            is_gait, confidence = self._classify_window(features, effective_min_amplitude=effective_min_amp)
            
            predictions.append(int(is_gait))
            confidences.append(confidence)
            timestamps.append(i / self.fs)
            
            if return_features:
                features_list.append(features)
        
        predictions = np.array(predictions)
        timestamps = np.array(timestamps)
        confidences = np.array(confidences)
        
        # Post-processing: Median filter to smooth predictions (larger window)
        predictions = median_filter(predictions, size=5)
        
        if return_features:
            return predictions, timestamps, confidences, features_list
        else:
            return predictions, timestamps, confidences
    
    def detect_bouts(self, predictions, timestamps, min_bout_duration=3.0):
        """
        Detect gait bouts from binary predictions
        
        Parameters:
        -----------
        predictions : np.array
            Binary predictions
        timestamps : np.array
            Timestamps of predictions
        min_bout_duration : float
            Minimum duration (seconds) for a valid gait bout
        
        Returns:
        --------
        bouts : list of tuples
            Each tuple is (start_time, end_time, duration)
        """
        bouts = []
        in_bout = False
        bout_start = 0
        
        for i, (pred, ts) in enumerate(zip(predictions, timestamps)):
            if pred == 1 and not in_bout:
                # Start of gait bout
                in_bout = True
                bout_start = ts
            elif pred == 0 and in_bout:
                # End of gait bout
                bout_end = timestamps[i-1]
                duration = bout_end - bout_start
                
                if duration >= min_bout_duration:
                    bouts.append((bout_start, bout_end, duration))
                
                in_bout = False
        
        # Handle case where bout extends to end
        if in_bout:
            bout_end = timestamps[-1]
            duration = bout_end - bout_start
            if duration >= min_bout_duration:
                bouts.append((bout_start, bout_end, duration))
        
        return bouts
    

import os
import glob
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def load_wisdm_6axis(acc_path):
    gyro_path = acc_path.replace("accel", "gyro")

    def read_file(path, prefix):
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip().rstrip(";")
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 6:
                    continue
                try:
                    rows.append((
                        parts[1],                  # activity
                        float(parts[2]),            # timestamp (ns)
                        float(parts[3]),            # x
                        float(parts[4]),            # y
                        float(parts[5])             # z
                    ))
                except ValueError:
                    continue

        df = pd.DataFrame(
            rows,
            columns=["activity", "timestamp", f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
        )
        return df.sort_values("timestamp")

    if not os.path.exists(acc_path):
        return pd.DataFrame()

    df_acc = read_file(acc_path, "acc")
    if df_acc.empty:
        return pd.DataFrame()

    if os.path.exists(gyro_path):
        df_gyr = read_file(gyro_path, "gyr")
    else:
        df_gyr = pd.DataFrame()

    # Merge BEFORE normalization
    if not df_gyr.empty:
        df = pd.merge_asof(
            df_acc,
            df_gyr.drop(columns=["activity"]),
            on="timestamp",
            direction="nearest",
            tolerance=5e7  # 50 ms max gap
        )
    else:
        df = df_acc.copy()
        df[["gyr_x", "gyr_y", "gyr_z"]] = 0.0

    # Fill missing gyro safely
    for c in ["gyr_x", "gyr_y", "gyr_z"]:
        df[c] = df[c].fillna(0.0)

    # Normalize timestamp to seconds (relative)
    df["timestamp"] = (df["timestamp"] - df["timestamp"].iloc[0]) * 1e-9

    return df.reset_index(drop=True)

def resample_to_100hz(df, fs=100.0):
    df = df.sort_values("timestamp")
    t = df["timestamp"].values

    if len(t) < 2:
        return pd.DataFrame()

    dt = np.diff(t)
    if np.median(dt) > 0.05:
        return pd.DataFrame()  # too sparse

    new_t = np.arange(t[0], t[-1], 1/fs)

    resampled = {"timestamp": new_t}
    for col in ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]:
        resampled[col] = np.interp(new_t, t, df[col].values)

    resampled = pd.DataFrame(resampled)

    resampled["activity"] = pd.merge_asof(
        resampled[["timestamp"]],
        df[["timestamp", "activity"]],
        on="timestamp",
        direction="nearest"
    )["activity"]

    return resampled


if __name__ == '__main__':
    # Process WISDM dataset
    # Default to a workspace-relative path but allow override with env var
    BASE_PATH = os.environ.get("WISDM_PATH", "wisdm-dataset/raw/watch")
    if not os.path.exists(BASE_PATH):
        alt = os.path.join("wisdm-dataset", "raw", "watch")
        if os.path.exists(alt):
            BASE_PATH = alt
        else:
            print(f"Warning: WISDM base path '{BASE_PATH}' not found. Set WISDM_PATH env var or place dataset at 'wisdm-dataset/raw/watch'.")

    ACC_FOLDER = os.path.join(BASE_PATH, "accel")
    if not os.path.exists(ACC_FOLDER):
        print(f"Warning: accel folder not found at {ACC_FOLDER}. No files will be processed.")
        files = []
    else:
        files = sorted(glob.glob(os.path.join(ACC_FOLDER, "*.txt")))

    GAIT_ACTIVITIES = {'A', 'C'}  # Walk, Stairs
    detector = WristGaitDetector(sampling_rate=100.0)

    # ----------------------------
    # Robust processing for WISDM watch
    # ----------------------------

    all_results = []

    for file_path in files[:5]:  # first 5 subjects
        subj_id = os.path.basename(file_path).split('_')[1]
        print(f"\nProcessing Subject {subj_id}...")

        try:
            # Load raw data
            df_raw = load_wisdm_6axis(file_path)
            if df_raw.empty:
                print("  Skipping: no raw data")
                continue

            # Resample to 100 Hz
            df_100hz = resample_to_100hz(df_raw)
            if df_100hz.empty:
                print("  Skipping: resampled data empty")
                continue

            # Detect gait
            predictions, timestamps, confidences = detector.detect(df_100hz)

            # Ground truth
            y_true_full = (df_100hz['activity'].isin(GAIT_ACTIVITIES)).astype(int).values
            y_pred_full = np.zeros(len(df_100hz), dtype=int)

            # Map window predictions to sample-level
            win_len = detector.window_size
            step_len = detector.step_size
            for i, pred in enumerate(predictions):
                start_idx = i * step_len
                end_idx = min(start_idx + win_len, len(y_pred_full))
                if pred == 1:
                    y_pred_full[start_idx:end_idx] = 1

            # Compute metrics safely
            if np.sum(y_true_full) == 0:
                print("  Warning: no gait labels in ground truth")
                p = r = f1 = acc = 0.0
            else:
                p, r, f1, _ = precision_recall_fscore_support(
                    y_true_full, y_pred_full, labels=[1], average='binary', zero_division=0
                )
                acc = accuracy_score(y_true_full, y_pred_full)

            print(f"  Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}, Accuracy: {acc:.3f}")

            all_results.append({
                'Subject': subj_id,
                'Precision': p,
                'Recall': r,
                'F1': f1,
                'Accuracy': acc
            })

        except Exception as e:
            print(f"  Error processing subject {subj_id}: {e}")

    # Summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        print("\n" + "="*60)
        print("OVERALL PERFORMANCE")
        print("="*60)
        print(results_df[['Precision','Recall','F1','Accuracy']].mean())
    else:
        print("No valid subjects processed.")
