import numpy as np
from scipy import signal
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

# implementation based on 
# https://github.com/EWarmerdam/ArmSwingAlgorithm/tree/master 

def arm_swing_algorithm(ang_vel, fs, TH_min_ampl, rmv_first_last=None):
    """
    Python implementation of the Arm Swing Algorithm.
    
    inputs:
    - ang_vel: Nx3 (one arm) or Nx6 (two arms) array of angular velocity in rad/s.
    - fs: Sampling frequency (Hz).
    - TH_min_ampl: Minimum amplitude threshold (degrees).
    - rmv_first_last: Optional list/tuple [start, end] number of swings to remove.
    
    outputs:
            amplitude  = amplitude per swing (range of motion) [deg]
            pk_ang_vel = peak angular velocity per swing [deg/s]
            start_idx  = sample number at which a swing starts
            end_idx    = sample number at which a swing ends
      regularity_angle = regularity of the angular signal (similarity of
                         neighbouring swings; 1 = similar to neighbouring swings) (0-1)
    regularity_ang_vel = regularity of the angular velocity signal
        pk_vel_forward = average peak angular velocity of all the forward swings
       pk_vel_backward = average peak angular velocity of all the backward swings
       perc_time_swing = time during walking that there were swings detected [%]
             frequency = frequency of the arm cycle [Hz]
       perc_both_swing = percentage time during walking bout that there is
                         arm swing detected in both arms [%]
   amplitude_asymmetry = asymmetry of the amplitude between left and right swings (0% means no asymmetry) [%]
peak_velocity_asymmetry= asymmetry of the peak angular velocity between left and right swings [%]
      coordination_max = coordination of the timing between left and right swings (1 when the arms
                        move exactly out of phase with each other) (0-1)
    
                        
    Written by Elke Warmerdam, Kiel University,
    e.warmerdam@neurologie.uni-kiel.de
    
    """
    
    # 1. Preprocessing: Filtering
    # 2nd order Butterworth low-pass filter at 3 Hz
    nyq = 0.5 * fs
    b_lp, a_lp = signal.butter(2, 3 / nyq, btype='low')
    ang_vel_filt = signal.filtfilt(b_lp, a_lp, ang_vel, axis=0)
    
    # Convert to degrees
    ang_vel_deg = np.degrees(ang_vel_filt)
    
    # Determine number of IMUs
    nr_imu = 1 if ang_vel.shape[1] == 3 else 2
    if ang_vel.shape[1] not in [3, 6]:
        raise ValueError("Angular velocity must have 3 or 6 channels.")

    arm_results = []
    pca_signals = []

    for i_imu in range(nr_imu):
        if nr_imu == 2:
            ang_vel_imu = ang_vel_deg[:, 0:3] if i_imu == 0 else ang_vel_deg[:, 3:6]
        else:
            ang_vel_imu = ang_vel_deg

        # 2. PCA and Integration
        # Use only X and Y (channels 0 and 1) for PCA
        data_pca_input = ang_vel_imu[:, 0:2] - np.nanmean(ang_vel_imu[:, 0:2], axis=0)
        pca_model = PCA(n_components=1)
        ang_vel_pca_raw = pca_model.fit_transform(data_pca_input)
        coeff_av = pca_model.components_

        # Align direction (ensure positive swing direction)
        if coeff_av[0, 1] < 0:
            ang_vel_pca_raw *= -1

        # Integration to find Angle
        angle = cumulative_trapezoid(ang_vel_pca_raw.flatten(), dx=1/fs, initial=0)
        
        # 3. Detrending using Symmetric Moving Average
        # Window length is fs (approx 0.5s half-window)
        wts = np.ones(fs) / fs
        # MATLAB conv 'valid' logic
        mov_avg = np.convolve(angle, wts, mode='valid')
        
        # Adjust lengths to match detrended signal
        half_fs = int(0.5 * fs)
        angle_pca = angle[half_fs : len(angle) - half_fs] - mov_avg
        ang_vel_pca = ang_vel_pca_raw[half_fs : len(angle) - half_fs].flatten()

        # 4. Frequency Analysis (FFT)
        window = int(3 * fs)
        steps = int(0.25 * window)
        num_windows = int((len(angle_pca) - window) / steps) + 1
        
        perc_power = np.zeros(num_windows)
        swing_freq = np.zeros(num_windows)

        if len(angle_pca) > window * 1.5:
            for i_w in range(num_windows):
                start = i_w * steps
                end = start + window
                segment = angle_pca[start:end] - np.nanmean(angle_pca[start:end])
                
                # FFT and Power Spectral Density
                freqs, psd = signal.welch(segment, fs, nperseg=window, scaling='spectrum')
                
                # Filter indices for 0.3 - 3 Hz
                mask = (freqs >= 0.3) & (freqs <= 3.0)
                power_band = np.trapz(psd[mask], freqs[mask])
                tot_power = np.trapz(psd, freqs)
                
                perc_power[i_w] = (power_band / tot_power) * 100 if tot_power > 0 else 0
                
                # Dominant frequency
                pks, props = signal.find_peaks(psd[mask])
                if len(pks) > 0:
                    dom_pk_idx = pks[np.argmax(props.get('peak_heights', psd[mask][pks]))]
                    swing_freq[i_w] = freqs[mask][dom_pk_idx]

            # 5. Peak Detection
            # Logic to find peaks based on extracted cycle time
            idx_remove = perc_power < 90
            
            # Find local maxima/minima in angle
            # Python equivalent of findpeaks with MinPeakProminence and MinPeakDistance
            # Here we simplify: detecting all significant peaks
            pos_pks, _ = signal.find_peaks(angle_pca, prominence=2, distance=int(0.6 * fs / (np.nanmedian(swing_freq) or 1)))
            neg_pks, _ = signal.find_peaks(-angle_pca, prominence=2, distance=int(0.6 * fs / (np.nanmedian(swing_freq) or 1)))
            
            all_pks = np.sort(np.concatenate([pos_pks, neg_pks]))
            
            # Filter alternating peaks (Max -> Min -> Max)
            # (Translation of the complex MATLAB alternating logic)
            final_pks = []
            if len(all_pks) > 1:
                for i in range(len(all_pks)-1):
                    p1, p2 = all_pks[i], all_pks[i+1]
                    if (angle_pca[p1] > 0 and angle_pca[p2] < 0) or (angle_pca[p1] < 0 and angle_pca[p2] > 0):
                        if not final_pks or final_pks[-1] != p1:
                            final_pks.append(p1)
                        if i == len(all_pks) - 2:
                            final_pks.append(p2)
            
            final_pks = np.array(final_pks)
            
            # 6. Calculate Swing Parameters
            res = parameter_nan()
            if len(final_pks) > 1:
                amplitudes = np.abs(np.diff(angle_pca[final_pks]))
                peak_vels = []
                for i in range(len(final_pks)-1):
                    segment_vel = ang_vel_pca[final_pks[i]:final_pks[i+1]]
                    peak_vels.append(np.max(np.abs(segment_vel)))
                
                peak_vels = np.array(peak_vels)
                
                # Apply Thresholds
                valid_mask = (amplitudes >= TH_min_ampl) & (peak_vels >= 10)
                res['amplitude'] = amplitudes[valid_mask]
                res['pk_ang_vel'] = peak_vels[valid_mask]
                res['start_idx'] = final_pks[:-1][valid_mask]
                res['end_idx'] = final_pks[1:][valid_mask]
                
                # Regularity via Autocorrelation
                res['regularity_angle'] = auto_cor_wrist(angle_pca, fs)
            
            arm_results.append(res)
            pca_signals.append(ang_vel_pca)

    # 7. Asymmetry and Coordination (If 2 arms)
    final_output = {'left': arm_results[0], 'right': arm_results[1] if nr_imu == 2 else None}
    
    if nr_imu == 2:
        # Simplified Asymmetry Index (ASI)
        l_ampl = np.mean(arm_results[0]['amplitude']) if not np.isnan(arm_results[0]['amplitude']).all() else 0
        r_ampl = np.mean(arm_results[1]['amplitude']) if not np.isnan(arm_results[1]['amplitude']).all() else 0
        if max(l_ampl, r_ampl) > 0:
            final_output['amplitude_asymmetry'] = (l_ampl - r_ampl) / max(l_ampl, r_ampl) * 100
        
        # Coordination: Cross-correlation of PCA signals
        # (Using a slice of the signal for coordination)
        corr = signal.correlate(pca_signals[0], pca_signals[1], mode='same')
        final_output['coordination_max'] = np.abs(np.min(corr) / (np.linalg.norm(pca_signals[0]) * np.linalg.norm(pca_signals[1])))

    return final_output

def auto_cor_wrist(sig, fs):
    """Calculates max autocorrelation for regularity."""
    if len(sig) < fs: return np.nan
    corr = signal.correlate(sig, sig, mode='full')
    corr = corr[len(corr)//2:] # Take second half
    corr /= corr[0] # Normalize
    # Look for first significant peak after 300ms
    start_idx = int(0.3 * fs)
    if len(corr) > start_idx:
        return np.max(corr[start_idx:])
    return np.nan

def parameter_nan():
    return {
        'amplitude': np.nan, 'pk_ang_vel': np.nan, 'start_idx': np.nan,
        'end_idx': np.nan, 'regularity_angle': np.nan
    }

