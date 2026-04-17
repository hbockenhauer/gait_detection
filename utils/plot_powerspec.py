'''
Power Spectral Density (PSD) Analysis for QSense Dataset
This script loads accelerometer data from the QSense dataset, selects only the walking samples, and computes the Power Spectral Density (PSD) for each axis (X, Y, Z) using Welch's method. 
The resulting PSDs are plotted on a logarithmic scale for better visualization of the frequency components.
'''

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.paths import QSENSE_MIXED

DATASET_PATH = QSENSE_MIXED
SAMPLE_RATE_QSENSE = 50.0  # Original sampling rate

# --- LOAD DATA (fixed for Label column) ---
def load_data(filepath):
    df = pd.read_csv(filepath, sep=None, engine="python")
    df = df.reset_index(drop=True)

    # Determine correct label column
    if 'label' in df.columns:
        label_col = 'label'
    elif 'Label' in df.columns:
        label_col = 'Label'
    else:
        raise ValueError(f"No 'label' or 'Label' column in {filepath}")
    
    # Keep only accelerometer and label
    df = df[['accX','accY','accZ', label_col]].apply(pd.to_numeric, errors='coerce')
    
    # Select only walking samples
    df = df[df[label_col] == 1]
    
    return df[['accX','accY','accZ']]
# --- COMPUTE PSD ---
def compute_psd(acc_array, fs=50.0):
    freqs_list, psd_list = [], []
    for axis in range(acc_array.shape[1]):
        f, Pxx = welch(acc_array[:, axis], fs=fs, nperseg=256)
        freqs_list.append(f)
        psd_list.append(Pxx)
    return freqs_list, psd_list

# --- MAIN ANALYSIS FUNCTION ---
def analyze_psd(folder_path):
    wrists = {'right': 's1_1RW.txt', 'left': 's2_2LW.txt'}
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    fig.suptitle(f"PSD Analysis (Walking Only): {os.path.basename(folder_path)}", fontsize=16)
    
    for i, (wrist, filename) in enumerate(wrists.items()):
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        df = load_data(file_path)
        if df.empty:
            print(f"No walking samples found in {file_path}")
            continue
        
        acc_array = df.values
        freqs_list, psd_list = compute_psd(acc_array, fs=SAMPLE_RATE_QSENSE)
        
        # Plot each axis
        colors = ['r', 'g', 'b']
        for j, axis in enumerate(['X','Y','Z']):
            axes[i].semilogy(freqs_list[j], psd_list[j], color=colors[j], label=f'acc{axis}')
        
        axes[i].set_title(f"{wrist.capitalize()} wrist")
        axes[i].set_xlabel("Frequency [Hz]")
        axes[i].set_ylabel("PSD [g^2/Hz]")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

# --- RUN OVER ALL FOLDERS ---
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(folder_path):
        analyze_psd(folder_path)