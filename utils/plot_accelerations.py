# Apply ElderNet gait detection to self-recorded data in QSense_data, recorded at 50Hz
import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import matplotlib.pyplot as plt
import pickle
import random
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from scipy.ndimage import median_filter, uniform_filter1d
import matplotlib.colors as mcolors
import colorsys
import time
import psutil

from models.ElderNet.eldernet_owndata import prepare_windows_overlapping, resample_to_30hz, set_seed
from models.ElderNet.eldernet_owndata import CONF_THRESH, MIN_ENERGY, MAX_ENERGY, MIN_FREQ, MAX_FREQ, WINDOW_SIZE, STEP_SIZE, REPO_NAME, GAIT_CLASSES, SAMPLE_RATE_QSENSE 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.paths import QSENSE_MIXED, PLOTS_DIR

DATASET_PATH = QSENSE_MIXED
PLOT_DATASET_NAME = os.path.basename(DATASET_PATH)

set_seed(42)
# --- RUN ELDERNET AND OBTAIN PROBABILITIES ---
def main():

    for folder in os.listdir(DATASET_PATH):
        if not os.path.isdir(os.path.join(DATASET_PATH, folder)):
            continue

        files = [
            os.path.join(DATASET_PATH, folder, 's1_1RW.txt'),  # Right wrist
            os.path.join(DATASET_PATH, folder, 's2_2LW.txt')   # Left wrist 
        ]

        # ADD these two lines before the for file in files loop
        right_df = None
        left_df = None

        for file in files:
            if not os.path.exists(file):
                continue
            wrist = "right" if "1RW" in file else "left"

            

            try:
                df_30hz = resample_to_30hz(file)

                # ADD inside the for file in files loop (after df_30hz is loaded)
                if wrist == "right":
                    right_df = df_30hz
                else:
                    left_df = df_30hz

                ## ADD this block AFTER the for file in files loop
                if right_df is not None and left_df is not None:
                    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

                    # Right wrist
                    axs[0].plot(right_df['time_sec'], right_df['accX'], label='accX')
                    axs[0].plot(right_df['time_sec'], right_df['accY'], label='accY')
                    axs[0].plot(right_df['time_sec'], right_df['accZ'], label='accZ')
                    axs[0].set_title(f"{folder} - Right wrist")
                    axs[0].set_ylabel("Acceleration")
                    axs[0].legend()
                    axs[0].grid()

                    # Left wrist
                    axs[1].plot(left_df['time_sec'], left_df['accX'], label='accX')
                    axs[1].plot(left_df['time_sec'], left_df['accY'], label='accY')
                    axs[1].plot(left_df['time_sec'], left_df['accZ'], label='accZ')
                    axs[1].set_title(f"{folder} - Left wrist")
                    axs[1].set_xlabel("Time (s)")
                    axs[1].set_ylabel("Acceleration")
                    axs[1].legend()
                    axs[1].grid()

                    plt.tight_layout()
                    plt.show()

                    # save to disk
                    save_path = os.path.join(PLOTS_DIR, PLOT_DATASET_NAME, 'eldernet', f"{folder}_raw_acceleration.png")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    fig.savefig(save_path)
                                
            except Exception as e:
                print(f"Error processing {os.path.basename(file)}: {e}")
                continue

if __name__ == "__main__":main()