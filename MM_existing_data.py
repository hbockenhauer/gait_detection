import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from Kheirkhahan.GSD3_test import KheirkhahanGSD
# from multimob.GSD.GSD3 import KheirkhahanGSD
# from multimob.GSD.GSD4 import MacLeanGSD
# from multimob.GSD.GSD5 import KerenGSD
# from GSD2a import HickeyGSD
import csv
from datetime import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from Kheirkhahan.free_living_test import merge_csv
from process_datasets import process_weargait, process_wisdm,  process_HMP, process_bioclite
# from singleGSD_robust import plot_results
from MM_own_all_robust import merge_all_wrists, process_gait, process_Hickey
from real_time_sim import process_realtime

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
DATA_PATHS = [
    # r"C:\Users\orlov\intern\gait_detection\QSense_data_edge",
    # r"C:\Users\orlov\intern\gait_detection\QSense_data_mixed",
    # r"C:\Users\orlov\intern\gait_detection\QSense_data",
    r"C:\Users\orlov\intern\gait_detection\QSense_data_clinic",
    # r"C:\Users\orlov\intern\gait_detection\Free_living",
    # r"C:\Users\orlov\intern\gait_detection\Datasets\WearGait-PD", 
    # r"C:\Users\orlov\intern\gait_detection\Datasets\wisdm-dataset\raw\watch\accel", 
    # r"C:\Users\orlov\intern\gait_detection\Datasets\HMP_Dataset", 
    # r"C:\Users\orlov\intern\gait_detection\Datasets\Bioclite\data_6activities_plain.mat"
]

# SAMPLING_RATE = 50 
# GAIT_CLASSES = {'walking', 'stairs'}
# CONDITION_KEYWORDS = ['pockets', 'phone', 'rail', 'free', 'crutches', 'walker', 
#                       'cane', 'limp', 'armfixed', 'stroke']
# MIN_SEGMENT_SAMPLES = 9*SAMPLING_RATE 

DEBUG = False
PRINT_STATS = True 
REALTIME = True
# HICKEY = False

SAVE_RESULTS = False 
OUTPUT_FILE = "Results/KheirkhahanGSD_Results_wHickey.csv"

PLOT = False
OUT_FOLDER = r"C:\Users\orlov\intern\gait_detection\Plots\Robust_Kheirkhahan\wHickey"


if __name__ == "__main__":

    if REALTIME:
        for data_path in DATA_PATHS:
            dataset_name = os.path.basename(data_path.rstrip('/\\'))
            if PRINT_STATS:
                print(f"\n{'=' * 80}")
                print(f"  Merging: {dataset_name}")
                print(f"{'=' * 80}")

            # check which dataset to process 
            if "QSense" in dataset_name: 
                rw, lw = merge_all_wrists(data_path)
                rw['dataset'] = dataset_name
                lw['dataset'] = dataset_name
                fl= pd.DataFrame()
                process_realtime(rw, lw, fl,print_stats=PRINT_STATS, save_results=SAVE_RESULTS)

            elif "Free_living" in dataset_name: 
                fl = merge_csv(data_path, PRINT_STATS)
                fl['dataset'] = dataset_name
                rw = pd.DataFrame()
                lw = pd.DataFrame()
                process_realtime(rw, lw, fl, save_results=SAVE_RESULTS)

            elif "WearGait" in dataset_name:
                fl = process_weargait(data_path, PRINT_STATS, realtime=REALTIME)
            
            elif "accel" in dataset_name:
                process_wisdm(data_path, PRINT_STATS, realtime=REALTIME)

            elif "HMP" in dataset_name: 
                process_HMP(data_path, PRINT_STATS, realtime=REALTIME)

            elif "6activities_plain.mat" in dataset_name:
                process_bioclite(data_path, PRINT_STATS, realtime=REALTIME)
    else:
        for data_path in DATA_PATHS:
            dataset_name = os.path.basename(data_path.rstrip('/\\'))
            if PRINT_STATS:
                print(f"\n{'=' * 80}")
                print(f"  Merging: {dataset_name}")
                print(f"{'=' * 80}")

            # check which dataset to process 
            if "QSense" in dataset_name: 
                rw, lw = merge_all_wrists(data_path)
                rw['dataset'] = dataset_name
                lw['dataset'] = dataset_name
                fl= pd.DataFrame()
                process_Hickey(rw, lw, fl, save_results=SAVE_RESULTS)

            elif "Free_living" in dataset_name: 
                fl = merge_csv(data_path, PRINT_STATS)
                fl['dataset'] = dataset_name
                rw = pd.DataFrame()
                lw = pd.DataFrame()
                process_gait(rw, lw, fl, save_results=SAVE_RESULTS)

            elif "WearGait" in dataset_name:
                fl = process_weargait(data_path, PRINT_STATS)
            
            elif "accel" in dataset_name:
                process_wisdm(data_path, PRINT_STATS)

            elif "HMP" in dataset_name: 
                process_HMP(data_path, PRINT_STATS)

            elif "6activities_plain.mat" in dataset_name:
                process_bioclite(data_path, PRINT_STATS)


    ###############################