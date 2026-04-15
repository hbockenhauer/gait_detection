import os
import pandas as pd
import warnings
from ..free_living_test import merge_csv
from ..process_datasets import process_weargait, process_wisdm,  process_HMP, process_bioclite
from ..MM_own_all_robust import merge_all_wrists, process_gait, process_Hickey


from config.paths import (
    HMP_PATH,
    WISDM_PATH,
    WEARGAIT_PD,
    WEARGAIT_CTRL,
    QSENSE_DATA,
    QSENSE_EDGE,
    QSENSE_MIXED,
    QSENSE_CLINIC,
    FREELIVING_PATH,
    BIOCLITE_PATH,
    STROKENET_WEIGHTS,
    PLOTS_DIR as OUTPUT_PLOTS_DIR,
    RESULTS_DIR,
)

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
# DATA_PATHS = [
#     # r"C:\Users\orlov\intern\gait_detection\QSense_data_edge",
#     # r"C:\Users\orlov\intern\gait_detection\QSense_data_mixed",
#     # r"C:\Users\orlov\intern\gait_detection\QSense_data",
#     r"C:\Users\orlov\intern\gait_detection\QSense_data_clinic",
#     # r"C:\Users\orlov\intern\gait_detection\Free_living",
#     # r"C:\Users\orlov\intern\gait_detection\Datasets\WearGait-PD", 
#     # r"C:\Users\orlov\intern\gait_detection\Datasets\wisdm-dataset\raw\watch\accel", 
#     # r"C:\Users\orlov\intern\gait_detection\Datasets\HMP_Dataset", 
#     # r"C:\Users\orlov\intern\gait_detection\Datasets\Bioclite\data_6activities_plain.mat"
# ]

DATA_PATHS = [    HMP_PATH,
    WISDM_PATH,
    WEARGAIT_PD,
    WEARGAIT_CTRL,
    QSENSE_DATA,
    QSENSE_EDGE,
    QSENSE_MIXED,
    QSENSE_CLINIC,
    FREELIVING_PATH,
    BIOCLITE_PATH,
    STROKENET_WEIGHTS]

DEBUG = False
PRINT_STATS = True 


SAVE_RESULTS = False 
OUTPUT_FILE = "Results/KheirkhahanGSD_Results_wHickey.csv"

PLOT = False
OUT_FOLDER = r"C:\Users\orlov\intern\gait_detection\Plots\Robust_Kheirkhahan\wHickey"


if __name__ == "__main__":
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


