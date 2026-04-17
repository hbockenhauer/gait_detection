import os
import pandas as pd
import warnings
from models.Kheirkhahan.free_living_test import merge_csv
from models.Kheirkhahan.process_datasets import process_weargait, process_wisdm,  process_HMP, process_bioclite
from models.Kheirkhahan.MM_own_all_robust import merge_all_wrists, process_gait

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
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
    RESULTS_DIR, 
    PLOTS_DIR
)

############## can be adjusted ####################
DEBUG = False
PRINT_STATS = True 

SAVE_RESULTS = True 
OUTPUT_FILE = "Results/KheirkhahanGSD_Results_wHickey.csv"
###################################################


DATA_PATHS = [ 
    HMP_PATH, 
    WISDM_PATH, 
    WEARGAIT_PD,
    WEARGAIT_CTRL,
    QSENSE_DATA,
    QSENSE_EDGE,
    QSENSE_MIXED,
    QSENSE_CLINIC,
    FREELIVING_PATH,
    BIOCLITE_PATH
    ]


if __name__ == "__main__":
    for data_path in DATA_PATHS:
        dataset_name = os.path.basename(data_path.rstrip('/\\'))
        if PRINT_STATS:
            print(f"\n{'=' * 80}")
            print(f"  Merging: {dataset_name}")
            print(f"{'=' * 80}")

        # check which dataset to process 
        if any(x in dataset_name for x in ["Baseline", "Clinical", "Edge_Cases", "Multiple"]):
            rw, lw = merge_all_wrists(data_path)
            rw['dataset'] = dataset_name
            lw['dataset'] = dataset_name
            fl= pd.DataFrame()
            process_gait(rw, lw, fl, save_results=SAVE_RESULTS)

        elif "Free_living" in dataset_name: 
            fl = merge_csv(data_path, PRINT_STATS)
            fl['dataset'] = dataset_name
            rw = pd.DataFrame()
            lw = pd.DataFrame()
            process_gait(rw, lw, fl, save_results=SAVE_RESULTS)

        elif "WearGait" in dataset_name:
            fl = process_weargait(data_path, PRINT_STATS, SAVE_RESULTS, 
                                  output_file="Kheirkhahan/WearGait.csv")
        
        elif "accel" in dataset_name:
            process_wisdm(data_path, PRINT_STATS, SAVE_RESULTS, 
                                  output_file="Kheirkhahan/WISDM.csv")

        elif "HMP" in dataset_name: 
            process_HMP(data_path, PRINT_STATS, SAVE_RESULTS, 
                                  output_file="Kheirkhahan/HMP.csv")

        elif "6activities_plain.mat" in dataset_name:
            process_bioclite(data_path, PRINT_STATS, SAVE_RESULTS, 
                                  output_file="Kheirkhahan/Bioclite.csv")


    ###############################