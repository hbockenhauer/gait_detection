"""
Runs the Hickey algorithm on all existing datasets. 
It cannot be tested on HMP and WISDM datasets without adjusting the 
low-pass filters cutoff frequency in the main algorithm, therefore they 
are commented out. 
"""

import os
import pandas as pd
import warnings
from models.Kheirkhahan.free_living_test import merge_csv
from models.Hickey.process_existing import process_weargait, process_wisdm,  process_HMP, process_bioclite
from models.Hickey.Hickey_own import merge_all_wrists, process_Hickey

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
    BIOCLITE_PATH
)

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

#################################
DEBUG = False
PRINT_STATS = True 

# to save all the restuls from the datasets run 
# names for each files with the dataset results are preset 
# but can be adjusted by providing it as an input to the 
# process functions
SAVE_RESULTS = True  
##################################

DATA_PATHS = [ 
    # HMP_PATH, # needs adjustment cuz to fs
    # WISDM_PATH, # needs adjustment cuz to fs
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
            # process_Hickey(rw, lw, fl, print_stats=PRINT_STATS,
            #                save_results=SAVE_RESULTS, output_file="Custom_name.csv")
            process_Hickey(rw, lw, fl, PRINT_STATS, SAVE_RESULTS)

        elif "Free_living" in dataset_name: 
            fl = merge_csv(data_path, PRINT_STATS)
            fl['dataset'] = dataset_name
            rw = pd.DataFrame()
            lw = pd.DataFrame()
            process_Hickey(rw, lw, fl, PRINT_STATS, SAVE_RESULTS)

        elif "WearGait" in dataset_name:
            fl = process_weargait(data_path, PRINT_STATS, SAVE_RESULTS)
        
        elif "accel" in dataset_name:
            process_wisdm(data_path, PRINT_STATS, SAVE_RESULTS)

        elif "HMP" in dataset_name: 
            process_HMP(data_path, PRINT_STATS, SAVE_RESULTS)

        elif "6activities_plain.mat" in dataset_name:
            process_bioclite(data_path, PRINT_STATS, SAVE_RESULTS)


