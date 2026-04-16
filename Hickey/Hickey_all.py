import os
import pandas as pd
import warnings
from Kheirkhahan.free_living_test import merge_csv
from Hickey.process_existing import process_weargait, process_wisdm,  process_HMP, process_bioclite
from Hickey.Hickey_own import merge_all_wrists, process_Hickey

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
SAVE_RESULTS = False  

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
    BIOCLITE_PATH]


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
            process_Hickey(rw, lw, fl, save_results=SAVE_RESULTS)

        elif "WearGait" in dataset_name:
            fl = process_weargait(data_path, PRINT_STATS)
        
        elif "accel" in dataset_name:
            process_wisdm(data_path, PRINT_STATS)

        elif "HMP" in dataset_name: 
            process_HMP(data_path, PRINT_STATS)

        elif "6activities_plain.mat" in dataset_name:
            process_bioclite(data_path, PRINT_STATS)


