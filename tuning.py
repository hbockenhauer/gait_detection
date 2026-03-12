from MM_own_all_robust import merge_all_wrists, process_gait
import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
DATA_PATHS = [
    # r"C:\Users\orlov\intern\gait_detection\QSense_data_edge",
    # r"C:\Users\orlov\intern\gait_detection\QSense_data_mixed",
    r"C:\Users\orlov\intern\gait_detection\QSense_data"
]
# GSD_n = 3
SAMPLING_RATE = 50 
DEBUG = False; 
GAIT_CLASSES = {'walking', 'stairs'}
CONDITION_KEYWORDS = ['pockets', 'phone', 'rail', 'free', 'crutches', 'walker', 'cane']
SAVE_RESULTS = False 
PRINT_STATS = False 
MIN_SEGMENT_SAMPLES = 9*SAMPLING_RATE 



if __name__ == "__main__":
    all_rw: list[pd.DataFrame] = []
    all_lw: list[pd.DataFrame] = []

    for data_path in DATA_PATHS:
        dataset_name = os.path.basename(data_path.rstrip('/\\'))
        if PRINT_STATS:
            print(f"\n{'=' * 80}")
            print(f"  Merging: {dataset_name}")
            print(f"{'=' * 80}")
        rw, lw = merge_all_wrists(data_path)
        # Tag each row with its source dataset for traceability
        rw['dataset'] = dataset_name
        lw['dataset'] = dataset_name
        all_rw.append(rw)
        all_lw.append(lw)

    # Pool across all datasets
    rw_merged = pd.concat(all_rw, ignore_index=True) if all_rw else pd.DataFrame()
    lw_merged = pd.concat(all_lw, ignore_index=True) if all_lw else pd.DataFrame()

    # Save pooled merged files
    # rw_merged.to_csv('merged_RW.csv', index=False)
    # lw_merged.to_csv('merged_LW.csv', index=False)
    # print(f"\n[POOLED RW] {len(rw_merged):,} rows → merged_RW.csv")
    # print(f"[POOLED LW] {len(lw_merged):,} rows → merged_LW.csv")

    print(f"\n{'=' * 80}")
    print(f"  Running GSD on pooled data ({len(DATA_PATHS)} dataset(s))")
    print(f"{'=' * 80}")
    process_gait(rw_merged, lw_merged, SAVE_RESULTS)