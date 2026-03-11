import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
#from multimob.GSD.GSD3 import KheirkhahanGSD
from GSD2_test import HickeyGSD
from GSD3_test import KheirkhahanGSD
import matplotlib.pyplot as plt
import csv
import datetime
from datetime import time
import matplotlib.ticker as mticker

# Suppress the DtypeWarning for the walkway columns
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
DATA_PATH = r"C:\Users\orlov\intern\gait_detection\QSense_tests\Overnight_test"
file_name = "s2_2LW.txt" #"s1_1RW.txt"
SAMPLING_RATE = 50 
DEBUG = True
MIN_SEGMENT_SAMPLES = 9*SAMPLING_RATE  # requirement for 9s window 
PLOT = False

# def parse_time(t_str):
#     h, m, s_ms = t_str.strip().split(':')
#     s, ms = s_ms.split('.')
#     return time(int(h), int(m), int(s), int(ms) * 1000)


def parse_time(t_str, d_str):

    d = datetime.strptime(d_str.strip(), '%Y-%m-%d')
    h, m, s_ms = t_str.strip().split(':')
    s, ms = s_ms.split('.')
    for i in np.arange(5):
        print('day is',d)
        print("time is", h, m, s_ms)
    return datetime(d.year, d.month, d.day, int(h), int(m), int(s), int(ms) * 1000)

def load_segmented(DATA_PATH, file_name) -> pd.DataFrame:
    try:
        # open the file 
        with open(os.path.join(DATA_PATH, file_name), newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)

        # clip the first 10 seconds depending on the data path 
        rows = rows if "mixed" in DATA_PATH else rows[500:]
        if DEBUG == True:
            print("Data taken fully.") if "mixed" in DATA_PATH else print("First 10s are clipped.")

        clean_rows = []
        segments   = []
        max_time   = None
        prev_time  = None
        segment_id = 0
        dropped_rows = 0
        

        for row in rows:
            try:
                t = parse_time(row['HH:mm:ss.fff'], row['yyyy-MM-dd'])
                # print(t)
                # # day = row['yyyy-MM-dd']
            except Exception:
                continue

            if max_time is None or t > max_time:
                if prev_time is not None:
                    # compute gap in ms (handles minute/hour rollover simply)
                    # gap_ms = (t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
                    #           - prev_time.hour * 3600 - prev_time.minute * 60 - prev_time.second - prev_time.microsecond / 1e6) * 1000
                    gap_ms = (t - prev_time).total_seconds() * 1000
                    for i in np.arange(5):
                        print('gap is',gap_ms)
                    
                    if gap_ms > ((1001/SAMPLING_RATE)):
                        segment_id += 1
                # if day != prev_day:
                #     max_time = None
                max_time  = t
                prev_time = t  
                # prev_day = row['yyyy-MM-dd'] 

                clean_rows.append(row)
                segments.append(segment_id)
            else:
                dropped_rows += 1
        
        if DEBUG:
            print(f"Dropped {dropped_rows} rows.")
            print(f"Found {segment_id+1} segments. ")
            print(f"Kept {len(clean_rows)} rows. \n")
        df = pd.DataFrame(clean_rows)
        df = df.reset_index(drop=True)
        df['segment'] = segments
        # df.columns are now :
        # ['yyyy-MM-dd', 'HH:mm:ss.fff', 'gyrX', 'gyrY', 'gyrZ', 
        # 'accX', 'accY', 'accZ', 'magX', 'magY', 'magZ', 
        # 'Marker', 'Energy', 'Angle', 'Classification', 'Label', 'segment']
    except Exception as e:
        print(f"{file_name[:25]:<25} | ERROR: {str(e)}")
    
    return df



if __name__ == "__main__":

    results = []
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.txt')]
    #files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv') and (f.startswith('W') or f.startswith('N'))]
    
    print(f"{'Subject':<25} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 75)

    
    try:
        # Load Data
        df = load_segmented(DATA_PATH, file_name)

    except Exception as e:
        print(f"{file_name[:25]:<25} | ERROR: {str(e)}")

    # Final Summary
    if results:
        res_df = pd.DataFrame(results)
        print("-" * 75)
        
        # Separate results by file type
        rw_files = res_df[res_df['Subject'].str.endswith('RW.txt')]
        other_files = res_df[~res_df['Subject'].str.endswith('RW.txt')]
        
        # Print RW.txt average
        if not rw_files.empty:
            print(f"{'AVERAGE right wrist ':<25} | {rw_files['Accuracy'].mean():.2f}   | {rw_files['Precision'].mean():.2f}   | {rw_files['Recall'].mean():.2f}   | {rw_files['F1'].mean():.2f}")
        
        # Print other files average
        if not other_files.empty:
            print(f"{'AVERAGE left wrist':<25} | {other_files['Accuracy'].mean():.2f}   | {other_files['Precision'].mean():.2f}   | {other_files['Recall'].mean():.2f}   | {other_files['F1'].mean():.2f}")
        
        #res_df.to_csv('HickeyGSD_Results.csv', index=False)
        plt.show()