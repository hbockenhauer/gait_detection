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
from datetime import time

# Suppress the DtypeWarning for the walkway columns
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
DATA_PATH = r"C:\Users\orlov\intern\gait_detection\QSense_data_mixed\test8_Hendrik"
file_name = "s2_2LW.txt"
SAMPLING_RATE = 50 
DEBUG = True
MIN_SEGMENT_SAMPLES = 9*SAMPLING_RATE  # requirement for the band pass filter

def parse_time(t_str):
    h, m, s_ms = t_str.strip().split(':')
    s, ms = s_ms.split('.')
    return time(int(h), int(m), int(s), int(ms) * 1000)

if __name__ == "__main__":

    results = []
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.txt')]
    #files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv') and (f.startswith('W') or f.startswith('N'))]
    
    print(f"{'Subject':<25} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 75)

    
    try:
        # 1. Load Data
        # df = pd.read_csv(os.path.join(DATA_PATH, file_name), 
        #                 sep='\t',  # Use whitespace as separator (adjust if needed)
        #                 low_memory=False)
        # #### CLIPPING THE FIST 10 SECONDS
        # df = df[500:]

        with open(os.path.join(DATA_PATH, file_name), newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)

        rows = rows if "mixed" in DATA_PATH else rows[500:]
        print("data is full") if "mixed" in DATA_PATH else print("10s clipped")

        clean_rows = []
        segments   = []
        max_time   = None
        prev_time  = None
        segment_id = 0
        dropped_rows = 0
        

        for row in rows:
            try:
                t = parse_time(row['HH:mm:ss.fff'])
            except Exception:
                continue

            if max_time is None or t > max_time:
                if prev_time is not None:
                    # compute gap in ms (handles minute/hour rollover simply)
                    gap_ms = (t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
                              - prev_time.hour * 3600 - prev_time.minute * 60 - prev_time.second - prev_time.microsecond / 1e6) * 1000
                    if gap_ms > ((1001/SAMPLING_RATE)):
                        segment_id += 1
                max_time  = t
                prev_time = t           
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
        

        # 2. Identify and Rename Columns to Anatomical Labels
        # The package requires: acc_pa, acc_ml, acc_is
        acc_cols = [c for c in df.columns if 'acc' in c]
        if len(acc_cols) < 3:
            print("Incorrect number of columns.")
            print(f"{len(acc_cols)} columns found instead.")
            
            
        # imu_df = df[acc_cols[:3]].copy()
        # imu_df = imu_df.astype(float) * 9.81
        # imu_df.columns = ['acc_pa', 'acc_ml', 'acc_is']

        # 3. Ground Truth
        if 'test' in DATA_PATH:
            y_true = df['Label'].astype(int).to_numpy()
        else:
            y_true = np.ones(len(df))
        #imu_df['y_true'] = y_true

        if DEBUG == True:
            print('y true is ', y_true)
        diffs = np.diff(y_true)
        diffs_pos = np.where((np.abs(diffs) == 1))
        # correct so far 

        # 4. Run GSD        
        # imu_cols = ['acc_pa', 'acc_ml', 'acc_is']
        detected_bouts = []
        skipped_segments = 0
        y_pred = np.zeros(len(df))
        detected_walking_all = []
        activity_counts_timeline = {}
        for segment, grp in df.groupby('segment', sort=True):
            global_start_idx = grp.index[0]  # offset into the full df

            if len(grp) < MIN_SEGMENT_SAMPLES:
                if DEBUG:
                    print(f"Segment {segment}: skipped ({len(grp)} samples < {MIN_SEGMENT_SAMPLES} min)")
                skipped_segments += 1
                continue
            # FIX: use a local variable, do NOT overwrite imu_df
            seg_imu = grp[acc_cols[:3]].copy().astype(float) * 9.81
            seg_imu.columns = ['acc_pa', 'acc_ml', 'acc_is']
            seg_imu.reset_index(drop=True)
            
            gsd = KheirkhahanGSD(cwb=False, visual=True, switch=diffs_pos[0])
            bout_result, detected_walking, activity_counts = gsd.detect(seg_imu, sampling_rate_hz=SAMPLING_RATE)
            print('detected_walking', detected_walking)
            print('detected_walking size', len(detected_walking))
            print()
            print('activity_counts', activity_counts)
            print('activity_counts size', len(activity_counts))
            print()
            print()
            # dw_arr = np.asarray(detected_walking, dtype=float)
            # g_end = global_start_idx + len(dw_arr)
            # detected_walking_all[global_start_idx:g_end] = dw_arr
            global_start_sec = global_start_idx // SAMPLING_RATE
            for i, val in enumerate(activity_counts):
                activity_counts_timeline[global_start_sec + i] = val

            if hasattr(bout_result, 'gs_list_') and not bout_result.gs_list_.empty:
                for _, bout_row in bout_result.gs_list_.iterrows():
                    # Offset local segment indices to global df indices
                    global_bout_start = int(bout_row['start']) + global_start_idx
                    global_bout_end = int(bout_row['end']) + global_start_idx
                    detected_bouts.append((global_bout_start, global_bout_end))
                    y_pred[global_bout_start:global_bout_end] = 1
            # if result is None:
            #     continue
            # metrics, output_name = result
        if DEBUG: 
            print(f"Total detected bouts across all segments: {len(detected_bouts)}")
        
        if hasattr(bout_result, 'gs_list_') and DEBUG:
            print(f"detected_bouts type: {type(detected_bouts)}")
            # print(f"detected_bouts empty: {detected_bouts.empty}")
            if detected_bouts != []:
                print(f"Detected bouts:\n{detected_bouts}")
                print(f'True switch times:\n{diffs_pos}')
            else:
                print("No walking bouts detected!")
        
        # Plot
        # fig, ax = plt.subplots(figsize=(14, 4))
        # x = np.arange(len(df))

        # # ax.plot(x, y_true,               label='Ground truth',      alpha=0.6, linewidth=1)
        # # ax.plot(x, detected_walking_all, label='Detected walking',  alpha=0.8, linewidth=1)
        # ax.plot(x, y_pred,               label='GSD bout mask',     alpha=0.6, linewidth=1, linestyle='--')

        # for sw in diffs_pos[0]:
        #     ax.axvline(x=sw, color='red', linewidth=0.8, linestyle=':', alpha=0.7)
        # ax.axvline(x=sw, color='red', linewidth=0.8, linestyle=':', alpha=0.7, label='Switch points')  # label once

        # ax.set_xlabel('Sample index')
        # ax.set_ylabel('Walking (1) / Not walking (0)')
        # ax.set_title(f'{file_name}')
        # ax.legend(loc='upper right')
        # ax.set_ylim(-0.1, 1.4)
        # plt.tight_layout()

        # 8. Plot activity counts with gaps
        # total_seconds = len(df) // SAMPLING_RATE
        # time_axis = np.arange(total_seconds)
        # ac_plot = np.full(total_seconds, np.nan)  # nan = gap, won't be plotted

        # for sec_idx, val in activity_counts_timeline.items():
        #     if sec_idx < total_seconds:
        #         ac_plot[sec_idx] = val

        # fig2, ax2 = plt.subplots(figsize=(14, 4))
        # ax2.plot(time_axis, ac_plot, label='Activity count', linewidth=1)

        # for sw in diffs_pos[0]:
        #     ax2.axvline(x=sw / SAMPLING_RATE, color='red', linewidth=0.8, 
        #                 linestyle=':', alpha=0.7)
        # ax2.axvline(x=diffs_pos[0][-1] / SAMPLING_RATE, color='red', linewidth=0.8,
        #             linestyle=':', alpha=0.7, label='Switch points')

        # ax2.set_xlabel('Time (s)')
        # ax2.set_ylabel('Activity count')
        # ax2.set_title(f'Activity counts – {file_name}')
        # ax2.legend(loc='upper right')
        # plt.tight_layout()

        if DEBUG == True:
            print(f"\nPrediction shape: {y_pred.shape}")
            print(f"Prediction sum (detected walking samples): {y_pred.sum()}")
            print(f"Prediction percentage walking: {y_pred.sum() / len(y_pred) * 100:.2f}%")
            
            print(f"--- Comparison ---")
            print(f"True Positives (both predict & true walking): {np.sum((y_pred == 1) & (y_true == 1))}")
            print(f"False Positives (predict walking, true not): {np.sum((y_pred == 1) & (y_true == 0))}")
            print(f"False Negatives (predict not walking, true walking): {np.sum((y_pred == 0) & (y_true == 1))}")
            print(f"True Negatives (both predict & true not walking): {np.sum((y_pred == 0) & (y_true == 0))}")
        
        # 6. Calculate Metrics
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        results.append({
            'Subject': file_name,
            'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1
        })

        print(f"{file_name[:25]:<25} | {acc:.2f}   | {prec:.2f}   | {rec:.2f}   | {f1:.2f}")

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