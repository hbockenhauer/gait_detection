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
import matplotlib.ticker as mticker

# Suppress the DtypeWarning for the walkway columns
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
DATA_PATH = r"C:\Users\orlov\intern\gait_detection\QSense_data_edge\temp\Walking_crutches_Tanya"
file_name = "s2_2LW.txt" #"s1_1RW.txt"
SAMPLING_RATE = 50 
DEBUG = True
MIN_SEGMENT_SAMPLES = 9*SAMPLING_RATE  # requirement for 9s window 
PLOT = True

def parse_time(t_str):
    h, m, s_ms = t_str.strip().split(':')
    s, ms = s_ms.split('.')
    return time(int(h), int(m), int(s), int(ms) * 1000)

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
    except Exception as e:
        print(f"{file_name[:25]:<25} | ERROR: {str(e)}")
    
    return df

def run_gsd_on_segment(grp) : 
    global_start_idx = grp.index[0]  # offset into the full df
    # fing the acceleration columns 
    acc_cols = [c for c in grp.columns if 'acc' in c]
    if len(acc_cols) < 3:
        print("Incorrect number of columns.")
        print(f"{len(acc_cols)} columns found instead.")
    # rename the columns and run the gsd on them
    seg_imu = grp[acc_cols[:3]].copy().astype(float) * 9.81
    seg_imu.columns = ['acc_pa', 'acc_ml', 'acc_is']
    seg_imu.reset_index(drop=True)
    
    gsd = KheirkhahanGSD(cwb=False)
    bout_result = gsd.detect(seg_imu, sampling_rate_hz=SAMPLING_RATE)
    activity_counts = gsd.get_activity(seg_imu, sampling_rate_hz=SAMPLING_RATE)

    return bout_result, activity_counts, global_start_idx


if __name__ == "__main__":

    results = []
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.txt')]
    #files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv') and (f.startswith('W') or f.startswith('N'))]
    
    print(f"{'Subject':<25} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 75)

    
    try:
        # 1. Load Data
        df = load_segmented(DATA_PATH, file_name)

        # 2. Identify and Rename Columns to Anatomical Labels
        # The package requires: acc_pa, acc_ml, acc_is
        # acc_cols = [c for c in df.columns if 'acc' in c]
        # if len(acc_cols) < 3:
        #     print("Incorrect number of columns.")
        #     print(f"{len(acc_cols)} columns found instead.")

        # 3. Ground Truth
        if 'test' in DATA_PATH:
            y_true = df['Label'].astype(int).to_numpy()
        else:
            y_true = np.ones(len(df))

        diffs = np.diff(y_true)
        diffs_pos = np.where((np.abs(diffs) == 1))

        # 4. Run GSD        
        detected_bouts = []
        y_pred = np.zeros(len(df))
        activity_counts_timeline = {}
        skipped_seg = 0
        for segment, grp in df.groupby('segment', sort=True):
            # global_start_idx = grp.index[0]  # offset into the full df
            # print('grp',grp.)
            print("segment", segment)
            
            if len(grp) < MIN_SEGMENT_SAMPLES:
                y_pred[grp.index] = np.nan
                skipped_seg += 1
                continue

            bout_result, activity_counts, global_start_idx = run_gsd_on_segment(grp)
            print("global_start_idx", global_start_idx)
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
       
        if hasattr(bout_result, 'gs_list_') and DEBUG:
            print(f"Total detected bouts across all segments: {len(detected_bouts)}")
            print(f"Skipped {skipped_seg} segments.")
            if detected_bouts != []:
                print(f"Detected bouts:\n{detected_bouts}")
                print(f'True switch times:\n{diffs_pos}')
            else:
                print("No walking bouts detected!")
        
        # Parse timestamps from df into timedeltas
        time_series = pd.to_timedelta(df['HH:mm:ss.fff'].str.strip())
        # Convert to total seconds (float) for plotting
        time_per_second_sec = time_series.iloc[::SAMPLING_RATE].reset_index(drop=True).dt.total_seconds()

        total_seconds = len(time_per_second_sec)
        ac_plot = np.full(total_seconds, np.nan)
        for sec_idx, val in activity_counts_timeline.items():
            if sec_idx < total_seconds:
                ac_plot[sec_idx] = val

        all_segment_first_rows = df.groupby('segment').nth(0).index
        jump_row_indices = all_segment_first_rows[1:]
        jump_times_sec = [time_series.iloc[idx].total_seconds() for idx in jump_row_indices]


        time_all_sec = time_series.dt.total_seconds() # seconds from midnight, accurate to 2 decimals
        if PLOT == True: 
            fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

            # ── Top: y_pred and y_true ────────────────────────────────────────────────
            ax1.fill_between(time_all_sec, 0, 1, where=(y_true == 1),
                            alpha=0.3, color='green', label='Ground truth (walking)')
            ax1.plot(time_all_sec, y_pred, label='y_pred (GSD)', alpha=0.8, 
                    linewidth=1, color='steelblue')

            for i, jt in enumerate(jump_times_sec):
                ax1.axvline(x=jt, color='orange', linewidth=1.0, linestyle='--', alpha=0.8,
                            label='Time gap' if i == 0 else None)

            ax1.set_ylabel('Walking (1) / Not (0)')
            ax1.set_title(f'{file_name}')
            ax1.legend(loc='upper left')
            ax1.set_ylim(-0.1, 1.4)

            # ── Bottom: activity counts ───────────────────────────────────────────────
            ax2.fill_between(time_all_sec, 0, 1, where=(y_true == 1),
                            alpha=0.2, color='green', transform=ax2.get_xaxis_transform(),
                            label='Ground truth (walking)')
            ax2.plot(time_per_second_sec, ac_plot, label='Activity count', 
                    linewidth=1, color='steelblue')

            for i, jt in enumerate(jump_times_sec):
                ax2.axvline(x=jt, color='orange', linewidth=1.0, linestyle='--', alpha=0.8,
                            label='Time gap' if i == 0 else None)

            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Activity count')
            ax2.legend(loc='upper left')

            # Format x-axis as HH:MM:SS
            ax2.xaxis.set_major_formatter(mticker.FuncFormatter(
                lambda x, _: f"{int(x//3600):02d}:{int((x%3600)//60):02d}:{int(x%60):02d}"
            ))
            fig2.autofmt_xdate()
            plt.tight_layout()




        if DEBUG == True:
            print(f"\nPrediction shape: {y_pred.shape}")
            
            print(f"--- Comparison ---")
            print(f"True Positives (both predict & true walking): {np.sum((y_pred == 1) & (y_true == 1))}")
            print(f"False Positives (predict walking, true not): {np.sum((y_pred == 1) & (y_true == 0))}")
            print(f"False Negatives (predict not walking, true walking): {np.sum((y_pred == 0) & (y_true == 1))}")
            print(f"True Negatives (both predict & true not walking): {np.sum((y_pred == 0) & (y_true == 0))}")
        
        # 6. Calculate Metrics
        valid_mask = ~np.isnan(y_pred)
        acc  = accuracy_score(y_true[valid_mask], y_pred[valid_mask])
        prec = precision_score(y_true[valid_mask], y_pred[valid_mask], zero_division=0)
        rec  = recall_score(y_true[valid_mask], y_pred[valid_mask], zero_division=0)
        f1   = f1_score(y_true[valid_mask], y_pred[valid_mask], zero_division=0)

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