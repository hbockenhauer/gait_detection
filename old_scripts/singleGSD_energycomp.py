import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
#from multimob.GSD.GSD3 import KheirkhahanGSD
from old_scripts.GSD2_test import HickeyGSD
from Kheirkhahan.GSD3_test import KheirkhahanGSD
import matplotlib.pyplot as plt

# Suppress the DtypeWarning for the walkway columns
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
DATA_PATH = r"C:\Users\orlov\intern\gait_detection\QSense_data_mixed\test8_Hendrik"
# file_name = "s1_1RW.txt"
# file_name = "s2_2LW.txt"
file_name = "s3_3RL.txt"
SAMPLING_RATE = 50 
DEBUG = False; 

if __name__ == "__main__":

    results = []
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.txt')]
    #files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv') and (f.startswith('W') or f.startswith('N'))]
    
    print(f"{'Subject':<25} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 75)

    
    try:
        # 1. Load Data
        df = pd.read_csv(os.path.join(DATA_PATH, file_name), 
                        sep='\t',  # Use whitespace as separator (adjust if needed)
                        low_memory=False)
        
        #### CLIPPING THE FIST 10 SECONDS
        df = df[500:]

        # 2. Identify and Rename Columns to Anatomical Labels
        # The package requires: acc_pa, acc_ml, acc_is
        acc_cols = [c for c in df.columns if 'acc' in c]
        if len(acc_cols) < 3:
            print("Incorrect number of columns")
            print(f"Only {len(acc_cols)} columns found")
            
            
        imu_df = df[acc_cols[:3]].copy()
        imu_df = imu_df * 9.81  
        #print(imu_df)
        imu_df.columns = ['acc_pa', 'acc_ml', 'acc_is'] 

        energy_col = 'Energy'
        energy = df[energy_col]

        
        # 3. Ground Truth
        # if False:
        #     y_true = np.ones(len(df))
        if 'test' in DATA_PATH:
            y_true = np.zeros(len(df))
            y_true = df['Label']
        else:
            y_true = np.ones(len(df))
        
        if DEBUG == True:
            print('y true', y_true)
            print('ones',len(y_true==1))
            print('zeros', len(y_true==0))
        diffs = np.diff(y_true)
        diffs_pos = np.where((np.abs(diffs) == 1))



        ## plotting energy  
        time = np.arange(len(energy)) / SAMPLING_RATE

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time, energy)
        if diffs_pos[0] is not None:
            sr = SAMPLING_RATE
            for sw in diffs_pos[0]: 
                if 0 <= sw < len(energy) * (sr / SAMPLING_RATE):  # bounds check in original samples
                    time_of_change = sw / sr  # convert switch index (original samples) → seconds
                    ax.axvline(x=time_of_change, color='red', linestyle='--', alpha=0.7, linewidth=1)

        ax.set_xlabel("Time(s)") 
        ax.set_ylabel("Energy")
        ax.set_title("Energy from sensors")

        fig.tight_layout()
        fig.show()

        # 4. Run GSD
        
        # HickeyGSD
        # gsd = HickeyGSD(debug=DEBUG, visual=True)
        # detected_bouts = gsd.preprocess(imu_df, sampling_rate_hz=SAMPLING_RATE, target_sampling_rate_hz=SAMPLING_RATE).detect_wrist()
        
        # KheirkhahanGSD
        gsd = KheirkhahanGSD(cwb=False, visual=True, switch=diffs_pos[0])
        detected_bouts = gsd.detect(imu_df, sampling_rate_hz=SAMPLING_RATE)
        
        if hasattr(detected_bouts, 'gs_list_') and DEBUG:
            print(f"gs_list_ type: {type(detected_bouts.gs_list_)}")
            print(f"gs_list_ empty: {detected_bouts.gs_list_.empty}")
            if not detected_bouts.gs_list_.empty:
                print(f"Detected bouts:\n{detected_bouts.gs_list_}")
            else:
                print("No walking bouts detected!")
        
        # 5. Convert Bout List to Binary Mask
        y_pred = np.zeros(len(df))
        if hasattr(detected_bouts, 'gs_list_') and not detected_bouts.gs_list_.empty:
            for idx, row in detected_bouts.gs_list_.iterrows():
                # Ensure indices are within bounds
                start = int(max(0, row['start']))
                end = int(min(len(df), row['end']))
                #print(f"Bout {idx}: start={start}, end={end}, duration={end-start} samples")
                y_pred[start:end] = 1
        # prints
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