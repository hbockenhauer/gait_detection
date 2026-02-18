import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from multimob.GSD.GSD3 import KheirkhahanGSD
from GSD2a import HickeyGSD

# Suppress the DtypeWarning for the walkway columns
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
DATA_PATH = r"C:\Users\orlov\intern\gait_detection\QSense_data_edge"
SAMPLING_RATE = 50 
DEBUG = False; 
GAIT_CLASSES = {'Walking', 'Stairs'}


def process_weargait():
    results = []
    
    print(f"{'Subject':<30} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 75)

    for folder in os.listdir(DATA_PATH):
        if not os.path.isdir(os.path.join(DATA_PATH, folder)):
            continue

        files = [
            os.path.join(DATA_PATH, folder, 's1_1RW.txt'),  # Right wrist
            os.path.join(DATA_PATH, folder, 's2_2LW.txt')   # Left wrist 
        ]
        

        for file in files:
            if not os.path.exists(file):
                continue
            wrist = "right" if "1RW" in file else "left"

            try:
                # 1. Load Data
                df = pd.read_csv(file, 
                            sep='\t',  # Use whitespace as separator (adjust if needed)
                            low_memory=False)

                # 2. Identify and Rename Columns to Anatomical Labels
                acc_cols = [c for c in df.columns if 'acc' in c]
                #print(f"the acc_cols is {acc_cols} ")
                if len(acc_cols) < 3:
                    continue
                    
                    
                imu_df = df[acc_cols[:3]].copy()
                imu_df = imu_df * 9.81  
                imu_df.columns = ['acc_pa', 'acc_ml', 'acc_is']  # <--- The key fix
                
                # 3. Ground Truth
                activity = folder.split('_')[0]
                if activity in GAIT_CLASSES:
                    y_true = np.ones(len(imu_df), dtype=int)
                else:
                    y_true = np.zeros(len(imu_df), dtype=int)

                # 4. Run Kheirkhahan GSD
                #gsd = KheirkhahanGSD()
                #detected_bouts = gsd.detect(imu_df, sampling_rate_hz=SAMPLING_RATE)
                # Note: KheirkhahanGSD in this package takes the DataFrame directly
                
                # HickeyGSD
                gsd = HickeyGSD(debug=DEBUG)
                detected_bouts = gsd.preprocess(imu_df, sampling_rate_hz=SAMPLING_RATE, target_sampling_rate_hz=SAMPLING_RATE).detect_wrist()
                                
                # debug prints
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

                # Extract folder name and condition
                folder_name = os.path.basename(os.path.dirname(file))
                display_label = f"{folder_name}/{os.path.basename(file)}"

                # Extract condition from folder name (e.g., "c1", "c2", etc.)
                folder_lower = folder_name.lower()
                if 'pockets' in folder_lower:
                    condition = 'pockets'
                elif 'rail' in folder_lower:
                    condition = 'rail'
                elif 'phone' in folder_lower:
                    condition = 'phone'
                elif 'limp' in folder_lower:
                    condition = 'limp'
                elif 'armfixed' in folder_lower:
                    condition = 'armfixed'
                elif 'free' in folder_lower:
                    condition = 'free'
                else:
                    condition = folder_name  # Use folder name if no condition pattern found

                results.append({
                    'Subject': display_label,
                    'Folder': folder_name,
                    'Condition': condition,
                    'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1
                })

                print(f"{display_label[:30]:<30} | {acc:.2f}   | {prec:.2f}   | {rec:.2f}   | {f1:.2f}")

            except Exception as e:
                folder_name = os.path.basename(os.path.dirname(file_path))
                display_label = f"{folder_name}/{file_name}"
                print(f"{display_label[:30]:<30} | ERROR: {str(e)}")

    # Final Summary
    if results:
        res_df = pd.DataFrame(results)
        print("-" * 75)
        
        # Separate results by wrist type (RW vs LW)
        rw_files = res_df[res_df['Subject'].str.contains('RW', case=False)]
        lw_files = res_df[res_df['Subject'].str.contains('LW', case=False)]
        
        # Print RW (Right Wrist) average
        if not rw_files.empty:
            print(f"{'AVERAGE (RW - Right)':<30} | {rw_files['Accuracy'].mean():.2f}   | {rw_files['Precision'].mean():.2f}   | {rw_files['Recall'].mean():.2f}   | {rw_files['F1'].mean():.2f}")
        
        # Print LW (Left Wrist) average
        if not lw_files.empty:
            print(f"{'AVERAGE (LW - Left)':<30} | {lw_files['Accuracy'].mean():.2f}   | {lw_files['Precision'].mean():.2f}   | {lw_files['Recall'].mean():.2f}   | {lw_files['F1'].mean():.2f}")
        
            # Group by condition and calculate averages
        unique_conditions = res_df['Condition'].unique()
        
        for condition in sorted(unique_conditions):
            condition_data = res_df[res_df['Condition'] == condition]
            avg_label = f"AV({condition})"
            print(f"{avg_label:<30} | {condition_data['Accuracy'].mean():.2f}   | {condition_data['Precision'].mean():.2f}   | {condition_data['Recall'].mean():.2f}   | {condition_data['F1'].mean():.2f}")
 
        
        # Print overall average
        print("-" * 75)
        print(f"{'AVERAGE (Overall)':<30} | {res_df['Accuracy'].mean():.2f}   | {res_df['Precision'].mean():.2f}   | {res_df['Recall'].mean():.2f}   | {res_df['F1'].mean():.2f}")
        
        res_df.to_csv('HickeyGSD_Results.csv', index=False)

if __name__ == "__main__":
    process_weargait()