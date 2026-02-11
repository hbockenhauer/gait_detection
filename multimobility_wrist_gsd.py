import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from multimob.GSD.GSD3 import KheirkhahanGSD
from GSD2a import HickeyGSD

# Suppress the DtypeWarning for the walkway columns
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
dataset = "QSense"
group = "healthy"
match dataset:
    case "WearGait":
        if group == "healthy":
            DATA_PATH = r'C:\Users\orlov\intern\gait_detection\WearGait-Ctrl'
        else:
            DATA_PATH = r'C:\Users\orlov\intern\gait_detection\WearGait-PD'
    case "HAR":
        DATA_PATH = r"C:\Users\orlov\intern\gait_detection\HAR_data_acc"
    case "HMP":
        DATA_PATH = r"C:\Users\orlov\intern\gait_detection\HMP_Dataset\Walk"
    case "QSense":
        DATA_PATH = r"C:\Users\orlov\intern\gait_detection\QSense_data"

SAMPLING_RATE = 100 

def process_weargait():
    results = []
    if dataset == "QSense":
        files = [f for f in os.listdir(DATA_PATH) if f.endswith('.txt')]
    else:
        files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv') and (f.startswith('W') or f.startswith('N'))]
    
    print(f"Processing {len(files)} files using HickeyGSD...")
    print(f"{'Subject':<25} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 75)

    for file_name in files:
        try:
            # 1. Load Data
            if dataset == "QSense":
                df = pd.read_csv(os.path.join(DATA_PATH, file_name), 
                           sep='\t',  # Use whitespace as separator (adjust if needed)
                           low_memory=False)
            else:
                df = pd.read_csv(os.path.join(DATA_PATH, file_name), low_memory=False)

            # 2. Identify and Rename Columns to Anatomical Labels
            # The package requires: acc_pa, acc_ml, acc_is
            #print("the df is ", df.columns)
            if dataset == "WearGait":
                acc_cols = [c for c in df.columns if 'Acc' in c]
                if len(acc_cols) < 3:
                    continue
            elif dataset in ["HAR", "QSense"]:
                #print(df.columns)
                acc_cols = [c for c in df.columns if 'acc' in c]
                #print(f"the acc_cols is {acc_cols} ")
                if len(acc_cols) < 3:
                    continue
                
                
            imu_df = df[acc_cols[:3]].copy()
            imu_df.columns = ['acc_pa', 'acc_ml', 'acc_is']  # <--- The key fix
            
            # 3. Ground Truth
            if dataset == "QSense":
                y_true = np.ones(len(df))
            else:
                label_col = [c for c in df.columns if any(word in c.lower() for word in ['activity', 'event', 'label', 'gt'])][0]
                y_true = df[label_col].str.contains('walk|gait|free|stair', case=False, na=False).astype(int).values

            # 4. Run Kheirkhahan GSD
            #gsd = KheirkhahanGSD()
            gsd = HickeyGSD()
            # Note: KheirkhahanGSD in this package takes the DataFrame directly
            # HickeyGSD
            detected_bouts = gsd.preprocess(imu_df, sampling_rate_hz=100).detect_wrist()
            # KheirkhahanGSD
            #detected_bouts = gsd.detect(imu_df, sampling_rate_hz=SAMPLING_RATE)
            
            # 5. Convert Bout List to Binary Mask
            y_pred = np.zeros(len(df))
            if hasattr(detected_bouts, 'gs_list_') and not detected_bouts.gs_list_.empty:
                for _, row in detected_bouts.gs_list_.iterrows():
                    # Ensure indices are within bounds
                    start = int(max(0, row['start']))
                    end = int(min(len(df), row['end']))
                    y_pred[start:end] = 1
            
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
        print(f"{'AVERAGE':<25} | {res_df['Accuracy'].mean():.2f}   | {res_df['Precision'].mean():.2f}   | {res_df['Recall'].mean():.2f}   | {res_df['F1'].mean():.2f}")
        res_df.to_csv('HickeyGSD_Results.csv', index=False)

if __name__ == "__main__":
    process_weargait()