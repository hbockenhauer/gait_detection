import os
import pandas as pd
import numpy as np
import glob
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from multimob.GSD.GSD3 import KheirkhahanGSD
from GSD2a import HickeyGSD

# Suppress the DtypeWarning for the walkway columns
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
DATA_PATH = r"C:\Users\orlov\intern\gait_detection\Qsense_data\QSense_data_edge"
SAMPLING_RATE = 50 
DEBUG = False; 


def merge_sensor_files(input_folder, output_file_rw, output_file_lw):
    """
    Merge multiple txt files containing sensor data, keeping only acceleration columns
    and setting classification based on folder name. Creates separate files for RW and LW.
    
    Parameters:
    -----------
    input_folder : str
        Path to root folder containing subfolders with txt files
    output_file_rw : str
        Path for the output merged file for RW (right wrist) files
    output_file_lw : str
        Path for the output merged file for LW (left wrist) files
    
    Returns:
    --------
    tuple : (rw_dataframe, lw_dataframe)
    """
    
    # Get all txt files recursively in the folder structure
    file_pattern = os.path.join(input_folder, "**", "*.txt")
    files = glob.glob(file_pattern, recursive=True)
    
    if not files:
        print(f"No txt files found in {input_folder}")
        return None, None
    
    print(f"Found {len(files)} files to merge")
    
    rw_data = []
    lw_data = []
    
    for file_path in files:
        filename = os.path.basename(file_path)
        folder_path = os.path.dirname(file_path)
        folder_name = os.path.basename(folder_path)
        
        print(f"\nProcessing: {folder_name}/{filename}")
        
        # Read the file
        try:
            df = pd.read_csv(file_path, sep='\t')
        except Exception as e:
            print(f"  -> ERROR reading file: {e}")
            continue
        
        # Determine classification based on FOLDER name
        folder_lower = folder_name.lower()
        
        # Extract case from folder name
        case = 'unknown'
        if 'stairs' in folder_lower:
            case = 'stairs'
            classification = 'walk'
        elif 'pockets' in folder_lower:
            case = 'pockets'
            classification = 'walk'
        elif 'rail' in folder_lower:
            case = 'rail'
            classification = 'walk'
        elif 'phone' in folder_lower:
            case = 'phone'
            classification = 'walk'
        elif 'limp' in folder_lower:
            case = 'limp'
            classification = 'walk'
        elif 'armfixed' in folder_lower or 'arm_fixed' in folder_lower:
            case = 'armfixed'
            classification = 'walk'
        elif 'walk' in folder_lower:
            case = 'walk'
            classification = 'walk'
        elif 'free' in folder_lower:
            case = 'free'
            classification = 'non-walk'
        else:
            # Default to non-walk for other folders
            case = folder_name
            classification = 'non-walk'
        
        # Keep only the columns we need: date, time, acc values, and classification
        columns_to_keep = ['yyyy-MM-dd', 'HH:mm:ss.fff', 'accX', 'accY', 'accZ']
        
        # Check if all required columns exist
        missing_cols = [col for col in columns_to_keep if col not in df.columns]
        if missing_cols:
            print(f"  -> WARNING: Missing columns {missing_cols}, skipping file")
            continue
            
        df_filtered = df[columns_to_keep].copy()
        
        # Add/update the classification column
        df_filtered['Classification'] = classification
        
        # Add case column
        df_filtered['Case'] = case
        
        # Add folder name for reference
        df_filtered['Folder'] = folder_name
        
        # Separate by wrist type (RW or LW) based on FILENAME
        filename_lower = filename.lower()
        if 'rw' in filename_lower:
            rw_data.append(df_filtered)
        elif 'lw' in filename_lower:
            lw_data.append(df_filtered)
        else:
            print(f'  -> WARNING: File does not contain RW or LW in name, skipping')
    
    # Process RW files
    if rw_data:
        rw_result = pd.concat(rw_data, ignore_index=True)
        rw_result.to_csv(output_file_rw, sep='\t', index=False)
        if DEBUG == True: 
            print(f"\n{'='*75}")
            print(f"=== RW (Right Wrist) Merge Complete ===")
            print(f"Total rows: {len(rw_result)}")
            print(f"Total files: {len(rw_data)}")
            print(f"Output saved to: {output_file_rw}")
            print(f"\nClassification distribution:")
            print(rw_result['Classification'].value_counts())
            print(f"\nCase distribution:")
            print(rw_result['Case'].value_counts())
            print(f"\nFolder distribution:")
            print(rw_result['Folder'].value_counts())
    else:
        print("\nNo RW files found!")
        rw_result = None
    
    # Process LW files
    if lw_data:
        lw_result = pd.concat(lw_data, ignore_index=True)
        lw_result.to_csv(output_file_lw, sep='\t', index=False)
        if DEBUG == True: 
            print(f"\n{'='*75}")
            print(f"=== LW (Left Wrist) Merge Complete ===")
            print(f"Total rows: {len(lw_result)}")
            print(f"Total files: {len(lw_data)}")
            print(f"Output saved to: {output_file_lw}")
            print(f"\nClassification distribution:")
            print(lw_result['Classification'].value_counts())
            print(f"\nCase distribution:")
            print(lw_result['Case'].value_counts())
            print(f"\nFolder distribution:")
            print(lw_result['Folder'].value_counts())
    else:
        print("\nNo LW files found!")
        lw_result = None
    
    return rw_result, lw_result



def process_weargait():
    results = []
    
    print(f"{'File':<30} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
    print("-" * 75)
    
    input_folder = DATA_PATH
    output_file_rw = "merged_data_RW.txt"
    output_file_lw = "merged_data_LW.txt"
    
    # Merge the sensor files into separate RW and LW files
    rw_df, lw_df = merge_sensor_files(input_folder, output_file_rw, output_file_lw)
    
    # Process each file separately
    output_files = []
    if rw_df is not None:
        output_files.append((output_file_rw, "RW"))
    if lw_df is not None:
        output_files.append((output_file_lw, "LW"))
    
    if not output_files:
        print("No data to process")
        return
    
    for output_file, wrist_type in output_files:
        print(f"\n{'='*75}")
        print(f"Processing {wrist_type} (Right Wrist) data from {output_file}" if wrist_type == "RW" 
              else f"Processing {wrist_type} (Left Wrist) data from {output_file}")
        print(f"{'='*75}")
        
        try:
            # 1. Load Data
            df = pd.read_csv(output_file, 
                            sep='\t',
                            low_memory=False)
            
            # Get unique cases in this dataset
            unique_cases = df['Case'].unique()
            print(f"Found {len(unique_cases)} unique cases: {', '.join(unique_cases)}")
            
            # Process each case separately
            for case in unique_cases:
                print(f"\n--- Processing Case: {case} ---")
                
                # Filter data for this case
                case_df = df[df['Case'] == case].copy()
                
                # 2. Identify and Rename Columns to Anatomical Labels
                acc_cols = [c for c in case_df.columns if 'acc' in c.lower()]
                
                if len(acc_cols) < 3:
                    print(f"Error: Found only {len(acc_cols)} acceleration columns, need at least 3")
                    continue
                
                imu_df = case_df[acc_cols[:3]].copy()
                imu_df = imu_df * 9.81  # Convert to m/s^2
                imu_df.columns = ['acc_pa', 'acc_ml', 'acc_is']
                
                # 3. Ground Truth - Get from Classification column
                y_true = (case_df['Classification'] == 'walk').astype(int).values
                
                # 4. Run Hickey GSD
                gsd = HickeyGSD(debug=DEBUG)
                detected_bouts = gsd.preprocess(imu_df, 
                                               sampling_rate_hz=SAMPLING_RATE, 
                                               target_sampling_rate_hz=SAMPLING_RATE).detect_wrist()
                
                # Debug prints
                if hasattr(detected_bouts, 'gs_list_') and DEBUG:
                    print(f"gs_list_ type: {type(detected_bouts.gs_list_)}")
                    print(f"gs_list_ empty: {detected_bouts.gs_list_.empty}")
                    if not detected_bouts.gs_list_.empty:
                        print(f"Detected bouts:\n{detected_bouts.gs_list_}")
                    else:
                        print("No walking bouts detected!")
                
                # 5. Convert Bout List to Binary Mask
                y_pred = np.zeros(len(case_df))
                if hasattr(detected_bouts, 'gs_list_') and not detected_bouts.gs_list_.empty:
                    for idx, row in detected_bouts.gs_list_.iterrows():
                        start = int(max(0, row['start']))
                        end = int(min(len(case_df), row['end']))
                        y_pred[start:end] = 1
                
                # Debug prints
                if DEBUG:
                    print(f"\nPrediction shape: {y_pred.shape}")
                    print(f"Prediction sum (detected walking samples): {y_pred.sum()}")
                    print(f"Prediction percentage walking: {y_pred.sum() / len(y_pred) * 100:.2f}%")
                    print(f"Ground truth percentage walking: {y_true.sum() / len(y_true) * 100:.2f}%")
                    
                    print(f"--- Comparison ---")
                    print(f"True Positives: {np.sum((y_pred == 1) & (y_true == 1))}")
                    print(f"False Positives: {np.sum((y_pred == 1) & (y_true == 0))}")
                    print(f"False Negatives: {np.sum((y_pred == 0) & (y_true == 1))}")
                    print(f"True Negatives: {np.sum((y_pred == 0) & (y_true == 0))}")
                
                # 6. Calculate Metrics
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                results.append({
                    'File': output_file,
                    'Wrist': wrist_type,
                    'Case': case,
                    'Accuracy': acc,
                    'Precision': prec,
                    'Recall': rec,
                    'F1': f1,
                    'Samples': len(case_df)
                })
                
                case_display = f"{wrist_type}-{case}"
                print(f"{case_display[:30]:<30} | {acc:.2f}   | {prec:.2f}   | {rec:.2f}   | {f1:.2f}")
        
        except Exception as e:
            print(f"ERROR processing {output_file}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Final Summary
    if results:

        res_df = pd.DataFrame(results)
        print("\n" + "=" * 75)
        print("SUMMARY - BY CASE")
        print("=" * 75)
        print(f"{'Case/Wrist':<30} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6}")
        print("-" * 75)
        
        # Get unique cases
        unique_cases = res_df['Case'].unique()
        
        # Print results for each case
        for case in sorted(unique_cases):
            case_data = res_df[res_df['Case'] == case]
            case_label = f"Case: {case}"
            print(f"{case_label:<30} | {case_data['Accuracy'].mean():.2f}   | {case_data['Precision'].mean():.2f}   | {case_data['Recall'].mean():.2f}   | {case_data['F1'].mean():.2f}")
            
            # Print breakdown by wrist for this case
            for wrist_type in ['RW', 'LW']:
                wrist_case_data = case_data[case_data['Wrist'] == wrist_type]
                if not wrist_case_data.empty:
                    wrist_label = f"  {wrist_type} ({case})"
                    print(f"{wrist_label:<30} | {wrist_case_data['Accuracy'].mean():.2f}   | {wrist_case_data['Precision'].mean():.2f}   | {wrist_case_data['Recall'].mean():.2f}   | {wrist_case_data['F1'].mean():.2f}")
        
        print("\n" + "=" * 75)
        print("SUMMARY - BY WRIST")
        print("=" * 75)
        
        # Print results for each wrist separately
        for wrist_type in ['RW', 'LW']:
            wrist_data = res_df[res_df['Wrist'] == wrist_type]
            if not wrist_data.empty:
                wrist_name = "Right Wrist" if wrist_type == "RW" else "Left Wrist"
                print(f"{f'AVERAGE ({wrist_type} - {wrist_name})':<30} | {wrist_data['Accuracy'].mean():.2f}   | {wrist_data['Precision'].mean():.2f}   | {wrist_data['Recall'].mean():.2f}   | {wrist_data['F1'].mean():.2f}")
        
        # Print overall average
        print("-" * 75)
        print(f"{'AVERAGE (Overall)':<30} | {res_df['Accuracy'].mean():.2f}   | {res_df['Precision'].mean():.2f}   | {res_df['Recall'].mean():.2f}   | {res_df['F1'].mean():.2f}")
        
        res_df.to_csv('HickeyGSD_Results.csv', index=False)
        print(f"\nResults saved to HickeyGSD_Results.csv")

if __name__ == "__main__":
    process_weargait()