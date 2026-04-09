import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from GSD3_test import KheirkhahanGSD
from multimob.GSD.GSD4 import MacLeanGSD
from multimob.GSD.GSD5 import KerenGSD
from GSD2a import HickeyGSD
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
DATA_PATHS = [
    # r"C:\Users\orlov\intern\gait_detection\QSense_data",
    # r"C:\Users\orlov\intern\gait_detection\QSense_data_edge",
    # r"C:\Users\orlov\intern\gait_detection\QSense_data_mixed", 
    r"C:\Users\orlov\intern\gait_detection\QSense_data_clinic"
]
GSD_n = 3
SAMPLING_RATE = 50 
DEBUG = False; 
GAIT_CLASSES = {'walking', 'stairs'}
CONDITION_KEYWORDS = ['pockets', 'phone', 'rail', 'free', 'crutches', 'walker', 'cane', 'mixed']
SAVE_RESULTS = True 
PRINT_STATS = True 
PLOT_SAVE_FOLDER = r"C:\Users\orlov\intern\gait_detection\Plots\Leg_comp"
PLOT_PRED = True

def extract_condition(folder_name: str) -> str:
    folder_lower = folder_name.lower()
    for kw in CONDITION_KEYWORDS:
        if kw in folder_lower:
            return kw
    return 'normal'

def is_gait(folder_name: str, df:pd.DataFrame = None) -> int:
    if 'test' in folder_name.lower() or 'sub' in folder_name.lower():
        return df['Label']
    else:
        activity = folder_name.split('_')[0].lower()
        return 1 if activity in GAIT_CLASSES else 0

def load_file(filepath: str) -> pd.DataFrame | None:
    """Read one sensor file, scale acc to m/s², rename columns. Returns None on failure."""
    try:
        df = pd.read_csv(filepath, sep=None, engine="python")
        df = df.reset_index(drop=True)

        # clip the first 10 seconds 
        if "s3_3RL.txt" in filepath:
            df = df
        else:
            df = df[500:]

        acc_cols = [c for c in df.columns if 'acc' in c.lower()]
        if len(acc_cols) < 3:
            print(f"  [SKIP] Not enough acc columns in {filepath} (found {len(acc_cols)})")
            return None




        imu_df = df[acc_cols[:3]].copy()
        imu_df = imu_df * 9.81          # convert g → m/s²
        imu_df.columns = ['acc_pa', 'acc_ml', 'acc_is']

        return imu_df, df
    except Exception as e:
        print(f"  [ERROR] Failed to load {filepath}: {e}")
        return None


def merge_all(data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk data_path, load every s1_1RW.txt and s2_2LW.txt, attach metadata,
    and return (rw_merged, lw_merged).

    Each returned DataFrame has columns:
        subject | condition | y_true | acc_pa | acc_ml | acc_is
    """
    rw_chunks: list[pd.DataFrame] = []
    lw_chunks: list[pd.DataFrame] = []
    rl_chunks: list[pd.DataFrame] = []

    if PRINT_STATS == True:
        print(f"Scanning: {data_path}\n")
        print(f"{'Folder':<35} | {'Cond':<10} | {'RW rows':>8} | {'LW rows':>8}")
        print("-" * 80)

    for folder in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue

        #y_label   = 
        condition = extract_condition(folder)
        subject   = folder

        rw_path = os.path.join(folder_path, 's1_1RW.txt')
        lw_path = os.path.join(folder_path, 's2_2LW.txt')
        rl_path = os.path.join(folder_path, 's3_3RL.txt')


        rw_rows = lw_rows = rl_rows = 0

        if os.path.exists(rw_path):
            rw_df, df_full = load_file(rw_path)
            if rw_df is not None:
                rw_df['y_true']    = is_gait(folder, df_full)
                rw_df['condition'] = condition
                rw_df['subject']   = subject
                rw_chunks.append(rw_df)
                rw_rows = len(rw_df)

        if os.path.exists(lw_path):
            lw_df, df_full = load_file(lw_path)
            if lw_df is not None:
                lw_df['y_true']    = is_gait(folder, df_full)
                lw_df['condition'] = condition
                lw_df['subject']   = subject
                lw_chunks.append(lw_df)
                lw_rows = len(lw_df)

        if os.path.exists(rl_path):
            rl_df, df_full = load_file(rl_path)
            if rl_df is not None:
                rl_df['y_true']    = is_gait(folder, df_full)
                rl_df['condition'] = condition
                rl_df['subject']   = subject
                rl_df['energy'] = df_full['Energy']
                rl_df['classification']   = df_full['Classification']
                rl_chunks.append(rl_df)
                rl_rows = len(rl_df)        

        
        if (rw_rows > 0 or lw_rows > 0 or rl_rows > 0) and PRINT_STATS:
            print(f"{folder[:35]:<35} | {condition:<10} | {rw_rows:>8} | {lw_rows:>8}")

    if PRINT_STATS == True:
        print("-" * 80)

    col_order = ['subject', 'condition', 'y_true', 'acc_pa', 'acc_ml', 'acc_is']
    col_order_leg = ['subject', 'condition', 'y_true', 'acc_pa', 'acc_ml', 'acc_is', 'energy', 'classification']

    rw_merged = (pd.concat(rw_chunks, ignore_index=True)[col_order]
                 if rw_chunks else pd.DataFrame(columns=col_order))
    lw_merged = (pd.concat(lw_chunks, ignore_index=True)[col_order]
                 if lw_chunks else pd.DataFrame(columns=col_order))
    rl_merged = (pd.concat(rl_chunks, ignore_index=True)[col_order_leg]
                 if rl_chunks else pd.DataFrame(columns=col_order_leg))

    return rw_merged, lw_merged, rl_merged

def process_leg(rl_merged: pd.DataFrame, save_results: bool = False):
            
    results = []

    print(f"\n{'Subject':<35} | "
          f"{'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
    print("-" * 90)

    if rl_merged.empty:
        print(f"[rl_merged] No data found.")
        return

    imu_cols           = ['acc_pa', 'acc_ml', 'acc_is']
    classification_col = 'classification'
    energy_col         = 'energy'

    for subject, grp in rl_merged.groupby('subject', sort=True):
        imu_df         = grp[imu_cols].reset_index(drop=True)
        y_true         = grp['y_true'].to_numpy()
        condition      = grp['condition'].iloc[0]
        label          = f"{subject}"
        energy         = grp[energy_col].to_numpy()
        classification = grp[classification_col].to_numpy()

        if DEBUG:
            print('group', grp)
            print('classification is', classification)

        # classification == 3 means walking; produce a per-sample binary mask
        y_pred = (classification == 3).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)

        if PRINT_STATS:
            print(f"{label[:35]:<35} | "
                  f"{acc:.4f}   | {prec:.4f}   | "
                  f"{rec:.4f}   | {f1:.4f}")

        results.append({
            'Subject':   label,
            'Condition': condition,
            'Accuracy':  acc,
            'Precision': prec,
            'Recall':    rec,
            'F1':        f1,
            'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
        })

        # plot 
        if PLOT_PRED == True: 
            time = np.arange(len(y_pred)) / SAMPLING_RATE
            fig, ax = plt.subplots(figsize=(12, 4))

            # Fill between for clear overlap visualization
            ax.fill_between(time, y_true, alpha=0.5, color='steelblue', label='False Negative', step='post')
            ax.fill_between(time, y_pred, alpha=0.5, color='tomato',    label='False Positive', step='post')

            # Highlight agreement region
            agreement = (np.array(y_true) == 1) & (np.array(y_pred) == 1)
            ax.fill_between(time, agreement, color='mediumseagreen', alpha=0.6, step='post', label='True Positive')

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Walking detected')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['No', 'Yes'])
            ax.set_ylim(-0.05, 1.3)
            ax.legend(loc='upper right')
            ax.set_title(label)
            plt.tight_layout()
            
            out_path = os.path.join(PLOT_SAVE_FOLDER, f"{subject}_leg_pred.png")
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved -> {out_path}")


    if results:
        res_df = pd.DataFrame(results)
        tp = res_df['TP'].sum()
        fp = res_df['FP'].sum()
        fn = res_df['FN'].sum()
        tn = res_df['TN'].sum()
        total = tp + fp + fn + tn

        acc_av  = (tp + tn) / total                                                          if total > 0             else 0.0
        prec_av = tp / (tp + fp)                                                             if (tp + fp) > 0         else 0.0
        rec_av  = tp / (tp + fn)                                                             if (tp + fn) > 0         else 0.0
        f1_av   = 2 * prec_av * rec_av / (prec_av + rec_av) if (prec_av + rec_av) > 0 else 0.0

        print("-" * 90)
        print(f"{'AVERAGE (Leg Overall)':<35} | "
              f"{acc_av:.4f}   | {prec_av:.4f}   | "
              f"{rec_av:.4f}   | {f1_av:.4f}")
        row_avg = {#'row_type':  'avg_overall',
            'Subject':   'AVERAGE (Overall)',
            'Condition': condition,
            'Accuracy':     acc_av,
            'Precision':    prec_av,
            'Recall':       rec_av,
            'F1':           f1_av,
            'TP':           res_df['TP'].sum(),
            "FP":           res_df['FP'].sum(),
            'FN':           res_df['FN'].sum(),
            'TN':           res_df['TN'].sum()
            }
            

        avg_rows: list[dict] = []
        avg_rows.append(row_avg)
        
        avg_df    = pd.DataFrame(avg_rows)
        blank_row = pd.DataFrame([{c: '' for c in res_df.columns}])

        csv_df = pd.concat(
            [res_df,
            blank_row,
            avg_df],
            ignore_index=True
        )
        output_name = "Results/Leg_comp_result.csv"
        if save_results == True:
            csv_df.to_csv(output_name, index=False)
            print(f"\nSaved → {output_name}")
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    all_rw: list[pd.DataFrame] = []
    all_lw: list[pd.DataFrame] = []
    all_rg: list[pd.DataFrame] = []

    for data_path in DATA_PATHS:
        dataset_name = os.path.basename(data_path.rstrip('/\\'))
        print(f"\n{'=' * 80}")
        print(f"  Merging: {dataset_name}")
        print(f"{'=' * 80}")
        rw, lw, rl = merge_all(data_path)
        # Tag each row with its source dataset for traceability
        # rw['dataset'] = dataset_name
        # lw['dataset'] = dataset_name
        rl['dataset'] = dataset_name
        # all_rw.append(rw)
        # all_lw.append(lw)
        all_rg.append(rl)

    # Pool across all datasets
    # rw_merged = pd.concat(all_rw, ignore_index=True) if all_rw else pd.DataFrame()
    # lw_merged = pd.concat(all_lw, ignore_index=True) if all_lw else pd.DataFrame()
    rl_merged = pd.concat(all_rg, ignore_index=True) if all_rg else pd.DataFrame()

    print(f"\n{'=' * 80}")
    print(f"  Running GSD on pooled data ({len(DATA_PATHS)} dataset(s))")
    print(f"{'=' * 80}")
    process_leg(rl_merged, SAVE_RESULTS)
    plt.show()