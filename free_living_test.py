import os
import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from GSD3_test import KheirkhahanGSD
from multimob.GSD.GSD4 import MacLeanGSD
from multimob.GSD.GSD5 import KerenGSD
from GSD2a import HickeyGSD
from singleGSD_robust import plot_results

warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

DATA_PATHS = [
    r"C:\Users\orlov\intern\gait_detection\Free_living"
]
GSD_n = 3
SAMPLING_RATE = 50
DEBUG = False
GAIT_CLASSES = {'walking', 'stairs'}
SAVE_RESULTS = False
PRINT_STATS = True



def load_csv(filepath: str):
    """
    Read one sensor file, scale acc to m/s², rename columns.
    Returns (imu_df, df_full) on success, or None on failure.
    
    FIX: was returning None on failure but callers tried to unpack as a tuple.
    FIX: acc_cols filter was 'a' in col — too broad, matched Label/gz/etc.
    """
    try:
        df = pd.read_csv(filepath, sep=None, engine="python")
        # df = df.reset_index(drop=True)
        
        # 3. Create datetime helper
        # Using Month/Day/Year as identified previously
        df['time_dt'] = pd.to_datetime(df['time'], format='%m/%d/%Y %H:%M:%S.%f')

        # 4. Filter non-increasing timestamps
        time_diffs = df['time_dt'].diff().dt.total_seconds()
        valid_mask = (time_diffs > 0) | (time_diffs.isna())
        df = df[valid_mask].copy()
        
        dropped_rows = len(df) - valid_mask.sum()
        df = df[valid_mask].copy()

        # 5. Add requested columns in specific string formats
        # yyyy-MM-dd (e.g., 2025-06-19)
        df['yyyy-MM-dd'] = df['time_dt'].dt.strftime('%Y-%m-%d')
        
        # HH:mm:ss.fff (e.g., 10:45:26.605)
        # Note: %f provides microseconds, so we slice to get 3 digits (milliseconds)
        df['HH:mm:ss.fff'] = df['time_dt'].dt.strftime('%H:%M:%S.%f').str[:-3]
        
        # 6. Identify Gaps and Segments
        # Re-calculate diffs on the cleaned data
        df['gap_ms'] = df['time_dt'].diff().dt.total_seconds() * 1000
        
        threshold = 1001 / SAMPLING_RATE # e.g., 20.02ms for 50Hz
        
        # Every time a gap exceeds the threshold, the cumsum increments the segment ID
        df['segment'] = (df['gap_ms'] > threshold).cumsum().fillna(0).astype(int)

        if DEBUG:
            print(f"Dropped {dropped_rows} rows.")
            print(f"Found {df['segment'].nunique()} segments.")
            print(f"Kept {len(df)} rows.")

        # Cleanup: Remove helper columns if you want to keep the DF clean
        # df = df.drop(columns=['time_dt', 'gap_ms'])

        # df = pd.DataFrame(clean_rows)
        # df = df.reset_index(drop=True)
        # df['segment'] = segments

        # FIX: use startswith('a') instead of 'a' in col.lower() to avoid
        # matching unrelated columns like 'Label', 'gz', 'mag_x', etc.
        acc_cols = [c for c in df.columns if c.lower().startswith('a')]
        if len(acc_cols) < 3:
            print(f"  [SKIP] Not enough acc columns in {filepath} (found {len(acc_cols)})")
            return None

        # imu_df = df[acc_cols[:3]].copy()
        # imu_df = imu_df * 9.81          # convert g → m/s²
        # imu_df['segment'] = df['segment']
        # imu_df['y_true'] = df['Label']
        # imu_df['yyyy-MM-dd'] = df['yyyy-MM-dd']
        # imu_df['HH:mm:ss.fff'] = df['HH:mm:ss.fff']
        # imu_df.columns = ['yyyy-MM-dd', 'HH:mm:ss.fff', 'acc_is', 'acc_ml', 'acc_pa', 'segment', 'y_true']
        imu_df = df[acc_cols[:3]].copy()
        imu_df.columns = ['acc_is', 'acc_ml', 'acc_pa'] # Rename just the accels first
        imu_df = imu_df * 9.81

        imu_df['segment'] = df['segment']
        # imu_df['y_true'] = df['Label']
        imu_df['yyyy-MM-dd'] = df['yyyy-MM-dd']
        imu_df['HH:mm:ss.fff'] = df['HH:mm:ss.fff']
        
        y_true_clean = pd.to_numeric(df['Label'], errors='coerce').fillna(0).astype(int)
        imu_df['y_true'] = y_true_clean

        
        return imu_df

    except Exception as e:
        print(f"  [ERROR] Failed to load {filepath}: {e}")
        return None


def merge_csv(data_path: str, PRINT_STATS: bool = False) -> pd.DataFrame:
    """
    Walk data_path, load every *_annotated.csv file, attach metadata,
    and return a single merged DataFrame.

    Columns: subject | y_true | acc_is | acc_ml | acc_pa
    """
    imu_merged_chunks: list[pd.DataFrame] = []

    if PRINT_STATS:
        print(f"Scanning: {data_path}\n")
        print(f"{'File':<35} | {'Rows':>8}")
        print("-" * 50)

    files = [f for f in os.listdir(data_path) if f.endswith('_annotated.csv')]

    for file in files:
        filepath = os.path.join(data_path, file)

        imu_df = load_csv(filepath)
        if imu_df is None:
            continue
        # imu_df = result
        imu_df['subject'] = file.split('.')[0]
        imu_df['condition'] = 'stroke'
        imu_merged_chunks.append(imu_df)

        if PRINT_STATS:
            print(f"{file[:35]:<35} | {len(imu_df):>8}")

    if PRINT_STATS:
        print("-" * 50)

    col_order = ['yyyy-MM-dd', 'HH:mm:ss.fff','subject', 'segment', 'condition', 'y_true', 'acc_is', 'acc_ml', 'acc_pa']

    imu_merged = (
        pd.concat(imu_merged_chunks, ignore_index=True)[col_order]
        if imu_merged_chunks
        else pd.DataFrame(columns=col_order)
    )

    return imu_merged


def _run_gsd_on_group(imu_df: pd.DataFrame, y_true: np.ndarray,
                      label: str):
    """
    Run GSD on a single contiguous imu_df block, evaluate against y_true,
    and return (metrics_dict, output_name) or None on error.
    """
    try:
        match GSD_n:
            case 2:
                gsd = HickeyGSD(debug=DEBUG)
                detected_bouts = (
                    gsd.preprocess(imu_df, sampling_rate_hz=SAMPLING_RATE,
                                   target_sampling_rate_hz=SAMPLING_RATE)
                    .detect_wrist()
                )
                output_name = 'HickeyGSD_Results.csv'
            case 3:
                gsd = KheirkhahanGSD()
                detected_bouts = gsd.detect(imu_df, sampling_rate_hz=SAMPLING_RATE)
                activity_counts = gsd.get_activity(imu_df, sampling_rate_hz=SAMPLING_RATE)
                std_norm = gsd.get_std_norm(imu_df, sampling_rate_hz=SAMPLING_RATE)
                output_name = 'KheirkhahanGSD_Results.csv'
            case 4:
                gsd = MacLeanGSD()
                detected_bouts = gsd.detect(imu_df)
                output_name = 'MacLeanGSD_Results.csv'
            case 5:
                gsd = KerenGSD()
                detected_bouts = gsd.detect(imu_df, sampling_rate_hz=SAMPLING_RATE)
                output_name = 'KerenGSD_Results.csv'
            case _:
                print(f"  [ERROR] Unknown GSD_n={GSD_n}")
                return None

        # Convert bout list → binary mask
        y_pred = np.zeros(len(imu_df), dtype=int)
        if hasattr(detected_bouts, 'gs_list_') and not detected_bouts.gs_list_.empty:
            for _, row in detected_bouts.gs_list_.iterrows():
                start = int(max(0, row['start']))
                end   = int(min(len(imu_df), row['end']))
                y_pred[start:end] = 1

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        if DEBUG:
            print(f"\n[{label}] pred walking: {y_pred.sum()} / {len(y_pred)} samples")
            print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

        return {
            'Accuracy':  accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall':    recall_score(y_true, y_pred, zero_division=0),
            'F1':        f1_score(y_true, y_pred, zero_division=0),
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
        }, output_name, activity_counts, std_norm

    except Exception as e:
        print(f"  [ERROR] GSD failed on {label}: {e}")
        return None


def process_gait(imu_merged: pd.DataFrame,
                 save_results: bool = True) -> pd.DataFrame:
    """
    Run GSD on every subject segment inside imu_merged.
    Prints a per-file table and overall averages; optionally saves results CSV.

    FIX: previously accepted (rw_merged, lw_merged) — simplified to one DataFrame
         since merge_all_wrists only returns one.
    FIX: _avg_row was called with an extra positional '' argument that didn't
         match its signature — removed the stray argument.
    """
    results = []

    print(f"\n{'Subject':<40} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8}")
    print("-" * 80)

    if imu_merged.empty:
        print("No data — skipping.")
        return pd.DataFrame()

    imu_cols = ['acc_is', 'acc_ml', 'acc_pa']

    for subject, grp in imu_merged.groupby('subject', sort=True):
        imu_df = grp[imu_cols].reset_index(drop=True)
        y_true = grp['y_true'].to_numpy()
        label  = str(subject)

        result = _run_gsd_on_group(imu_df, y_true, label)
        if result is None:
            continue

        metrics, output_name, _, _ = result
        output_name = 'Results/Free_living_Results.csv'  

        results.append({
            'Subject': label,
            'Folder':  subject,
            **metrics,
        })

        if PRINT_STATS:
            print(f"{label[:40]:<40} | {metrics['Accuracy']:.4f}   | "
                  f"{metrics['Precision']:.4f}   | {metrics['Recall']:.4f}   | "
                  f"{metrics['F1']:.4f}")

    if not results:
        print("No results to summarise.")
        return pd.DataFrame()

    res_df = pd.DataFrame(results)
    VARIABLES = ['TP', 'FP', 'FN', 'TN']

    def _avg_row(row_type: str, label: str, subset: pd.DataFrame) -> dict:
        """
        Build a summary row from a subset of res_df using macro-averaged metrics.

        FIX: previously called with an extra positional '' argument (for a removed
             'condition' parameter) — signature now matches all call sites.
        """
        tp = subset['TP'].sum()
        fp = subset['FP'].sum()
        fn = subset['FN'].sum()
        tn = subset['TN'].sum()
        total = tp + fp + fn + tn

        accuracy_av  = (tp + tn) / total                    if total > 0             else 0.0
        precision_av = tp / (tp + fp)                       if (tp + fp) > 0         else 0.0
        recall_av    = tp / (tp + fn)                       if (tp + fn) > 0         else 0.0
        f1_av        = (2 * precision_av * recall_av /
                        (precision_av + recall_av))         if (precision_av + recall_av) > 0 else 0.0

        return {
            'row_type':  row_type,
            'Subject':   label,
            'Folder':    '',
            'Accuracy':  accuracy_av,
            'Precision': precision_av,
            'Recall':    recall_av,
            'F1':        f1_av,
            **{p: round(subset[p].sum(), 4) for p in VARIABLES},
        }

    def _print_avg(label: str, subset: pd.DataFrame):
        if subset.empty:
            return
        tp = subset['TP'].sum()
        fp = subset['FP'].sum()
        fn = subset['FN'].sum()
        tn = subset['TN'].sum()
        total = tp + fp + fn + tn

        accuracy_av  = (tp + tn) / total                    if total > 0             else 0.0
        precision_av = tp / (tp + fp)                       if (tp + fp) > 0         else 0.0
        recall_av    = tp / (tp + fn)                       if (tp + fn) > 0         else 0.0
        f1_av        = (2 * precision_av * recall_av /
                        (precision_av + recall_av))         if (precision_av + recall_av) > 0 else 0.0

        print(f"{label:<40} | {accuracy_av:.5f}  | {precision_av:.5f}  | "
              f"{recall_av:.5f}  | {f1_av:.5f}")

    # FIX: _avg_row called without the stale extra '' positional argument
    avg_rows = [_avg_row('avg_overall', 'AVERAGE (Overall)', res_df)]

    print("-" * 80)
    _print_avg("AVERAGE (Overall)", res_df)

    # Build CSV
    res_df.insert(0, 'row_type', 'result')
    avg_df    = pd.DataFrame(avg_rows)
    blank_row = pd.DataFrame([{c: '' for c in res_df.columns}])

    csv_df = pd.concat([res_df, blank_row, avg_df], ignore_index=True)

    if save_results:
        csv_df.to_csv(output_name, index=False)
        print(f"\nSaved → {output_name}")

    return res_df


if __name__ == "__main__":
    all_imu: list[pd.DataFrame] = []

    for data_path in DATA_PATHS:
        dataset_name = os.path.basename(data_path.rstrip('/\\'))
        print(f"\n{'=' * 80}")
        print(f"  Merging: {dataset_name}")
        print(f"{'=' * 80}")

        imu = merge_csv(data_path)
        imu['dataset'] = dataset_name
        all_imu.append(imu)

    # Pool across all datasets
    imu_merged = pd.concat(all_imu, ignore_index=True) if all_imu else pd.DataFrame()

    print(f"\n{'=' * 80}")
    print(f"  Running GSD on pooled data ({len(DATA_PATHS)} dataset(s))")
    print(f"{'=' * 80}")

    # FIX: process_gait now takes one DataFrame + save_results flag,
    #      matching the updated function signature.
    process_gait(imu_merged, SAVE_RESULTS)