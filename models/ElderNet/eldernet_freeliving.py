## Run ElderNet on free living stroke patient data

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import pickle
from scipy import signal
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import random

# --- REPRODUCIBILITY ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.paths import FREELIVING_PATH, PLOTS_DIR, RESULTS_DIR
from utils.hub_utils import safe_hub_load

DATA_PATH = FREELIVING_PATH
PLOT_DATASET_NAME = os.path.basename(DATA_PATH)
WINDOW_SIZE = 300
STEP_SIZE = 30
REPO_NAME = 'yonbrand/ElderNet'
SAMPLE_RATE_QSENSE = 50.0
RESULTS_DIR = os.path.join(RESULTS_DIR, 'ElderNet')

# Thresholds
CONF_THRESH = 0.65
MIN_ENERGY = 0.07
MAX_ENERGY = 0.4
MIN_FREQ = 0.5
MAX_FREQ = 3.5


def load_data(filepath, annotated_path=None):
    # PEEK AT FILE TO DETECT HEADER AND SEPARATOR
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()

    skiprows = 1 if first_line.startswith('BLE') else 0
    
    # Detect separator from the actual data line
    data_line = second_line if skiprows == 1 else first_line
    sep = '\t' if '\t' in data_line else ','

    df = pd.read_csv(filepath, sep=sep, engine='python', skiprows=skiprows, on_bad_lines='skip')
    df = df.reset_index(drop=True)

    # --- CREATE DATETIME COLUMN ---
    first_col = df.columns[0].strip().lower()
    if first_col == 'time':
        df['datetime'] = pd.to_datetime(df.iloc[:, 0], dayfirst=True, errors='coerce')
    else:
        df['datetime'] = pd.to_datetime(
            df.iloc[:, 0].astype(str) + ' ' + df.iloc[:, 1].astype(str),
            errors='coerce'
        )

    # Remove rows with invalid timestamps
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime')
    df = df.drop_duplicates(subset='datetime', keep='first')
    df = df.reset_index(drop=True)

    # --- COMPUTE time_seconds BEFORE using it below ---
    time_seconds = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds().values

    # --- LOAD GROUND TRUTH ---
    if annotated_path and os.path.exists(annotated_path):
        df_ann = pd.read_csv(annotated_path, sep=None, engine="python")
        label_col = 'label' if 'label' in df_ann.columns else 'Label'
        sample_gt = pd.to_numeric(df_ann[label_col], errors='coerce').fillna(0).astype(int)
        min_len = min(len(df), len(sample_gt))
        df = df.iloc[:min_len].reset_index(drop=True)
        time_seconds = time_seconds[:min_len]
        sample_gt = sample_gt.iloc[:min_len].values

    elif 'label' in df.columns or 'Label' in df.columns:
        label_col = 'label' if 'label' in df.columns else 'Label'
        sample_gt = pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(int).values

    else:
        print(f"Warning: No label found for {filepath}, skipping.")
        return None

    # --- DETECT COLUMN NAMES ---
    if 'ax' in df.columns:
        acc_x_col, acc_y_col, acc_z_col = 'ax', 'ay', 'az'
    else:
        acc_x_col, acc_y_col, acc_z_col = 'accX', 'accY', 'accZ'

    energy_col = 'Energy' if 'Energy' in df.columns else None

    data = pd.DataFrame({
        'time_sec': time_seconds,
        'accX': pd.to_numeric(df[acc_x_col], errors='coerce'),
        'accY': pd.to_numeric(df[acc_y_col], errors='coerce'),
        'accZ': pd.to_numeric(df[acc_z_col], errors='coerce'),
        'gt': sample_gt,
        'energy': pd.to_numeric(df[energy_col], errors='coerce') if energy_col else np.zeros(len(df))
    })

    return data

# --- RESAMPLE DATA TO 30Hz ----
def resample_to_30hz(filepath, annotated_path=None, original_fs=SAMPLE_RATE_QSENSE):
    df = load_data(filepath, annotated_path=annotated_path)

    if df is None:
        return None

    if abs(original_fs - 30.0) < 0.1:
        return df

    t = df['time_sec'].values
    new_time = np.linspace(t[0], t[-1], int((t[-1] - t[0]) * 30.0) + 1)

    resampled = pd.DataFrame({
        'time_sec': new_time,
        'accX': np.interp(new_time, t, df['accX'].values),
        'accY': np.interp(new_time, t, df['accY'].values),
        'accZ': np.interp(new_time, t, df['accZ'].values),
        'gt': np.round(np.interp(new_time, t, df['gt'].values)).astype(int),
        'energy': np.interp(new_time, t, df['energy'].values)
    })

    return resampled


# --- PREPARE WINDOWS FOR ELDERNET ---
def prepare_windows_overlapping(df):
    acc_data = df[['accX', 'accY', 'accZ']].values
    gt_raw = df['gt'].values
    times = df['time_sec'].values
    q_energies = df['energy'].values
    windows, energies, freqs, activities, timestamps, Q_energies = [], [], [], [], [], []

    def get_dominant_freq(win, fs=30):
        mag = np.sqrt(np.sum(win**2, axis=0))
        mag = mag - np.mean(mag)
        freqs_fft = np.fft.rfftfreq(len(mag), d=1/fs)
        fft_vals = np.abs(np.fft.rfft(mag))
        return freqs_fft[np.argmax(fft_vals)]

    for i in range(WINDOW_SIZE, len(acc_data), STEP_SIZE):
        win = acc_data[i - WINDOW_SIZE:i]
        act_win = gt_raw[i - WINDOW_SIZE:i]

        windows.append(win.T)
        energies.append(np.std(np.sqrt(np.sum(win**2, axis=1))))
        freqs.append(get_dominant_freq(win.T))
        activities.append(int(np.mean(act_win) > 0.5))
        timestamps.append(times[i-1])
        Q_energies.append(np.mean(q_energies[i - WINDOW_SIZE:i]))

    return (
        torch.FloatTensor(np.array(windows)),
        np.array(energies),
        np.array(freqs),
        activities,
        np.array(timestamps),
        np.array(Q_energies)
    )


# --- PLOTTING FUNCTION ---
def plot_per_subject(results_list, metrics):
    """One plot per subject showing all metrics and GT/predictions."""

    subjects = sorted(set(r['subject'] for r in results_list))
    print(f"\nPlotting {len(subjects)} subjects: {subjects}")

    for subject in subjects:
        subject_results = [r for r in results_list if r['subject'] == subject]

        if not subject_results:
            print(f"  No data for {subject}, skipping.")
            continue

        n_rows = len(metrics) + 1  # metrics rows + GT/Pred row
        fig, axes = plt.subplots(n_rows, 1,
                                  figsize=(18, 4 * n_rows),
                                  sharex=False)

        if n_rows == 1:
            axes = [axes]

        fig.suptitle(f"Subject: {subject}", fontsize=16, fontweight='bold')

        # --- METRIC ROWS ---
        for ax, metric in zip(axes[:-1], metrics):

            for result in subject_results:
                if metric == 'probability':
                    values = result['probability']
                elif metric == 'energy':
                    values = result['energy']
                elif metric == 'Q_energies':
                    values = result['Q_energies']
                elif metric == 'frequency':
                    values = result['frequency']
                else:
                    continue

                x = result['timestamps']
                label = result['file']

                ax.plot(x, values,
                        linewidth=1.5,
                        alpha=0.9,
                        label=label)

            # --- Threshold lines ---
            if metric == 'probability':
                ax.axhline(CONF_THRESH,
                           color='black', linestyle='--', linewidth=1.5,
                           label=f'Threshold = {CONF_THRESH}')
                ax.set_ylim(-0.05, 1.1)

            elif metric == 'energy':
                ax.axhline(MIN_ENERGY,
                           color='black', linestyle='--', linewidth=1.5,
                           label=f'Min energy = {MIN_ENERGY}')
                ax.axhline(MAX_ENERGY,
                           color='red', linestyle='--', linewidth=1.5,
                           label=f'Max energy = {MAX_ENERGY}')

            elif metric == 'Q_energies':
                ax.axhline(MIN_ENERGY,
                           color='black', linestyle='--', linewidth=1.5,
                           label=f'Min Q-energy = {MIN_ENERGY}')
                ax.axhline(MAX_ENERGY,
                           color='red', linestyle='--', linewidth=1.5,
                           label=f'Max Q-energy = {MAX_ENERGY}')

            elif metric == 'frequency':
                ax.axhline(MIN_FREQ,
                           color='black', linestyle='--', linewidth=1.5,
                           label=f'Min freq = {MIN_FREQ}')
                ax.axhline(MAX_FREQ,
                           color='red', linestyle='--', linewidth=1.5,
                           label=f'Max freq = {MAX_FREQ}')

            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=10)

        # --- GT / PREDICTION ROW ---
        ax_gt_pred = axes[-1]

        for result in subject_results:
            timestamps_raw = result['raw_timestamps']
            y_true_raw = result['raw_gt']
            y_pred = result['y_pred']
            timestamps = result['timestamps']
            label = result['file']

            # Ground truth as shaded band
            ax_gt_pred.fill_between(
                timestamps_raw,
                0, y_true_raw,
                step='post',
                alpha=0.25,
                label=f'{label} | GT'
            )

            # Prediction as stepped line, slightly offset for visibility
            ax_gt_pred.step(
                timestamps,
                y_pred + 0.05,
                where='post',
                linewidth=2.5,
                label=f'{label} | Pred'
            )

        ax_gt_pred.set_ylabel("GT / Prediction", fontsize=12)
        ax_gt_pred.set_ylim(-0.1, 1.15)
        ax_gt_pred.set_xlabel("Time (seconds)", fontsize=12)
        ax_gt_pred.legend(fontsize=9, loc='upper right')
        ax_gt_pred.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        plots_dir = os.path.join(PLOTS_DIR, PLOT_DATASET_NAME, 'eldernet')
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, f"subject_{subject}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot: {save_path}")
        plt.show()


# --- RUN ELDERNET AND OBTAIN PROBABILITIES ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = safe_hub_load(REPO_NAME, 'eldernet_ft', trust_repo=True).to(device)
    model.eval()

    results = []

    for file in sorted(os.listdir(DATA_PATH)):
        if not file.endswith('.csv'):
            continue
        if '_annotated' not in file:  # ONLY process annotated files
            continue

        filepath = os.path.join(DATA_PATH, file)

        # Extract subject, e.g. Device2_sub1_annotated.csv -> sub1
        parts = file.replace('_annotated.csv', '').split('_')
        subject = parts[1] if len(parts) > 1 else 'Unknown'

        try:
            df_30hz = resample_to_30hz(filepath, annotated_path=None)  # labels are inside the file itself
            if df_30hz is None:
                continue

            Q_energy = df_30hz['energy'].values
            wins, engs, frqs, acts, tmstps, Q_energies = prepare_windows_overlapping(df_30hz)

            with torch.no_grad():
                logits = model(wins.to(device))
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            # Save per-window outputs
            output_df = pd.DataFrame({
                "timestamp": tmstps,
                "probability": probs,
                "energy": engs,
                "frequency": frqs,
                "Q_energies": Q_energies
            })
            save_path = os.path.join(DATA_PATH, file.replace('.csv', '_window_outputs.csv'))
            output_df.to_csv(save_path, index=False)

            print(f"  Processed {file}: {len(probs)} windows")

            y_pred = (probs > CONF_THRESH).astype(int)

            # --- Window-level GT ---
            n_windows = len(probs)
            y_true_full = df_30hz['gt'].values

            start_idxs = np.arange(0, n_windows * STEP_SIZE, STEP_SIZE)
            start_idxs = start_idxs[start_idxs + WINDOW_SIZE <= len(y_true_full)]

            y_true = np.array([
                int(np.mean(y_true_full[i:i + WINDOW_SIZE]) > 0.5)
                for i in start_idxs
            ])
            timestamps = df_30hz['time_sec'].values[start_idxs]

            # Pad if needed
            if len(y_true) < len(y_pred):
                n_pad = len(y_pred) - len(y_true)
                y_true = np.pad(y_true, (0, n_pad), mode='edge')
                timestamps = np.pad(timestamps, (0, n_pad), mode='edge')

            # --- Metrics ---
            if np.sum(y_true) == 0:
                precision = recall = f1 = 0.0
            else:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=[1], average='binary', zero_division=0
                )
            accuracy = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            print(
                f"  {file} | Precision: {precision:.3f} | Recall: {recall:.3f} | "
                f"F1: {f1:.3f} | Accuracy: {accuracy:.3f}"
            )

            results.append({
                'subject': subject,
                'file': file.replace('.csv', ''),
                'file_path': filepath,
                # Raw data for plotting
                'raw_timestamps': df_30hz['time_sec'].values,
                'raw_gt': df_30hz['gt'].values,
                'Q_energy': Q_energy,
                'timestamps': timestamps,
                'y_true': y_true,
                'y_pred': y_pred,
                'probability': probs,
                'energy': engs,
                'frequency': frqs,
                'Q_energies': Q_energies,
                # Metrics
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'confusion_matrix': cm.tolist()
            })

        except Exception as e:
            print(f"  Error processing {file}: {e}")
            continue

    if not results:
        print("No results to summarise or plot.")
        return

    # --- METRICS SUMMARY ---
    df_metrics = pd.DataFrame([{
        'subject': r['subject'],
        'file': r['file'],
        'precision': r['precision'],
        'recall': r['recall'],
        'f1': r['f1'],
        'accuracy': r['accuracy']
    } for r in results])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_csv = os.path.join(RESULTS_DIR, 'eldernet_FreeLiving_metrics.csv')
    df_metrics.to_csv(metrics_csv, index=False)
    print(f"Saved metrics summary to: {metrics_csv}")

    def pooled_metrics(rows):
        tn = fp = fn = tp = 0
        for row in rows:
            c = np.array(row.get('confusion_matrix', []))
            if c.shape != (2, 2):
                continue
            tn += int(c[0, 0])
            fp += int(c[0, 1])
            fn += int(c[1, 0])
            tp += int(c[1, 1])

        total = tp + tn + fp + fn
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        acc = (tp + tn) / total if total > 0 else 0.0
        return {'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc}

    def pooled_by_group(rows, key):
        groups = {}
        for row in rows:
            g = row.get(key)
            if g is None:
                continue
            groups.setdefault(g, []).append(row)

        out = []
        for g, g_rows in sorted(groups.items()):
            m = pooled_metrics(g_rows)
            m[key] = g
            out.append(m)

        return pd.DataFrame(out, columns=[key, 'precision', 'recall', 'f1', 'accuracy'])

    print("\n=== OVERALL SUMMARY ===")
    print("\nBy Subject:")
    print(pooled_by_group(results, 'subject').set_index('subject'))
    print("\nOverall:")
    print(pd.Series(pooled_metrics(results)))

    # --- PLOTTING ---
    metrics = ['probability', 'energy', 'Q_energies', 'frequency']
    plot_per_subject(results, metrics)


if __name__ == "__main__":
    main()