import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix as cm_func
import matplotlib.pyplot as plt
import psutil

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.paths import QSENSE_DATA, QSENSE_EDGE, QSENSE_MIXED, FREELIVING_PATH, STROKENET_WEIGHTS, PLOTS_DIR, RESULTS_DIR
from utils.hub_utils import safe_hub_load

DATASET_PATH = FREELIVING_PATH  # change to QSENSE_DATA/QSENSE_MIXED/QSENSE_EDGE/FREELIVING_PATH as needed
WEIGHTS_PATH = STROKENET_WEIGHTS
REPO_NAME     = 'yonbrand/ElderNet'

# Updated for finetuned model: 2s at 50Hz
WINDOW_SIZE        = 100
STEP_SIZE          = 50
SAMPLE_RATE_QSENSE = 50.0
GAP_THRESHOLD      = 0.1

GAIT_CLASSES  = {'Walking', 'Stairs'}
CONF_THRESH   = 0.5    # update threshold for finetuned model
MIN_ENERGY    = 0.07
MAX_ENERGY    = 0.4
MIN_FREQ      = 0.5
MAX_FREQ      = 3.5

def get_memory_usage():
    """Total process RAM usage."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def get_model_memory(model):
    """Memory occupied by model parameters and buffers."""
    param_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes  = param_bytes + buffer_bytes
    return {
        'parameters_mb': param_bytes  / 1024**2,
        'buffers_mb':    buffer_bytes / 1024**2,
        'total_mb':      total_bytes  / 1024**2,
        'n_params':      sum(p.numel() for p in model.parameters()),
        'n_trainable':   sum(p.numel() for p in model.parameters() if p.requires_grad)
    }


def get_gpu_memory():
    """Current GPU memory allocation and peak usage."""
    if not torch.cuda.is_available():
        return None
    return {
        'allocated_mb':  torch.cuda.memory_allocated()  / 1024**2,
        'reserved_mb':   torch.cuda.memory_reserved()   / 1024**2,
        'peak_mb':       torch.cuda.max_memory_allocated() / 1024**2,
        'total_mb':      torch.cuda.get_device_properties(0).total_memory / 1024**2
    }


def print_memory_report(model, label=''):
    """Print a full memory report."""
    m   = get_model_memory(model)
    gpu = get_gpu_memory()
    ram = get_memory_usage() / 1024**2

    print(f"\n=== Memory Report {f'({label}) ' if label else ''}===")
    print(f"  Model parameters:  {m['n_params']:,} total, {m['n_trainable']:,} trainable")
    print(f"  Model size:        {m['total_mb']:.2f} MB  "
          f"(params={m['parameters_mb']:.2f} MB, buffers={m['buffers_mb']:.2f} MB)")
    print(f"  Process RAM:       {ram:.1f} MB")
    if gpu:
        print(f"  GPU allocated:     {gpu['allocated_mb']:.1f} MB")
        print(f"  GPU reserved:      {gpu['reserved_mb']:.1f} MB")
        print(f"  GPU peak:          {gpu['peak_mb']:.1f} MB")
        print(f"  GPU total:         {gpu['total_mb']:.1f} MB")
        print(f"  GPU free:          {gpu['total_mb'] - gpu['reserved_mb']:.1f} MB")

# ============================================================
# MODEL LOADING
# ============================================================

def fix_circular_padding(model):
    for module in model.modules():
        if isinstance(module, nn.Conv1d) and module.padding_mode == 'circular':
            module.padding_mode = 'zeros'
            module._reversed_padding_repeated_twice = (
                module.padding[0], module.padding[0]
            )
    return model

def remove_last_downsample(model):
    layer5 = model.feature_extractor.layer5
    new_layer5 = nn.Sequential(
        *[child for idx, child in enumerate(layer5.children()) if idx != 3]
    )
    model.feature_extractor.layer5 = new_layer5
    return model

def load_finetuned_model(weights_path):
    model = safe_hub_load(REPO_NAME, 'eldernet_ft', trust_repo=True)
    model = fix_circular_padding(model)
    model = remove_last_downsample(model)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    print(f"Loaded finetuned weights from {weights_path}")
    return model


# ============================================================
# DATA LOADING
# ============================================================

def load_data(filepath):
    df = pd.read_csv(filepath, sep=None, engine='python')
    df = df.reset_index(drop=True)

    parent_folder = os.path.basename(os.path.dirname(filepath))

    df['datetime'] = pd.to_datetime(
        df.iloc[:, 0].astype(str) + ' ' + df.iloc[:, 1].astype(str),
        errors='coerce'
    )
    df = df.dropna(subset=['datetime']).reset_index(drop=True)

    # Step 0: Remove backwards-jump blocks
    running_max = df['datetime'].iloc[0]
    keep = []
    for t in df['datetime']:
        if t < running_max:
            keep.append(False)
        else:
            keep.append(True)
            running_max = t
    df = df[keep].reset_index(drop=True)

    # Step 1: Fix time travelers
    dt = df['datetime'].diff()
    jump_idx = dt[abs(dt) > pd.Timedelta(days=100)].index
    for idx in jump_idx:
        false_gap = dt[idx] - pd.Timedelta(seconds=1/50)
        df.loc[idx:, 'datetime'] = df.loc[idx:, 'datetime'] - false_gap
        dt = df['datetime'].diff()

    # Step 2: Sort
    df = df.sort_values('datetime').reset_index(drop=True)

    # Step 3: Remove duplicates
    df = df.drop_duplicates(subset='datetime', keep='first').reset_index(drop=True)

    df['time_sec'] = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds()

    # Ground truth
    if 'Label' in df.columns or 'label' in df.columns:
        label_col = 'Label' if 'Label' in df.columns else 'label'
        sample_gt = pd.to_numeric(df[label_col], errors='coerce').fillna(0).astype(int)
    else:
        activity_name = parent_folder.split('_')[0]
        sample_gt = np.ones(len(df), dtype=int) if activity_name in GAIT_CLASSES else np.zeros(len(df), dtype=int)

    col_names = df.columns.tolist()
    data = pd.DataFrame({
        'time_sec': df['time_sec'].values,
        'accX':     pd.to_numeric(df[col_names[5]], errors='coerce'),
        'accY':     pd.to_numeric(df[col_names[6]], errors='coerce'),
        'accZ':     pd.to_numeric(df[col_names[7]], errors='coerce'),
        'gt':       sample_gt,
        'energy':   pd.to_numeric(df[col_names[12]], errors='coerce')
    })

    return data


# ============================================================
# WINDOWING — gap-aware, 50Hz native (no resampling)
# ============================================================

def prepare_windows(df, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    acc_data  = df[['accX', 'accY', 'accZ']].values
    gt_raw    = df['gt'].values
    times     = df['time_sec'].values
    q_energies = df['energy'].values

    def get_dominant_freq(win, fs=50, nfft_size=512):
        mag = np.sqrt(np.sum(win**2, axis=0))
        mag = mag - np.mean(mag)
        hann_win = np.hanning(len(mag))
        mag_windowed = mag * hann_win
        freqs_fft = np.fft.rfftfreq(nfft_size, d=1/fs)
        power_spectrum = np.abs(np.fft.rfft(mag_windowed, n=nfft_size))**2
        return freqs_fft[np.argmax(power_spectrum)]

    # Gap-aware segmentation
    dt       = np.diff(times)
    gap_idx  = np.where(dt > GAP_THRESHOLD)[0] + 1
    bounds   = np.concatenate([[0], gap_idx, [len(acc_data)]])

    windows, energies, freqs, activities, timestamps, Q_energies_out = [], [], [], [], [], []

    for k in range(len(bounds) - 1):
        seg_start = bounds[k]
        seg_end   = bounds[k + 1]
        if (seg_end - seg_start) < window_size:
            continue

        for i in range(seg_start + window_size, seg_end, step_size):
            win     = acc_data[i - window_size:i]
            act_win = gt_raw[i - window_size:i]
            windows.append(win.T)                                              # (3, 100)
            energies.append(np.std(np.sqrt(np.sum(win**2, axis=1))))          # no 9.81^3 — 50Hz native
            freqs.append(get_dominant_freq(win.T, fs=50))
            activities.append(int(np.mean(act_win) > 0.5))
            timestamps.append(times[i - 1])
            Q_energies_out.append(np.mean(q_energies[i - window_size:i]))

    return (
        torch.FloatTensor(np.array(windows)),
        np.array(energies),
        np.array(freqs),
        np.array(activities),
        np.array(timestamps),
        np.array(Q_energies_out),
        times,           # raw sample-level times
        gt_raw           # raw sample-level GT
    )


# ============================================================
# PLOTTING — reused from original script
# ============================================================

def plot_per_activity(results_list, subjects, metrics, plots_dir):
    activities_by_type = {}
    for result in results_list:
        act = result['activity_type']
        activities_by_type.setdefault(act, []).append(result)

    activity_types = sorted(activities_by_type.keys())

    # Color per subject, linestyle per wrist
    color_palette = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    all_subjects_in_results = sorted(set(r['subject'].capitalize() for r in results_list))
    subject_colors = {subj: color_palette[idx % len(color_palette)]
                      for idx, subj in enumerate(all_subjects_in_results)}
    wrist_styles = {'right': '-', 'left': '--'}

    for activity_type, activity_results in activities_by_type.items():
        n_rows = len(metrics) + 1
        fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4 * n_rows), sharex=False)
        if n_rows == 1:
            axes = [axes]
        fig.suptitle(f"Activity: {activity_type} (StrokeNet)",
                     fontsize=16, fontweight='bold')

        has_data = False

        # --- Metric subplots ---
        for ax, metric in zip(axes[:-1], metrics):
            for subject in subjects:
                subject_results = [r for r in activity_results
                                   if r['subject'].capitalize() == subject.capitalize()]
                for result in subject_results:
                    if metric not in ['probability', 'energy', 'Q_energies', 'frequency']:
                        continue
                    has_data = True

                    subj_cap  = result['subject'].capitalize()
                    color     = subject_colors.get(subj_cap, '#333333')
                    style     = wrist_styles[result['wrist']]
                    label_str = f"{subj_cap} | {result['wrist']}"

                    if metric == 'probability':
                        values = result['probability']
                    elif metric == 'energy':
                        values = result['energy']
                    elif metric == 'Q_energies':
                        values = result['Q_energies']
                    elif metric == 'frequency':
                        values = result['frequency']

                    ax.plot(result['timestamps'], values,
                            color=color, linestyle=style,
                            linewidth=1.5, alpha=0.95, label=label_str)

            if metric == 'probability':
                ax.axhline(CONF_THRESH, color='black', linestyle='--', linewidth=1.5,
                           label=f'Threshold = {CONF_THRESH}')
                ax.set_ylim(-0.05, 1.1)
            elif metric in ['energy', 'Q_energies']:
                ax.axhline(MIN_ENERGY, color='black', linestyle='--', linewidth=1.5,
                           label=f'Min = {MIN_ENERGY}')
                ax.axhline(MAX_ENERGY, color='black', linestyle='--', linewidth=1.5,
                           label=f'Max = {MAX_ENERGY}')
            elif metric == 'frequency':
                ax.axhline(MIN_FREQ, color='black', linestyle='--', linewidth=1.5,
                           label=f'Min = {MIN_FREQ}')
                ax.axhline(MAX_FREQ, color='black', linestyle='--', linewidth=1.5,
                           label=f'Max = {MAX_FREQ}')

            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.grid(True, alpha=0.3)

        # --- GT vs Prediction subplot ---
        ax_gt = axes[-1]
        for subject in subjects:
            subject_results = [r for r in activity_results
                               if r['subject'].capitalize() == subject.capitalize()]
            for result in subject_results:
                subj_cap = result['subject'].capitalize()
                color    = subject_colors.get(subj_cap, '#333333')
                style    = wrist_styles[result['wrist']]

                ax_gt.fill_between(result['raw_timestamps'], 0, result['raw_gt'],
                                   step='post', alpha=0.25, color=color,
                                   label=f"{subj_cap} | {result['wrist']} | GT")
                ax_gt.step(result['timestamps'], result['y_pred'] + 0.05,
                           where='post', color=color, linestyle=style, linewidth=2.5,
                           label=f"{subj_cap} | {result['wrist']} | Pred")

        ax_gt.set_ylabel('GT / Prediction', fontsize=12)
        ax_gt.set_ylim(-0.1, 1.15)
        ax_gt.grid(True, alpha=0.3)
        ax_gt.set_xlabel('Time (s)', fontsize=12)

        if not has_data:
            plt.close(fig)
            continue

        # Deduplicated legend
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        seen = set()
        uh, ul = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l)
                uh.append(h)
                ul.append(l)

        fig.legend(uh, ul, loc='center left', bbox_to_anchor=(0.87, 0.5),
                   fontsize=10, title='Subject | Wrist', frameon=True, ncol=1)
        plt.tight_layout(rect=[0, 0, 0.86, 0.95])

        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, f'activity_{activity_type}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    model = load_finetuned_model(WEIGHTS_PATH).to(device)

    results = []

    # # QSENSE DATA
    # for folder in sorted(os.listdir(DATASET_PATH)):
    #     folder_path = os.path.join(DATASET_PATH, folder)
    #     if not os.path.isdir(folder_path):
    #         continue

    #     parts         = folder.split('_')
    #     activity_type = '_'.join(parts[:-1]) if len(parts) > 1 else folder
    #     subject       = parts[-1] if len(parts) > 1 else 'Unknown'

    #     for fname, wrist in [('s1_1RW.txt', 'right'), ('s2_2LW.txt', 'left')]:
    #         fpath = os.path.join(folder_path, fname)
    #         if not os.path.exists(fpath):
    #             continue

    #         try:
    #             df = load_data(fpath)

    #             wins, engs, frqs, acts, tmstps, Q_energies, raw_times, raw_gt = \
    #                 prepare_windows(df)

    #             if len(wins) == 0:
    #                 print(f"  Skipping {folder}/{fname}: no valid windows")
    #                 continue

    #             with torch.no_grad():
    #                 logits = model(wins.to(device))
    #                 probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    #             y_pred = (probs > CONF_THRESH).astype(int)
    #             y_true = acts  # window-level GT from prepare_windows

    #             # Metrics
    #             if y_true.sum() == 0:
    #                 precision = recall = f1 = 0.0
    #             else:
    #                 precision, recall, f1, _ = precision_recall_fscore_support(
    #                     y_true, y_pred, labels=[1], average='binary', zero_division=0)
    #             accuracy = accuracy_score(y_true, y_pred)
    #             cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    #             print(f"{folder} | {wrist.upper()} | "
    #                   f"Prec={precision:.3f}  Rec={recall:.3f}  "
    #                   f"F1={f1:.3f}  Acc={accuracy:.3f}")

    #             results.append({
    #                 'subject':       subject,
    #                 'folder':        folder,
    #                 'activity_type': activity_type,
    #                 'wrist':         wrist,
    #                 'raw_timestamps': raw_times,
    #                 'raw_gt':        raw_gt,
    #                 'timestamps':    tmstps,
    #                 'y_true':        y_true,
    #                 'y_pred':        y_pred,
    #                 'probability':   probs,
    #                 'energy':        engs,
    #                 'frequency':     frqs,
    #                 'Q_energies':    Q_energies,
    #                 'precision':     precision,
    #                 'recall':        recall,
    #                 'f1':            f1,
    #                 'accuracy':      accuracy,
    #                 'confusion_matrix': cm.tolist()
    #             })

    #         except Exception as e:
    #             print(f"  Error in {folder}/{fname}: {e}")

    ## FREE-LIVING DATA
    for fname in sorted(os.listdir(DATASET_PATH)):
        if not fname.endswith('_annotated.csv'):
            continue

        fpath   = os.path.join(DATASET_PATH, fname)
        parts   = fname.replace('_annotated.csv', '').split('_')
        subject = parts[1] if len(parts) > 1 else fname   # e.g. sub1
        activity_type = subject  # one plot per subject
        wrist   = 'left'  # single wrist device

        try:
            # Free-Living specific loading
            raw = pd.read_csv(fpath)
            raw['datetime'] = pd.to_datetime(
                raw['time'], format='%m/%d/%Y %H:%M:%S.%f', errors='coerce'
            )
            raw = raw.dropna(subset=['datetime']).reset_index(drop=True)
            raw['label'] = pd.to_numeric(raw['Label'], errors='coerce').fillna(0).astype(int)

            # Clean timestamps (no firmware artifacts but run for consistency)
            running_max = raw['datetime'].iloc[0]
            keep = []
            for t in raw['datetime']:
                if t < running_max:
                    keep.append(False)
                else:
                    keep.append(True)
                    running_max = t
            raw = raw[keep].reset_index(drop=True)
            raw = raw.sort_values('datetime').reset_index(drop=True)
            raw = raw.drop_duplicates(subset='datetime', keep='first').reset_index(drop=True)
            raw['time_sec'] = (raw['datetime'] - raw['datetime'].iloc[0]).dt.total_seconds()

            df = pd.DataFrame({
                'time_sec': raw['time_sec'].values,
                'accX':     pd.to_numeric(raw['ax'], errors='coerce'),
                'accY':     pd.to_numeric(raw['ay'], errors='coerce'),
                'accZ':     pd.to_numeric(raw['az'], errors='coerce'),
                'gt':       raw['label'].values,
                'energy':   np.zeros(len(raw))  # no energy column in Free-Living
            })

            wins, engs, frqs, acts, tmstps, Q_energies, raw_times, raw_gt = \
                prepare_windows(df)

            if len(wins) == 0:
                print(f"  Skipping {fname}: no valid windows")
                continue

            with torch.no_grad():
                logits = model(wins.to(device))
                probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            y_pred = (probs > CONF_THRESH).astype(int)
            y_true = acts

            if y_true.sum() == 0:
                precision = recall = f1 = 0.0
            else:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, labels=[1], average='binary', zero_division=0)
            accuracy = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            print(f"{fname} | "
                  f"Prec={precision:.3f}  Rec={recall:.3f}  "
                  f"F1={f1:.3f}  Acc={accuracy:.3f}")

            results.append({
                'subject':        subject,
                'folder':         fname,
                'activity_type':  activity_type,
                'wrist':          wrist,
                'raw_timestamps': raw_times,
                'raw_gt':         raw_gt,
                'timestamps':     tmstps,
                'y_true':         y_true,
                'y_pred':         y_pred,
                'probability':    probs,
                'energy':         engs,
                'frequency':      frqs,
                'Q_energies':     Q_energies,
                'precision':      precision,
                'recall':         recall,
                'f1':             f1,
                'accuracy':       accuracy,
                'confusion_matrix': cm.tolist()
            })

        except Exception as e:
            print(f"  Error in {fname}: {e}")

    # --- Summary ---
    if results:
        # Global F1 from pooled counts
        
        
        total_tp = total_fp = total_fn = total_tn = 0
        metrics_rows = []
        
        for r in results:
            c = np.array(r['confusion_matrix'])  # [[TN, FP], [FN, TP]]
            total_tn += c[0, 0]
            total_fp += c[0, 1]
            total_fn += c[1, 0]
            total_tp += c[1, 1]
            metrics_rows.append({
                'subject':   r['subject'].capitalize(),
                'activity':  r['activity_type'],
                'wrist':     r['wrist'],
                'precision': r['precision'],
                'recall':    r['recall'],
                'f1':        r['f1'],
                'accuracy':  r['accuracy']
            })

        g_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        g_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        g_f1   = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0
        g_acc  = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)

        print(f'\n=== GLOBAL DATASET TOTALS ===')
        print(f'Precision: {g_prec:.3f} | Recall: {g_rec:.3f} | F1: {g_f1:.3f} | Acc: {g_acc:.3f}')

        # --- Pooled metrics by group ---
        def pooled_metrics(rows, group_key):
            groups = {}
            for r in rows:
                key = r[group_key]
                if key not in groups:
                    groups[key] = [0, 0, 0, 0]  # tp, fp, fn, tn
                c = np.array(r['confusion_matrix'])
                groups[key][0] += c[1, 1]  # tp
                groups[key][1] += c[0, 1]  # fp
                groups[key][2] += c[1, 0]  # fn
                groups[key][3] += c[0, 0]  # tn

            print(f"\nBy {group_key}:")
            print(f"  {'Group':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}")
            print(f"  {'-'*65}")
            for key, (tp, fp, fn, tn) in sorted(groups.items()):
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0
                acc  = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) > 0 else 0
                print(f"  {key:<25} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {acc:>10.3f}")

        # Add group keys to results for easy lookup
        results_with_groups = []
        for r in results:
            results_with_groups.append({
                'subject':          r['subject'].capitalize(),
                'activity':         r['activity_type'],
                'wrist':            r['wrist'],
                'confusion_matrix': r['confusion_matrix']
            })

        pooled_metrics(results_with_groups, 'wrist')
        pooled_metrics(results_with_groups, 'activity')
        pooled_metrics(results_with_groups, 'subject')

        # --- PLOTTING ---
        dataset_name = os.path.basename(DATASET_PATH)
        plots_dir = os.path.join(PLOTS_DIR, dataset_name, 'strokenet')
        os.makedirs(plots_dir, exist_ok=True)

        df_metrics = pd.DataFrame(metrics_rows)
        dataset_name_safe = os.path.basename(DATASET_PATH)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        results_csv = os.path.join(RESULTS_DIR, f'strokenet_{dataset_name_safe}_metrics.csv')
        df_metrics.to_csv(results_csv, index=False)
        print(f"\nSaved metrics: {results_csv}")

        subjects = sorted(set(r['subject'].capitalize() for r in results))
        is_freeliving = (DATASET_PATH == FREELIVING_PATH)
        plot_metrics = ['probability', 'frequency'] if is_freeliving else ['probability', 'energy', 'Q_energies', 'frequency']
        plot_per_activity(results, subjects, plot_metrics, plots_dir)


if __name__ == '__main__':
    main()