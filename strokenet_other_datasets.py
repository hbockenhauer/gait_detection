import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from eldernet_WearGait import load_weargait_data, detect_sampling_rate
from scipy import signal
import glob
from eldernet_ADL import extract_subject_id_and_timestamp

# --- CONFIGURATION ---
ADL_PATH = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\HMP_Dataset'
WISDM_PATH    = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\wisdm-dataset\raw\watch\accel'
WEARGAIT_PD_PATH = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\WearGait-PD'
WEARGAIT_CTRL_PATH = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\WearGait-Ctrl'
BIOCLITE_PATH = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\data_6activities_plain.mat'
WEIGHTS_PATH  = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\eldernet_finetuned.pth'
PLOTS_DIR =  r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\StrokeNet_Plots'
REPO_NAME     = 'yonbrand/ElderNet'

WINDOW_SIZE   = 100    # 2s at 50Hz
STEP_SIZE     = 50     # 1s stride
GAP_THRESHOLD = 0.1
CONF_THRESH   = 0.5

WISDM_GAIT_CODES  = {'A', 'C'}   # Walk, Stairs
WEARGAIT_PATTERNS = ['walk', 'jog', 'run', 'stair', 'climb', 'freewalk', 'gait']

ACTIVITY_MAP = {
    'A':'Walk', 'B':'Jog', 'C':'Stairs', 'D':'Sit', 'E':'Stand',
    'F':'Type', 'G':'Teeth', 'H':'Soup', 'I':'Chips', 'J':'Pasta',
    'K':'Drink', 'L':'Sandwich', 'M':'Kicking', 'O':'Catch',
    'P':'Dribbling', 'Q':'Writing', 'R':'Clapping', 'S':'Folding'
}


# ============================================================
# MODEL LOADING — identical to training
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
    model.feature_extractor.layer5 = nn.Sequential(
        *[child for idx, child in enumerate(layer5.children()) if idx != 3]
    )
    return model

def load_finetuned_model(weights_path):
    model = torch.hub.load(REPO_NAME, 'eldernet_ft', trust_repo=True)
    model = fix_circular_padding(model)
    model = remove_last_downsample(model)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    print(f"Loaded finetuned weights from {weights_path}")
    return model


# ============================================================
# WINDOWING — gap-aware, used for WearGait (has timestamps)
# ============================================================

def extract_windows_with_gaps(times, acc_data, labels):
    """For data with real timestamps — respects gaps."""
    dt      = np.diff(times)
    gap_idx = np.where(dt > GAP_THRESHOLD)[0] + 1
    bounds  = np.concatenate([[0], gap_idx, [len(times)]])

    windows, targets, win_times = [], [], []

    for k in range(len(bounds) - 1):
        seg_start = bounds[k]
        seg_end   = bounds[k + 1]
        if (seg_end - seg_start) < WINDOW_SIZE:
            continue
        for i in range(seg_start + WINDOW_SIZE, seg_end, STEP_SIZE):
            win     = acc_data[i - WINDOW_SIZE:i]
            lab_win = labels[i - WINDOW_SIZE:i]
            windows.append(win.T)                          # (3, 100)
            targets.append(int(np.mean(lab_win) > 0.5))
            win_times.append(times[i - 1])

    if len(windows) == 0:
        return None, None, None
    return (np.array(windows, dtype=np.float32),
            np.array(targets),
            np.array(win_times))


def extract_windows_no_gaps(acc_data, labels):
    """For WISDM — no real timestamps, treat as continuous."""
    windows, targets = [], []
    for i in range(WINDOW_SIZE, len(acc_data), STEP_SIZE):
        win     = acc_data[i - WINDOW_SIZE:i]
        lab_win = labels[i - WINDOW_SIZE:i]
        windows.append(win.T)
        targets.append(int(np.mean(lab_win) > 0.5))

    if len(windows) == 0:
        return None, None
    return (np.array(windows, dtype=np.float32),
            np.array(targets))


# ============================================================
# INFERENCE
# ============================================================

def run_inference(model, windows_np, device):
    wins  = torch.FloatTensor(windows_np).to(device)
    with torch.no_grad():
        logits = model(wins)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return probs


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if y_true.sum() == 0:
        return 0.0, 0.0, 0.0, accuracy_score(y_true, y_pred), cm
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average='binary', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return prec, rec, f1, acc, cm


# ============================================================
# WISDM LOADER
# ============================================================

def evaluate_wisdm(model, device):
    print("\n" + "="*60)
    print("EVALUATING: WISDM (watch accelerometer)")
    print("="*60)

    files = sorted([f for f in os.listdir(WISDM_PATH) if f.endswith('.txt')])
    print(f"Found {len(files)} subject files")

    results = []
    total_tp = total_fp = total_fn = total_tn = 0

    for fname in files:
        fpath   = os.path.join(WISDM_PATH, fname)
        subject = fname.replace('.txt', '')

        try:
            df = pd.read_csv(fpath, header=None,
                             names=['user', 'activity', 'ts', 'x', 'y', 'z'])
            df['z'] = df['z'].astype(str).str.replace(';', '').astype(float)
            df = df.dropna(subset=['x', 'y', 'z'])

            # WISDM is 20Hz — resample to 50Hz to match your model's training data
            # (your model was trained on 50Hz QSense + Free-Living data)
            acc_raw    = df[['x', 'y', 'z']].values
            labels_raw = df['activity'].values

            original_fs = 20.0
            target_fs   = 50.0
            n_new       = int(len(acc_raw) * (target_fs / original_fs))

            from scipy import signal as sp_signal
            acc_resampled = sp_signal.resample(acc_raw, n_new)

            # Resample labels by nearest-neighbour
            idx_map        = np.linspace(0, len(labels_raw) - 1, n_new).astype(int)
            labels_resampled = labels_raw[idx_map]

            # Binary gait labels
            y_binary = np.array([
                1 if l in WISDM_GAIT_CODES else 0
                for l in labels_resampled
            ])

            wins_np, y_true = extract_windows_no_gaps(acc_resampled, y_binary)
            if wins_np is None:
                continue

            probs  = run_inference(model, wins_np, device)
            y_pred = (probs > CONF_THRESH).astype(int)

            prec, rec, f1, acc, cm = compute_metrics(y_true, y_pred)

            total_tn += cm[0, 0]; total_fp += cm[0, 1]
            total_fn += cm[1, 0]; total_tp += cm[1, 1]

            print(f"  {subject:<20} | Prec={prec:.3f}  Rec={rec:.3f}  "
                  f"F1={f1:.3f}  Acc={acc:.3f}  "
                  f"[gait={y_true.sum()}/{len(y_true)} windows]")

            results.append({
                'subject': subject, 'dataset': 'WISDM',
                'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc,
                'confusion_matrix': cm.tolist(),
                'probs': probs, 'y_true': y_true, 'y_pred': y_pred
            })

        except Exception as e:
            print(f"  Error in {fname}: {e}")

    # Global metrics
    g_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    g_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    g_f1   = 2*g_prec*g_rec / (g_prec+g_rec) if (g_prec+g_rec) > 0 else 0
    g_acc  = (total_tp+total_tn) / (total_tp+total_tn+total_fp+total_fn)
    print(f"\nWISDM GLOBAL: Prec={g_prec:.3f} | Rec={g_rec:.3f} | "
          f"F1={g_f1:.3f} | Acc={g_acc:.3f}")

    return results, {'precision': g_prec, 'recall': g_rec, 'f1': g_f1, 'accuracy': g_acc}


# ============================================================
# WEARGAIT LOADER
# ============================================================

def evaluate_weargait(model, device):
    print("\n" + "="*60)
    print("EVALUATING: WearGait-PD")
    print("="*60)

    csv_PD_files = sorted(glob.glob(os.path.join(WEARGAIT_PD_PATH, '**', '*.csv'),
                                  recursive=True)
                       if False else  # placeholder — adjust to your folder structure
                       [os.path.join(WEARGAIT_PD_PATH, f)
                        for f in os.listdir(WEARGAIT_PD_PATH)
                        if f.endswith('.csv')])

    print(f"Found {len(csv_PD_files)} CSV files")

    csv_CTRL_files = sorted(glob.glob(os.path.join(WEARGAIT_CTRL_PATH, '**', '*.csv'),
                                  recursive=True)
                       if False else  # placeholder — adjust to your folder structure
                       [os.path.join(WEARGAIT_CTRL_PATH, f)
                        for f in os.listdir(WEARGAIT_CTRL_PATH)
                        if f.endswith('.csv')])

    print(f"Found {len(csv_CTRL_files)} CSV files")

    csv_files = csv_PD_files + csv_CTRL_files

    results = []
    total_tp = total_fp = total_fn = total_tn = 0

    for fpath in csv_files:
        fname   = os.path.basename(fpath)
        subject = fname.replace('.csv', '')

        for wrist in ['right', 'left']:
            try:
                # Reuse your existing WearGait loader
                df = load_weargait_data(fpath, wrist=wrist)

                # Detect sampling rate from timestamps
                original_fs = detect_sampling_rate(df['time'].values)

                # Resample to 50Hz if needed
                if abs(original_fs - 50.0) > 1.0:
                    t     = df['time'].values
                    n_new = int((t[-1] - t[0]) * 50.0) + 1
                    new_t = np.linspace(t[0], t[-1], n_new)
                    acc_resampled = np.column_stack([
                        np.interp(new_t, t, df['acc_x'].values),
                        np.interp(new_t, t, df['acc_y'].values),
                        np.interp(new_t, t, df['acc_z'].values)
                    ])
                    # Resample activity labels by nearest neighbour
                    idx_map = np.linspace(0, len(df) - 1, n_new).astype(int)
                    activities_resampled = df['activity'].values[idx_map]
                    times_resampled = new_t
                else:
                    acc_resampled        = df[['acc_x', 'acc_y', 'acc_z']].values
                    activities_resampled = df['activity'].values
                    times_resampled      = df['time'].values

                # Binary gait labels
                y_binary = np.array([
                    1 if any(p in str(a).lower() for p in WEARGAIT_PATTERNS) else 0
                    for a in activities_resampled
                ])

                wins_np, y_true, win_times = extract_windows_with_gaps(
                    times_resampled, acc_resampled, y_binary
                )
                if wins_np is None:
                    continue

                probs  = run_inference(model, wins_np, device)
                y_pred = (probs > CONF_THRESH).astype(int)

                prec, rec, f1, acc, cm = compute_metrics(y_true, y_pred)

                total_tn += cm[0, 0]; total_fp += cm[0, 1]
                total_fn += cm[1, 0]; total_tp += cm[1, 1]

                print(f"  {subject:<20} {wrist:<6} | Prec={prec:.3f}  "
                      f"Rec={rec:.3f}  F1={f1:.3f}  Acc={acc:.3f}  "
                      f"[gait={y_true.sum()}/{len(y_true)} windows]")

                results.append({
                    'subject': subject, 'dataset': 'WearGait', 'wrist': wrist,
                    'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc,
                    'confusion_matrix': cm.tolist(),
                    'probs': probs, 'y_true': y_true, 'y_pred': y_pred,
                    'win_times': win_times
                })

            except Exception as e:
                print(f"  Error in {fname} ({wrist}): {e}")

    g_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    g_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    g_f1   = 2*g_prec*g_rec / (g_prec+g_rec) if (g_prec+g_rec) > 0 else 0
    g_acc  = (total_tp+total_tn) / (total_tp+total_tn+total_fp+total_fn)
    print(f"\nWearGait GLOBAL: Prec={g_prec:.3f} | Rec={g_rec:.3f} | "
          f"F1={g_f1:.3f} | Acc={g_acc:.3f}")

    return results, {'precision': g_prec, 'recall': g_rec, 'f1': g_f1, 'accuracy': g_acc}

def evaluate_hmp(model, device):
    print("\n" + "="*60)
    print("EVALUATING: HMP Dataset (ADL)")
    print("="*60)

    GAIT_CLASSES = {'Walk', 'Climb_stairs', 'Descend_stairs'}
    HMP_WINDOW_SIZE = 100   # 2s at 50Hz
    TARGET_FS = 50.0
    SOURCE_FS = 32.0

    categories = sorted([d for d in os.listdir(ADL_PATH)
                         if os.path.isdir(os.path.join(ADL_PATH, d))
                         and "_MODEL" not in d])

    # Group files by subject
    subject_files = {}
    for cat in categories:
        for f in glob.glob(os.path.join(ADL_PATH, cat, '*.txt')):
            base = os.path.basename(f)
            subj_id, ts = extract_subject_id_and_timestamp(base)
            if subj_id and ts:
                subject_files.setdefault(subj_id, []).append((ts, cat, f))

    print(f"Found {len(subject_files)} subjects across {len(categories)} activities")

    results = []
    total_tp = total_fp = total_fn = total_tn = 0

    for subj_id in sorted(subject_files.keys()):
        files_sorted = sorted(subject_files[subj_id], key=lambda x: x[0])
        all_data, all_labels = [], []

        for ts, cat, f in files_sorted:
            try:
                raw = np.loadtxt(f)
                if len(raw) < 150:
                    continue

                # Unit conversion + lowpass filter (same as original)
                data = (raw.astype(float) - 32.0) * (1.5 / 32.0)
                nyq  = 0.5 * SOURCE_FS
                b, a = signal.butter(4, 10.0 / nyq, btype='low')
                data = signal.filtfilt(b, a, data, axis=0)

                # Resample 32Hz -> 50Hz (changed from 30Hz)
                new_len = int(len(data) * (TARGET_FS / SOURCE_FS))
                data    = signal.resample(data, new_len)

                label = 1 if cat in GAIT_CLASSES else 0
                all_data.append(data)
                all_labels.extend([label] * len(data))

            except Exception as e:
                continue

        if not all_data or len(all_data) == 0:
            continue

        concat_data   = np.vstack(all_data)
        concat_labels = np.array(all_labels)

        if len(concat_data) < HMP_WINDOW_SIZE:
            continue

        # Window — no real timestamps so treat as continuous
        windows, y_true = [], []
        for i in range(HMP_WINDOW_SIZE, len(concat_data), STEP_SIZE):
            win     = concat_data[i - HMP_WINDOW_SIZE:i]
            lab_win = concat_labels[i - HMP_WINDOW_SIZE:i]
            windows.append(win.T)                           # (3, 100)
            y_true.append(int(np.mean(lab_win) > 0.5))

        if len(windows) == 0:
            continue

        wins_np = np.array(windows, dtype=np.float32)
        probs   = run_inference(model, wins_np, device)
        y_true  = np.array(y_true)
        y_pred  = (probs > CONF_THRESH).astype(int)

        prec, rec, f1, acc, cm = compute_metrics(y_true, y_pred)

        total_tn += cm[0, 0]; total_fp += cm[0, 1]
        total_fn += cm[1, 0]; total_tp += cm[1, 1]

        print(f"  {subj_id:<10} | Prec={prec:.3f}  Rec={rec:.3f}  "
              f"F1={f1:.3f}  Acc={acc:.3f}  "
              f"[gait={y_true.sum()}/{len(y_true)} windows]")

        results.append({
            'subject': subj_id, 'dataset': 'HMP',
            'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc,
            'confusion_matrix': cm.tolist(),
            'probs':  probs,        # add this
            'y_true': y_true,       # add this
            'y_pred': y_pred       # add this
        })

    g_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    g_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    g_f1   = 2*g_prec*g_rec / (g_prec+g_rec) if (g_prec+g_rec) > 0 else 0
    g_acc  = (total_tp+total_tn) / (total_tp+total_tn+total_fp+total_fn)
    print(f"\nHMP GLOBAL: Prec={g_prec:.3f} | Rec={g_rec:.3f} | "
          f"F1={g_f1:.3f} | Acc={g_acc:.3f}")

    return results, {'precision': g_prec, 'recall': g_rec, 'f1': g_f1, 'accuracy': g_acc}


BIOCLITE_GAIT_LABEL = 6
BIOCLITE_LABEL_MAP  = {
    0: 'Transition', 1: 'Spiral',  2: 'Typing',
    3: 'Resting',    4: 'Beating', 5: 'Brushing', 6: 'Walking'
}

def evaluate_bioclite(model, device):
    print("\n" + "="*60)
    print("EVALUATING: BIOCLITE Free-Living Activities")
    print("="*60)

    import scipy.io
    from collections import defaultdict

    mat  = scipy.io.loadmat(BIOCLITE_PATH, squeeze_me=True)
    Data = mat['Data_plain']
    print(f"Found {len(Data)} subjects")

    results   = []
    total_tp  = total_fp = total_fn = total_tn = 0

    for i, trial in enumerate(Data):
        try:
            ts_ms       = trial[:, 0].astype(float)
            acc         = trial[:, 1:4].astype(float)
            participant = int(trial[0, 7])
            act_labels  = trial[:, 8].astype(int)

            times    = (ts_ms - ts_ms[0]) / 1000.0
            y_binary = (act_labels == BIOCLITE_GAIT_LABEL).astype(int)

            print(f"  Trial {i+1:02d} P{participant:02d} | "
                  f"activities: {np.unique(act_labels)}, "
                  f"gait samples: {y_binary.sum()}/{len(y_binary)}")

            # Gap-aware windowing (already 50Hz — no resampling needed)
            dt      = np.diff(times)
            gap_idx = np.where(dt > GAP_THRESHOLD)[0] + 1
            bounds  = np.concatenate([[0], gap_idx, [len(times)]])

            windows, y_true, win_times, win_activities = [], [], [], []

            for k in range(len(bounds) - 1):
                seg_start = bounds[k]
                seg_end   = bounds[k + 1]
                if (seg_end - seg_start) < WINDOW_SIZE:
                    continue
                for wi in range(seg_start + WINDOW_SIZE, seg_end, STEP_SIZE):
                    win     = acc[wi - WINDOW_SIZE:wi]
                    lab_win = y_binary[wi - WINDOW_SIZE:wi]
                    act_win = act_labels[wi - WINDOW_SIZE:wi]

                    windows.append(win.T)
                    y_true.append(int(np.mean(lab_win) > 0.5))
                    win_times.append(times[wi - 1])

                    # Majority activity label for this window
                    unique, counts = np.unique(act_win, return_counts=True)
                    win_activities.append(int(unique[np.argmax(counts)]))

            if len(windows) == 0:
                print(f"  Trial {i+1:02d} P{participant:02d} | skipped (no valid windows)")
                continue

            wins_np      = np.array(windows, dtype=np.float32)
            y_true       = np.array(y_true)
            win_times    = np.array(win_times)
            win_activities = np.array(win_activities)

            probs  = run_inference(model, wins_np, device)
            y_pred = (probs > CONF_THRESH).astype(int)

            prec, rec, f1, acc_score, cm = compute_metrics(y_true, y_pred)

            total_tn += cm[0, 0]; total_fp += cm[0, 1]
            total_fn += cm[1, 0]; total_tp += cm[1, 1]

            print(f"  Trial {i+1:02d} P{participant:02d} | "
                  f"Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}  Acc={acc_score:.3f}  "
                  f"[gait={y_true.sum()}/{len(y_true)} windows]")

            results.append({
                'subject':        f'P{participant:02d}',
                'dataset':        'BIOCLITE',
                'activity':       'FreeLiving',
                'wrist':          'preferred',
                'precision':      prec,
                'recall':         rec,
                'f1':             f1,
                'accuracy':       acc_score,
                'confusion_matrix': cm.tolist(),
                'probs':          probs,
                'y_true':         y_true,
                'y_pred':         y_pred,
                'win_times':      win_times,
                'win_activities': win_activities
            })

        except Exception as e:
            print(f"  Error in trial {i+1}: {e}")

    # --- Global metrics ---
    g_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    g_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    g_f1   = 2*g_prec*g_rec / (g_prec+g_rec) if (g_prec+g_rec) > 0 else 0
    g_acc  = (total_tp+total_tn) / (total_tp+total_tn+total_fp+total_fn) \
             if (total_tp+total_tn+total_fp+total_fn) > 0 else 0

    print(f"\nBIOCLITE GLOBAL: Prec={g_prec:.3f} | Rec={g_rec:.3f} | "
          f"F1={g_f1:.3f} | Acc={g_acc:.3f}")

    # --- Breakdown by activity type (pooled across all subjects) ---
    by_act = defaultdict(lambda: [0, 0, 0, 0])
    for r in results:
        yt = r['y_true']
        yp = r['y_pred']
        wa = r['win_activities']
        for act_idx, act_name in BIOCLITE_LABEL_MAP.items():
            mask = (wa == act_idx)
            if mask.sum() == 0:
                continue
            by_act[act_name][0] += int(((yp[mask]==1) & (yt[mask]==1)).sum())  # tp
            by_act[act_name][1] += int(((yp[mask]==1) & (yt[mask]==0)).sum())  # fp
            by_act[act_name][2] += int(((yp[mask]==0) & (yt[mask]==1)).sum())  # fn
            by_act[act_name][3] += int(((yp[mask]==0) & (yt[mask]==0)).sum())  # tn

    print("\nBy activity (pooled across all subjects):")
    print(f"  {'Activity':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} "
          f"{'Accuracy':>10} {'Windows':>10}")
    print(f"  {'-'*64}")
    for act_name, (tp, fp, fn, tn) in sorted(by_act.items()):
        p = tp/(tp+fp) if (tp+fp) > 0 else 0
        r = tp/(tp+fn) if (tp+fn) > 0 else 0
        f = 2*p*r/(p+r) if (p+r) > 0 else 0
        a = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) > 0 else 0
        print(f"  {act_name:<14} {p:>10.3f} {r:>10.3f} {f:>10.3f} "
              f"{a:>10.3f} {tp+fp+fn+tn:>10}")

    return results, {'precision': g_prec, 'recall': g_rec,
                     'f1': g_f1,          'accuracy': g_acc}
def plot_subject_timeline(results, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    for r in results:
        dataset  = r['dataset']
        subject  = r['subject']
        wrist    = r.get('wrist', 'right')
        probs    = r['probs']
        y_pred   = r['y_pred']
        y_true   = r['y_true']

        # X-axis: use timestamps if available, else window indices
        if 'win_times' in r and r['win_times'] is not None:
            x      = r['win_times']
            xlabel = 'Time (s)'
        else:
            x      = np.arange(len(probs))
            xlabel = 'Window index'

        fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
        fig.suptitle(f"{dataset} — {subject} ({wrist})", fontsize=14, fontweight='bold')

        # --- Probability ---
        axes[0].plot(x, probs, color='steelblue', linewidth=1.5, label='Gait probability')
        axes[0].axhline(CONF_THRESH, color='black', linestyle='--', linewidth=1,
                        label=f'Threshold = {CONF_THRESH}')
        axes[0].fill_between(x, 0, probs, alpha=0.15, color='steelblue')
        axes[0].set_ylim(-0.05, 1.1)
        axes[0].set_ylabel('Probability', fontsize=11)
        axes[0].legend(fontsize=9, loc='upper right')
        axes[0].grid(True, alpha=0.3)

        # --- Prediction vs Ground Truth ---
        axes[1].step(x, y_true, where='post', color='green',
                     linewidth=2, alpha=0.7, label='Ground truth')
        axes[1].step(x, y_pred + 0.05, where='post', color='crimson',
                     linewidth=1.5, linestyle='--', label='Prediction')
        axes[1].fill_between(x, 0, y_true, step='post',
                             alpha=0.15, color='green')
        axes[1].set_ylim(-0.15, 1.2)
        axes[1].set_ylabel('Gait (0/1)', fontsize=11)
        axes[1].legend(fontsize=9, loc='upper right')
        axes[1].grid(True, alpha=0.3)

        # --- Agreement / Error ---
        correct = (y_pred == y_true).astype(int)
        tp_mask = (y_pred == 1) & (y_true == 1)
        fp_mask = (y_pred == 1) & (y_true == 0)
        fn_mask = (y_pred == 0) & (y_true == 1)
        tn_mask = (y_pred == 0) & (y_true == 0)

        axes[2].fill_between(x, 0, tp_mask.astype(float), step='post',
                             color='green',  alpha=0.6, label='TP')
        axes[2].fill_between(x, 0, tn_mask.astype(float), step='post',
                             color='lightgrey', alpha=0.6, label='TN')
        axes[2].fill_between(x, 0, fp_mask.astype(float), step='post',
                             color='orange', alpha=0.8, label='FP')
        axes[2].fill_between(x, 0, fn_mask.astype(float), step='post',
                             color='crimson', alpha=0.8, label='FN')
        axes[2].set_ylim(-0.1, 1.2)
        axes[2].set_ylabel('Classification', fontsize=11)
        axes[2].set_xlabel(xlabel, fontsize=11)
        axes[2].legend(fontsize=9, loc='upper right', ncol=4)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        safe_subject = subject.replace('/', '_').replace(' ', '_')
        save_path = os.path.join(plots_dir, f'{dataset}_{safe_subject}_{wrist}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    import glob
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    model = load_finetuned_model(WEIGHTS_PATH).to(device)

    wisdm_results,    wisdm_global    = evaluate_wisdm(model, device)
    weargait_results, weargait_global = evaluate_weargait(model, device)
    hmp_results, hmp_global = evaluate_hmp(model, device)
    bioclite_results, bioclite_global = evaluate_bioclite(model, device)

    all_results = wisdm_results + weargait_results + hmp_results + bioclite_results 
    plot_subject_timeline(all_results, PLOTS_DIR)

    # Save combined results
    all_rows = []
    for r in wisdm_results + weargait_results + hmp_results + bioclite_results:
        all_rows.append({
            'dataset':   r['dataset'],
            'subject':   r['subject'],
            'wrist':     r.get('wrist', 'right'),
            'precision': r['precision'],
            'recall':    r['recall'],
            'f1':        r['f1'],
            'accuracy':  r['accuracy']
        })
    pd.DataFrame(all_rows).to_csv('cross_dataset_results.csv', index=False)
    print("\nSaved: cross_dataset_results.csv")


if __name__ == '__main__':
    main()