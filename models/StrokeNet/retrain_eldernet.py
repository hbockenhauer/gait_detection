import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.paths import QSENSE_MIXED, QSENSE_EDGE, FREELIVING_PATH
from utils.hub_utils import safe_hub_load

QSENSE_MIXED_PATH = QSENSE_MIXED
QSENSE_EDGE_PATH = QSENSE_EDGE
SAVE_PATH = os.path.join(SCRIPT_DIR, 'eldernet_finetuned.pth')

WINDOW_SIZE    = 100    # 2s at 50Hz
STEP_SIZE      = 50     # 1s stride
FS             = 50
GAP_THRESHOLD  = 0.1    # seconds — gaps larger than this break windowing

# ============================================================
# 1. DATA CLEANING
# ============================================================

def clean_timestamps(df, datetime_col='datetime', fs=50):
    """Apply the same 4-step cleaning pipeline as the MATLAB scripts."""
    timestamps = df[datetime_col].values

    # Step 0: Remove backwards-jump blocks (firmware buffer re-dumps)
    running_max = timestamps[0]
    keep = []
    for t in timestamps:
        if t < running_max:
            keep.append(False)
        else:
            keep.append(True)
            running_max = t
    df = df[keep].reset_index(drop=True)

    # Step 1: Fix time travelers (>100 day jumps)
    dt = df[datetime_col].diff()
    jump_idx = dt[abs(dt) > pd.Timedelta(days=100)].index
    for idx in jump_idx:
        false_gap = dt[idx] - pd.Timedelta(seconds=1/fs)
        df.loc[idx:, datetime_col] = df.loc[idx:, datetime_col] - false_gap
        dt = df[datetime_col].diff()

    # Step 2: Sort
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # Step 3: Remove duplicate timestamps
    df = df.drop_duplicates(subset=datetime_col, keep='first').reset_index(drop=True)

    # Step 4: Time vector in seconds
    df['time_sec'] = (df[datetime_col] - df[datetime_col].iloc[0]).dt.total_seconds()

    return df


def extract_windows(df, window_size, step_size, label_col, acc_cols):
    """
    Extract windows that never cross a gap.
    Returns windows (N, 3, window_size), labels (N,), times (N,)
    """
    times    = df['time_sec'].values
    acc_data = df[acc_cols].values
    labels   = df[label_col].values

    # Find segment boundaries at gaps
    dt       = np.diff(times)
    gap_idx  = np.where(dt > GAP_THRESHOLD)[0] + 1
    bounds   = np.concatenate([[0], gap_idx, [len(times)]])

    windows, targets, win_times = [], [], []

    for k in range(len(bounds) - 1):
        seg_start = bounds[k]
        seg_end   = bounds[k + 1]

        # Skip segments too short for even one window
        if (seg_end - seg_start) < window_size:
            continue

        for i in range(seg_start + window_size, seg_end, step_size):
            win = acc_data[i - window_size:i]   # (window_size, 3)
            lab = labels[i - window_size:i]
            windows.append(win.T)               # (3, window_size) for PyTorch conv1d
            targets.append(int(np.mean(lab) > 0.5))
            win_times.append(times[i - 1])

    if len(windows) == 0:
        return None, None, None

    return np.array(windows, dtype=np.float32), np.array(targets), np.array(win_times)


# ============================================================
# 2. DATASET LOADERS
# ============================================================

def load_qsense_dataset(data_path, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    all_windows, all_labels, all_subjects = [], [], []

    for folder in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue

        parts        = folder.split('_')
        activity     = parts[0].lower()
        subject      = parts[-1]  # last part is always the subject name
        folder_label = int(activity in ['walking', 'stairs'])

        for fname, wrist in [('s1_1RW.txt', 'R'), ('s2_2LW.txt', 'L')]:
            fpath = os.path.join(folder_path, fname)
            if not os.path.isfile(fpath):
                continue

            try:
                # Load with header so column names are preserved
                raw = pd.read_csv(fpath, sep=None, engine='python')
                raw['datetime'] = pd.to_datetime(
                    raw.iloc[:, 0].astype(str) + ' ' + raw.iloc[:, 1].astype(str),
                    format='%Y-%m-%d %H:%M:%S.%f', errors='coerce'
                )
                raw = raw.dropna(subset=['datetime']).reset_index(drop=True)

                # Rename accelerometer columns by position (cols 5,6,7 = accX,Y,Z)
                col_names = raw.columns.tolist()
                raw = raw.rename(columns={
                    col_names[5]: 'accX',
                    col_names[6]: 'accY',
                    col_names[7]: 'accZ'
                })

                # Ground truth: named Label column if present, else folder-level
                if 'Label' in raw.columns or 'label' in raw.columns:
                    label_col = 'Label' if 'Label' in raw.columns else 'label'
                    raw['label'] = pd.to_numeric(raw[label_col], errors='coerce').fillna(0).astype(int)
                    source = 'annotated'
                else:
                    raw['label'] = folder_label
                    source = f'folder ({folder_label})'

                # Clean timestamps
                df = clean_timestamps(raw, datetime_col='datetime')

                wins, labs, _ = extract_windows(
                    df, window_size, step_size,
                    label_col='label',
                    acc_cols=['accX', 'accY', 'accZ']
                )

                if wins is None:
                    continue

                all_windows.append(wins)
                all_labels.append(labs)
                all_subjects.extend([f"{folder}_{wrist}"] * len(labs))
                print(f"  QSense {folder}/{fname}: {len(labs)} windows  "
                      f"(gait={labs.sum()}, non-gait={(labs==0).sum()})  [{source}]")

            except Exception as e:
                print(f"  Error in {folder}/{fname}: {e}")

    return (np.concatenate(all_windows),
            np.concatenate(all_labels),
            np.array(all_subjects))


def load_freeliving_dataset(data_path, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    all_windows, all_labels, all_subjects = [], [], []

    for fname in sorted(os.listdir(data_path)):
        if not fname.endswith('_annotated.csv'):
            continue

        subject = fname.split('_')[1]
        fpath   = os.path.join(data_path, fname)

        try:
            raw = pd.read_csv(fpath)
            raw['datetime'] = pd.to_datetime(
                raw['time'],
                format='%m/%d/%Y %H:%M:%S.%f', errors='coerce'
            )
            raw = raw.dropna(subset=['datetime']).reset_index(drop=True)
            raw['label'] = pd.to_numeric(raw['Label'], errors='coerce').fillna(0).astype(int)
            raw = raw.rename(columns={'ax': 'accX', 'ay': 'accY', 'az': 'accZ'})

            # Free-Living has no firmware artifacts so skip Step 0,
            # but still run the full cleaner for consistency
            df = clean_timestamps(raw, datetime_col='datetime')

            wins, labs, _ = extract_windows(
                df, window_size, step_size,
                label_col='label',
                acc_cols=['accX', 'accY', 'accZ']
            )

            if wins is None:
                continue

            all_windows.append(wins)
            all_labels.append(labs)
            all_subjects.extend([subject] * len(labs))
            print(f"  FreeLiving {fname}: {len(labs)} windows  "
                  f"(gait={labs.sum()}, non-gait={(labs==0).sum()})")

        except Exception as e:
            print(f"  Error in {fname}: {e}")

    return (np.concatenate(all_windows),
            np.concatenate(all_labels),
            np.array(all_subjects))


# ============================================================
# 3. DATASET CLASS
# ============================================================

class GaitDataset(Dataset):
    def __init__(self, windows, labels):
        self.X = torch.FloatTensor(windows)
        self.y = torch.LongTensor(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# 4. MODEL ADAPTATION
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
    """
    Remove the Downsample from layer5 so the sequence only halves 4 times.
    100 -> 50 -> 25 -> 12 -> 6  (stops here, kernel size 5 is fine on 6)
    instead of going to 3 which breaks kernel size 5.
    """
    layer5 = model.feature_extractor.layer5

    # Print layer5 so we can see what index Downsample is at
    print("layer5 children:")
    for idx, child in enumerate(layer5.children()):
        print(f"  [{idx}] {child}")

    # Rebuild layer5 without the final Downsample
    # From the architecture printout: layer5 = [Conv1d, BN, ReLU, Downsample]
    # indices                                     0       1   2     3
    new_layer5 = nn.Sequential(
        *[child for idx, child in enumerate(layer5.children()) if idx != 3]
    )
    model.feature_extractor.layer5 = new_layer5
    return model


def adapt_eldernet(pretrained_model, new_window_size=WINDOW_SIZE):
    model = copy.deepcopy(pretrained_model)

    model = fix_circular_padding(model)
    model = remove_last_downsample(model)

    # After removing one Downsample, the feature map going into fc is now
    # (batch, 1024, 6) instead of (batch, 1024, 3) — we need to check
    # what the fc layer expects and whether global pooling handles this.
    # Pass a dummy input to find out.
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, new_window_size)
        try:
            out = model(dummy)
            print(f"Model accepts {new_window_size}-sample input. Output: {out.shape}")
        except Exception as e:
            print(f"Still failing: {e}")
            # If it still fails, print the feature extractor output shape to debug
            try:
                feat = model.feature_extractor(dummy)
                print(f"Feature extractor output shape: {feat.shape}")
            except Exception as e2:
                print(f"Feature extractor also failed: {e2}")
            raise

    return model

# ============================================================
# 5. TRAINING
# ============================================================

def train_model(model, train_loader, val_loader, epochs=40, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")
    model = model.to(device)

    # Differential learning rates: slow for pretrained backbone, faster for head
    backbone_params = [p for n, p in model.named_parameters()
                       if 'fc' not in n and 'classifier' not in n]
    head_params     = [p for n, p in model.named_parameters()
                       if 'fc' in n or 'classifier' in n]

    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': lr * 0.1},
        {'params': head_params,     'lr': lr}
    ])

    # Weighted loss for class imbalance
    all_labels = torch.cat([y for _, y in train_loader])
    n_pos = all_labels.sum().item()
    n_neg = len(all_labels) - n_pos
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    print(f"Class weights — non-gait: 1.00, gait: {weight[1]:.2f}")
    criterion = nn.CrossEntropyLoss(weight=weight)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5
    )

    best_val_f1 = 0.0

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validate ---
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for X, y in val_loader:
                preds = model(X.to(device)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(y.numpy())

        val_f1   = f1_score(all_true, all_preds, zero_division=0)
        val_prec, val_rec, _, _ = precision_recall_fscore_support(
            all_true, all_preds, labels=[1], average='binary', zero_division=0
        )
        val_acc  = accuracy_score(all_true, all_preds)
        avg_loss = train_loss / len(train_loader)

        print(f"Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | "
              f"val_F1={val_f1:.4f}  prec={val_prec:.4f}  rec={val_rec:.4f}  acc={val_acc:.4f}")

        scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  --> Saved (best val F1: {best_val_f1:.4f})")

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    print(f"Model saved to: {SAVE_PATH}")
    return model


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("Loading pretrained ElderNet...")
    pretrained = safe_hub_load('yonbrand/ElderNet', 'eldernet_ft', trust_repo=True)

    print("\nAdapting model for 100-sample (2s @ 50Hz) input...")
    model = adapt_eldernet(pretrained, new_window_size=WINDOW_SIZE)

    print("\nLoading QSense_mixed dataset...")
    qs_wins, qs_labs, qs_subs = load_qsense_dataset(QSENSE_MIXED_PATH)
    
    print("\nLoading QSense_edge dataset...")
    qe_wins, qe_labs, qe_subs = load_qsense_dataset(QSENSE_EDGE_PATH)

    print("\nLoading Free-living dataset...")
    fl_wins, fl_labs, fl_subs = load_freeliving_dataset(FREELIVING_PATH)

    all_wins = np.concatenate([qs_wins, qe_wins, fl_wins])
    all_labs = np.concatenate([qs_labs, qe_labs, fl_labs])
    all_subs = np.concatenate([qs_subs, qe_subs, fl_subs])

    print(f"\nTotal windows : {len(all_labs)}")
    print(f"Gait          : {all_labs.sum()}")
    print(f"Non-gait      : {(all_labs == 0).sum()}")
    print(f"Unique subjects: {np.unique(all_subs)}")

    # Subject-aware split — no subject appears in both train and val
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(all_wins, all_labs, groups=all_subs))

    print(f"\nTrain windows : {len(train_idx)}  |  Val windows: {len(val_idx)}")
    print(f"Train subjects: {np.unique(all_subs[train_idx])}")
    print(f"Val subjects  : {np.unique(all_subs[val_idx])}")

    train_ds = GaitDataset(all_wins[train_idx], all_labs[train_idx])
    val_ds   = GaitDataset(all_wins[val_idx],   all_labs[val_idx])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=0)

    train_model(model, train_loader, val_loader, epochs=40, lr=1e-4)


if __name__ == '__main__':
    main()
