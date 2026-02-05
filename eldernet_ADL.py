import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import matplotlib.pyplot as plt
import pickle
import random
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

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
DATASET_PATH = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\HMP_Dataset'
REPO_NAME = 'yonbrand/ElderNet'
WINDOW_SIZE = 300      
GAIT_CLASSES = {'Walk', 'Climb_stairs', 'Descend_stairs'}

def extract_subject_id_and_timestamp(filename):
    """Extract subject ID and timestamp from filename.
    Format: Accelerometer-YYYY-MM-DD-HH-MM-SS-activity-subjectid.txt
    e.g., Accelerometer-2011-03-24-09-51-07-walk-f1.txt
    """
    parts = filename.replace('.txt', '').split('-')
    # parts = ['Accelerometer', '2011', '03', '24', '09', '51', '07', 'activity', 'subjectid']
    # So we need at least 9 parts
    if len(parts) < 9:
        return None, None
    # Last part is subject id (f1, m11, etc.)
    subject_id = parts[-1]
    # Timestamp is parts[1:7]
    try:
        ts_str = f"{parts[1]}-{parts[2]}-{parts[3]} {parts[4]}:{parts[5]}:{parts[6]}"
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        return subject_id, ts
    except:
        return None, None

class HMPDatasetDebug(Dataset):
    def __init__(self, root_dir):
        self.samples, self.labels, self.activity_names = [], [], []
        self.subject_ids = []
        self.categories = sorted([d for d in os.listdir(root_dir) 
                                if os.path.isdir(os.path.join(root_dir, d)) 
                                and "_MODEL" not in d])
        
        # Group files by subject
        subject_files = {}  # subject_id -> [(timestamp, category, filepath), ...]
        for cat in self.categories:
            for f in glob.glob(os.path.join(root_dir, cat, '*.txt')):
                base = os.path.basename(f)
                subj_id, ts = extract_subject_id_and_timestamp(base)
                if subj_id and ts:
                    if subj_id not in subject_files:
                        subject_files[subj_id] = []
                    subject_files[subj_id].append((ts, cat, f))
        
        # Process each subject: concatenate files in chronological order
        for subj_id in sorted(subject_files.keys()):
            files_sorted = sorted(subject_files[subj_id], key=lambda x: x[0])
            all_data = []
            all_activities = []
            
            for ts, cat, f in files_sorted:
                try:
                    raw = np.loadtxt(f)
                    if len(raw) < 150:
                        continue
                    
                    # Unit Conversion & Filtering
                    data = (raw.astype(float) - 32.0) * (1.5 / 32.0)
                    nyq = 0.5 * 32.0
                    b, a = signal.butter(4, 10.0/nyq, btype='low')
                    data = signal.filtfilt(b, a, data, axis=0)
                    
                    # Resample 32Hz -> 30Hz
                    new_len = int(len(data) * (30 / 32))
                    data = signal.resample(data, new_len)
                    
                    all_data.append(data)
                    all_activities.extend([cat] * len(data))
                except:
                    continue
            
            if len(all_data) == 0:
                continue
            
            # Concatenate all data for this subject
            concat_data = np.vstack(all_data)
            
            # Skip if not enough data for even one window
            if len(concat_data) < WINDOW_SIZE:
                continue
            
            # Create windows
            for i in range(0, len(concat_data) - WINDOW_SIZE + 1, WINDOW_SIZE):
                win = concat_data[i : i + WINDOW_SIZE]
                activity_window = all_activities[i : i + WINDOW_SIZE]
                # Find majority activity
                unique, counts = np.unique(activity_window, return_counts=True)
                majority_activity = unique[np.argmax(counts)]
                is_gait = 1 if majority_activity in GAIT_CLASSES else 0
                
                self.samples.append(win.T)
                self.labels.append(is_gait)
                self.activity_names.append(majority_activity)
                self.subject_ids.append(subj_id)
        
        print(f"Loaded {len(self.samples)} windows from {len(set(self.subject_ids))} subjects")

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): 
        # return sample, label, activity_name, subject_id
        return torch.FloatTensor(self.samples[i]), self.labels[i], self.activity_names[i], self.subject_ids[i]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HMPDatasetDebug(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = torch.hub.load(REPO_NAME, 'eldernet_ft').to(device)
    model.eval()
    
    # Thresholds
    CONF_THRESH = 0.60  
    ENERGY_THRESH = 0.10 
    
    # Collectors for Metrics and per-window details
    all_y_true = []
    all_y_pred = []
    all_names = []
    all_subject_ids = []
    all_probs = []
    all_energies = []
    all_freqs = []

    # helper for dominant frequency
    def get_dominant_freq(win, fs=30):
        mag = np.sqrt(np.sum(win**2, axis=0))
        mag = mag - np.mean(mag)
        freqs = np.fft.rfftfreq(len(mag), d=1/fs)
        fft_vals = np.abs(np.fft.rfft(mag))
        return freqs[np.argmax(fft_vals)]

    print("Evaluating HMP Dataset...")
    with torch.no_grad():
        for batch in loader:
            # dataset __getitem__ now returns 4 items
            x, y, names, subject_ids = batch
            x_dev = x.to(device)
            out = model(x_dev)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            
            # Energy Calculation (per-window)
            mags = torch.sqrt(torch.sum(x**2, dim=1)).numpy()  # shape (batch, T)
            batch_energy = np.std(mags, axis=1) 
            
            # Frequency per window
            batch_freq = []
            for i in range(x.shape[0]):
                win = x[i].numpy()  # shape channels x T
                try:
                    fdom = get_dominant_freq(win, fs=30)
                except Exception:
                    fdom = np.nan
                batch_freq.append(fdom)
            batch_freq = np.array(batch_freq)

            # Decision Logic
            preds = ((probs > CONF_THRESH) & (batch_energy > ENERGY_THRESH)).astype(int)

            all_y_true.extend(y.numpy())
            all_y_pred.extend(preds)
            all_names.extend(names)
            all_subject_ids.extend(subject_ids)
            all_probs.extend(probs.tolist())
            all_energies.extend(batch_energy.tolist())
            all_freqs.extend(batch_freq.tolist())

    # Convert to arrays for sklearn
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_names = np.array(all_names)
    all_subject_ids = np.array(all_subject_ids)
    all_probs = np.array(all_probs)
    all_energies = np.array(all_energies)
    all_freqs = np.array(all_freqs)
    
    # Check if we have any data
    if len(all_y_true) == 0:
        print('ERROR: No data loaded from HMP dataset. Check file paths and data integrity.')
        return

    # --- 1. GLOBAL METRICS ---
    p, r, f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='binary', zero_division=0)
    acc = accuracy_score(all_y_true, all_y_pred)

    print("\n" + "="*40)
    print("GLOBAL HMP PERFORMANCE")
    print("="*40)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # --- 2. PER-ACTIVITY BREAKDOWN ---
    print("\n" + "="*40)
    print(f"{'ACTIVITY':<20} | {'PREC':<6} | {'REC':<6} | {'F1':<6}")
    print("-" * 45)
    
    unique_activities = sorted(list(set(all_names)))
    for act in unique_activities:
        idx = np.where(all_names == act)
        y_t_sub = all_y_true[idx]
        y_p_sub = all_y_pred[idx]
        
        # If the activity is a GAIT class, we measure how many we found.
        # If it's a NON-GAIT class, we measure False Positives.
        # Note: Precision/Recall on a single class can be tricky, so we use labels=[1]
        p_a, r_a, f1_a, _ = precision_recall_fscore_support(y_t_sub, y_p_sub, labels=[1], average='binary', zero_division=0)
        
        print(f"{act:<20} | {p_a:>6.2f} | {r_a:>6.2f} | {f1_a:>6.2f}")

    # --- 2b. PER-SUBJECT BREAKDOWN ---
    print("\n" + "="*40)
    print(f"{'SUBJECT':<10} | {'PREC':<6} | {'REC':<6} | {'F1':<6}")
    print("-" * 40)
    
    unique_subjects = sorted(list(set(all_subject_ids)))
    for subj in unique_subjects:
        idx = np.where(all_subject_ids == subj)
        y_t_sub = all_y_true[idx]
        y_p_sub = all_y_pred[idx]
        
        p_s, r_s, f1_s, _ = precision_recall_fscore_support(y_t_sub, y_p_sub, labels=[1], average='binary', zero_division=0)
        
        print(f"{subj:<10} | {p_s:>6.2f} | {r_s:>6.2f} | {f1_s:>6.2f}")

    # --- 3. Prepare per-subject sequences for plotting ---
    plot_dir = 'ElderNet_ADL_Plots'
    os.makedirs(plot_dir, exist_ok=True)

    subject_groups = {}
    for subj_id, prob, eng, frq, act in zip(all_subject_ids, all_probs, all_energies, all_freqs, all_names):
        if subj_id not in subject_groups:
            subject_groups[subj_id] = {'ConfSeq': [], 'EnergiesSeq': [], 'FreqsSeq': [], 'ActivitySeq': []}
        subject_groups[subj_id]['ConfSeq'].append(prob)
        subject_groups[subj_id]['EnergiesSeq'].append(eng)
        subject_groups[subj_id]['FreqsSeq'].append(frq)
        subject_groups[subj_id]['ActivitySeq'].append(act)

    # Present available subject ids
    subject_list = sorted(list(subject_groups.keys()))
    if len(subject_list) == 0:
        print('No per-subject sequences to plot.')
        return

    print('\n' + '='*40)
    print('AVAILABLE SUBJECTS FOR PLOTTING:')
    for i, subj_id in enumerate(subject_list):
        n_windows = len(subject_groups[subj_id]['ConfSeq'])
        print(f"  {i}: {subj_id} ({n_windows} windows)")

    def parse_selection(input_str, max_idx):
        parts = [p.strip() for p in input_str.split(',') if p.strip()]
        idxs = set()
        for part in parts:
            if '-' in part:
                try:
                    a, b = part.split('-')
                    a, b = int(a), int(b)
                    for j in range(a, b+1):
                        if 0 <= j < max_idx:
                            idxs.add(j)
                except Exception:
                    return None
            else:
                try:
                    v = int(part)
                    if 0 <= v < max_idx:
                        idxs.add(v)
                except Exception:
                    return None
        return sorted(list(idxs))

    while True:
        sel = input("Enter subject index/indices to plot (e.g. '0' or '0,2-4' or 'quit'): ").strip()
        if sel.lower() == 'quit':
            return
        sel_idxs = parse_selection(sel, len(subject_list))
        if sel_idxs is None or len(sel_idxs) == 0:
            print('Invalid selection')
            continue
        break

    for s in sel_idxs:
        subj_id = subject_list[s]
        seqs = subject_groups[subj_id]
        fig, axes = plt.subplots(3,1, figsize=(14,10))
        info = [('Confidence','ConfSeq','Confidence (probability)', axes[0]), ('Energy','EnergiesSeq','Energy (std)', axes[1]), ('Frequency','FreqsSeq','Mean Frequency (Hz)', axes[2])]
        acts = seqs['ActivitySeq']
        conf_seq = np.array(seqs['ConfSeq'])
        energy_seq = np.array(seqs['EnergiesSeq'])
        freq_seq = np.array(seqs['FreqsSeq'])
        # optional freq thresholds (useful if available)
        min_freq = 0.5
        max_freq = 3.0
        # Gait detection mask: confidence AND energy (and freq if valid)
        gait_mask = (conf_seq > CONF_THRESH) & (energy_seq > ENERGY_THRESH)
        if not np.all(np.isnan(freq_seq)):
            gait_mask &= (freq_seq > min_freq) & (freq_seq < max_freq)
        # identify contiguous gait regions for shading
        gait_regions = []
        in_gait = False
        start_idx = 0
        for i, val in enumerate(gait_mask):
            if val and not in_gait:
                start_idx = i
                in_gait = True
            elif not val and in_gait:
                gait_regions.append((start_idx, i-1))
                in_gait = False
        if in_gait:
            gait_regions.append((start_idx, len(gait_mask)-1))
        
        # Group consecutive activities
        activity_transitions = []  # list of (start_idx, activity_name)
        last_act = None
        for xi, act in enumerate(acts):
            if act != last_act:
                activity_transitions.append((xi, act))
                last_act = act
        
        for title, key, ylabel, ax in info:
            seq = seqs[key]
            x = np.arange(len(seq))
            # Shade gait detection regions with light gold
            for start, end in gait_regions:
                ax.axvspan(start - 0.5, end + 0.5, alpha=0.15, color='gold')
            ax.plot(x, seq, color='steelblue', linewidth=1.5)

            # Add threshold lines in gold
            if key == 'ConfSeq':
                ax.axhline(CONF_THRESH, color='gold', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Conf threshold={CONF_THRESH}')
                ax.legend(loc='upper right', fontsize=8)
            elif key == 'EnergiesSeq':
                ax.axhline(ENERGY_THRESH, color='gold', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Energy threshold={ENERGY_THRESH}')
                ax.legend(loc='upper right', fontsize=8)
            elif key == 'FreqsSeq':
                # draw optional freq bounds if desired
                ax.axhline(min_freq, color='gold', linestyle='--', linewidth=1.0, alpha=0.6, label=f'Min freq={min_freq}')
                ax.axhline(max_freq, color='gold', linestyle='--', linewidth=1.0, alpha=0.6, label=f'Max freq={max_freq}')
                ax.legend(loc='upper right', fontsize=8)

            # Mark activity transitions only (green for gait)
            for xi, act in activity_transitions:
                line_color = 'green' if act in GAIT_CLASSES else 'red'
                ax.axvline(xi, color=line_color, linestyle='--', alpha=0.4)
                yval = seq[xi] if xi < len(seq) else None
                if yval is not None:
                    ax.text(xi + 0.5, yval, act, rotation=90, fontsize=7, color=line_color, va='bottom')

            ax.set_ylabel(ylabel)
            ax.set_xlabel('Window index')
            ax.set_title(f"Subject {subj_id} - {title}")
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Subject {subj_id}: Gait Detection Metrics with Activity Annotations", fontsize=14, fontweight='bold')
        fig.tight_layout()
        out = os.path.join(plot_dir, f'subject_{subj_id}_gait_detection.png')
        plt.savefig(out, dpi=150)
        print(f"Saved plot: {out}")
        plt.show()

if __name__ == "__main__":
    main()