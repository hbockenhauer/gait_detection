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
STEP_SIZE = 30         # NEW: Slide by 1 second (30 samples at 30Hz)
GAIT_CLASSES = {'Walk', 'Climb_stairs', 'Descend_stairs'}

def extract_subject_id_and_timestamp(filename):
    """Extract subject ID and timestamp from filename.
    Format: Accelerometer-YYYY-MM-DD-HH-MM-SS-activity-subjectid.txt
    e.g., Accelerometer-2011-03-24-09-51-07-walk-f1.txt
    """
    parts = filename.replace('.txt', '').split('-')
    if len(parts) < 9:
        return None, None
    subject_id = parts[-1]
    try:
        ts_str = f"{parts[1]}-{parts[2]}-{parts[3]} {parts[4]}:{parts[5]}:{parts[6]}"
        ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        return subject_id, ts
    except:
        return None, None

class HMPDatasetOverlapping(Dataset):
    """ADL Dataset with overlapping windows for better temporal resolution"""
    
    def __init__(self, root_dir, window_size=300, step_size=30):
        self.window_size = window_size
        self.step_size = step_size
        self.samples, self.labels, self.activity_names = [], [], []
        self.subject_ids = []
        self.window_timestamps = []  # Track window start time
        
        self.categories = sorted([d for d in os.listdir(root_dir) 
                                if os.path.isdir(os.path.join(root_dir, d)) 
                                and "_MODEL" not in d])
        
        # Group files by subject
        subject_files = {}
        for cat in self.categories:
            for f in glob.glob(os.path.join(root_dir, cat, '*.txt')):
                base = os.path.basename(f)
                subj_id, ts = extract_subject_id_and_timestamp(base)
                if subj_id and ts:
                    if subj_id not in subject_files:
                        subject_files[subj_id] = []
                    subject_files[subj_id].append((ts, cat, f))
        
        # Process each subject
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
            
            if len(concat_data) < self.window_size:
                continue
            
            # Create OVERLAPPING windows
            num_windows = 0
            for i in range(0, len(concat_data) - self.window_size + 1, self.step_size):
                win = concat_data[i : i + self.window_size]
                activity_window = all_activities[i : i + self.window_size]
                
                # Find majority activity
                unique, counts = np.unique(activity_window, return_counts=True)
                majority_activity = unique[np.argmax(counts)]
                is_gait = 1 if majority_activity in GAIT_CLASSES else 0
                
                self.samples.append(win.T)
                self.labels.append(is_gait)
                self.activity_names.append(majority_activity)
                self.subject_ids.append(subj_id)
                self.window_timestamps.append(i / 30.0)  # Time in seconds
                num_windows += 1
            
            print(f"  Subject {subj_id}: {num_windows} windows (step={self.step_size})")
        
        overlap_pct = (1 - self.step_size / self.window_size) * 100
        print(f"\nLoaded {len(self.samples)} windows from {len(set(self.subject_ids))} subjects")
        print(f"Window overlap: {overlap_pct:.1f}% (step size: {self.step_size} samples = {self.step_size/30:.1f}s)")

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, i): 
        return (torch.FloatTensor(self.samples[i]), 
                self.labels[i], 
                self.activity_names[i], 
                self.subject_ids[i],
                self.window_timestamps[i])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("ADL DATASET - ELDERNET WITH OVERLAPPING WINDOWS")
    print("="*60)
    print(f"Window size: {WINDOW_SIZE} samples ({WINDOW_SIZE/30:.1f}s)")
    print(f"Step size: {STEP_SIZE} samples ({STEP_SIZE/30:.1f}s)")
    print(f"Overlap: {(1 - STEP_SIZE/WINDOW_SIZE)*100:.1f}%")
    print("="*60 + "\n")
    
    dataset = HMPDatasetOverlapping(DATASET_PATH, 
                                    window_size=WINDOW_SIZE, 
                                    step_size=STEP_SIZE)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = torch.hub.load(REPO_NAME, 'eldernet_ft', trust_repo=True).to(device)
    model.eval()
    
    # Thresholds
    CONF_THRESH = 0.60  
    ENERGY_THRESH = 0.10 
    
    # Collectors
    all_y_true = []
    all_y_pred = []
    all_names = []
    all_subject_ids = []
    all_timestamps = []
    all_probs = []
    all_energies = []
    all_freqs = []

    def get_dominant_freq(win, fs=30):
        mag = np.sqrt(np.sum(win**2, axis=0))
        mag = mag - np.mean(mag)
        freqs = np.fft.rfftfreq(len(mag), d=1/fs)
        fft_vals = np.abs(np.fft.rfft(mag))
        return freqs[np.argmax(fft_vals)]

    print("Evaluating HMP Dataset with overlapping windows...")
    with torch.no_grad():
        for batch in loader:
            x, y, names, subject_ids, timestamps = batch
            x_dev = x.to(device)
            out = model(x_dev)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            
            # Energy Calculation
            mags = torch.sqrt(torch.sum(x**2, dim=1)).numpy()
            batch_energy = np.std(mags, axis=1) 
            
            # Frequency per window
            batch_freq = []
            for i in range(x.shape[0]):
                win = x[i].numpy()
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
            all_timestamps.extend(timestamps.numpy())
            all_probs.extend(probs.tolist())
            all_energies.extend(batch_energy.tolist())
            all_freqs.extend(batch_freq.tolist())

    # Convert to arrays
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_names = np.array(all_names)
    all_subject_ids = np.array(all_subject_ids)
    all_timestamps = np.array(all_timestamps)
    all_probs = np.array(all_probs)
    all_energies = np.array(all_energies)
    all_freqs = np.array(all_freqs)
    
    if len(all_y_true) == 0:
        print('ERROR: No data loaded')
        return

    # --- 1. GLOBAL METRICS ---
    p, r, f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='binary', zero_division=0)
    acc = accuracy_score(all_y_true, all_y_pred)

    print("\n" + "="*60)
    print("GLOBAL PERFORMANCE (WITH OVERLAPPING WINDOWS)")
    print("="*60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Total windows evaluated: {len(all_y_true)}")

    # --- 2. PER-ACTIVITY BREAKDOWN ---
    print("\n" + "="*60)
    print(f"{'ACTIVITY':<20} | {'Windows':>8} | {'PREC':>6} | {'REC':>6} | {'F1':>6}")
    print("-" * 60)
    
    unique_activities = sorted(list(set(all_names)))
    for act in unique_activities:
        idx = np.where(all_names == act)
        y_t_sub = all_y_true[idx]
        y_p_sub = all_y_pred[idx]
        n_windows = len(y_t_sub)
        
        p_a, r_a, f1_a, _ = precision_recall_fscore_support(
            y_t_sub, y_p_sub, labels=[1], average='binary', zero_division=0
        )
        
        marker = " ✓" if act in GAIT_CLASSES else ""
        print(f"{act:<20} | {n_windows:>8} | {p_a:>6.2f} | {r_a:>6.2f} | {f1_a:>6.2f}{marker}")

    # --- 3. PER-SUBJECT BREAKDOWN ---
    print("\n" + "="*60)
    print(f"{'SUBJECT':<10} | {'Windows':>8} | {'PREC':>6} | {'REC':>6} | {'F1':>6}")
    print("-" * 60)
    
    unique_subjects = sorted(list(set(all_subject_ids)))
    for subj in unique_subjects:
        idx = np.where(all_subject_ids == subj)
        y_t_sub = all_y_true[idx]
        y_p_sub = all_y_pred[idx]
        n_windows = len(y_t_sub)
        
        p_s, r_s, f1_s, _ = precision_recall_fscore_support(
            y_t_sub, y_p_sub, labels=[1], average='binary', zero_division=0
        )
        
        print(f"{subj:<10} | {n_windows:>8} | {p_s:>6.2f} | {r_s:>6.2f} | {f1_s:>6.2f}")

    # --- 4. PLOTTING ---
    plot_dir = 'ElderNet_ADL_Plots_Overlapping'
    os.makedirs(plot_dir, exist_ok=True)

    # Group by subject
    subject_groups = {}
    for subj_id, ts, prob, eng, frq, act, y_t, y_p in zip(
        all_subject_ids, all_timestamps, all_probs, all_energies, 
        all_freqs, all_names, all_y_true, all_y_pred
    ):
        if subj_id not in subject_groups:
            subject_groups[subj_id] = {
                'timestamps': [], 'ConfSeq': [], 'EnergiesSeq': [], 
                'FreqsSeq': [], 'ActivitySeq': [], 'y_true': [], 'y_pred': []
            }
        subject_groups[subj_id]['timestamps'].append(ts)
        subject_groups[subj_id]['ConfSeq'].append(prob)
        subject_groups[subj_id]['EnergiesSeq'].append(eng)
        subject_groups[subj_id]['FreqsSeq'].append(frq)
        subject_groups[subj_id]['ActivitySeq'].append(act)
        subject_groups[subj_id]['y_true'].append(y_t)
        subject_groups[subj_id]['y_pred'].append(y_p)

    subject_list = sorted(list(subject_groups.keys()))
    if len(subject_list) == 0:
        print('No subjects to plot')
        return

    print('\n' + '='*60)
    print('AVAILABLE SUBJECTS FOR PLOTTING:')
    for i, subj_id in enumerate(subject_list):
        n_windows = len(subject_groups[subj_id]['ConfSeq'])
        duration = subject_groups[subj_id]['timestamps'][-1]
        print(f"  {i}: {subj_id} ({n_windows} windows, {duration:.1f}s duration)")

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
        sel = input("Enter subject index to plot (e.g. '0' or '0,2-4' or 'quit'): ").strip()
        if sel.lower() == 'quit':
            return
        sel_idxs = parse_selection(sel, len(subject_list))
        if sel_idxs is None or len(sel_idxs) == 0:
            print('Invalid selection')
            continue
        break

    # Plot selected subjects
    for s in sel_idxs:
        subj_id = subject_list[s]
        seqs = subject_groups[subj_id]
        
        # Convert to arrays
        timestamps = np.array(seqs['timestamps'])
        conf_seq = np.array(seqs['ConfSeq'])
        energy_seq = np.array(seqs['EnergiesSeq'])
        freq_seq = np.array(seqs['FreqsSeq'])
        acts = seqs['ActivitySeq']
        y_pred = np.array(seqs['y_pred'])
        
        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        
        # Gait detection regions
        min_freq, max_freq = 0.5, 3.0
        gait_mask = (conf_seq > CONF_THRESH) & (energy_seq > ENERGY_THRESH)
        if not np.all(np.isnan(freq_seq)):
            gait_mask &= (freq_seq > min_freq) & (freq_seq < max_freq)
        
        # Find contiguous gait regions
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
        
        # Activity transitions
        activity_transitions = []
        last_act = None
        for xi, act in enumerate(acts):
            if act != last_act:
                activity_transitions.append((xi, act))
                last_act = act
        
        # Plot 1: Confidence
        ax = axes[0]
        for start, end in gait_regions:
            ax.axvspan(timestamps[start], timestamps[end], alpha=0.15, color='gold')
        ax.plot(timestamps, conf_seq, color='steelblue', linewidth=1.5)
        ax.axhline(CONF_THRESH, color='gold', linestyle='--', linewidth=1.5, 
                   alpha=0.8, label=f'Threshold={CONF_THRESH}')
        ax.set_ylabel('Confidence')
        ax.set_title(f'Subject {subj_id} - Gait Detection (Overlapping Windows)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Energy
        ax = axes[1]
        for start, end in gait_regions:
            ax.axvspan(timestamps[start], timestamps[end], alpha=0.15, color='gold')
        ax.plot(timestamps, energy_seq, color='steelblue', linewidth=1.5)
        ax.axhline(ENERGY_THRESH, color='gold', linestyle='--', linewidth=1.5,
                   alpha=0.8, label=f'Threshold={ENERGY_THRESH}')
        ax.set_ylabel('Energy (std)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Frequency
        ax = axes[2]
        for start, end in gait_regions:
            ax.axvspan(timestamps[start], timestamps[end], alpha=0.15, color='gold')
        ax.plot(timestamps, freq_seq, color='steelblue', linewidth=1.5)
        ax.axhline(min_freq, color='gold', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.axhline(max_freq, color='gold', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.set_ylabel('Dominant Freq (Hz)')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Activity Timeline
        ax = axes[3]
        # Color code by activity
        for xi, act in activity_transitions:
            color = 'green' if act in GAIT_CLASSES else 'red'
            if xi < len(timestamps):
                ax.axvline(timestamps[xi], color=color, linestyle='--', alpha=0.4)
                ax.text(timestamps[xi], 0.5, act, rotation=90, fontsize=8, 
                       color=color, va='bottom')
        
        # Show prediction as binary
        ax.fill_between(timestamps, 0, y_pred, alpha=0.3, color='blue', 
                        label='Gait Predicted')
        ax.set_ylabel('Prediction')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylim([-0.1, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        out = os.path.join(plot_dir, f'subject_{subj_id}_overlapping_gait.png')
        plt.savefig(out, dpi=150)
        print(f"Saved plot: {out}")
        plt.show()

if __name__ == "__main__":
    main()