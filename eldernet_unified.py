import torch
import numpy as np
import pandas as pd
import os
import glob
import random
from scipy import signal
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# --- 0. REPRODUCIBILITY ---
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# --- 1. DATASET LOADERS ---

class GaitDataset(Dataset):
    def __init__(self):
        self.samples, self.labels, self.activity_names = [], [], []
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): 
        return torch.FloatTensor(self.samples[i]), self.labels[i], self.activity_names[i]

class WISDMLoader(GaitDataset):
    # WISDM is high-res; can handle strict gates
    CONF_THRESH = 0.45 
    ENERGY_THRESH = 0.07
    def __init__(self, folder_path, window_size=300):
        super().__init__()
        GAIT_CODES = {'A', 'C'} # Walk, Stairs (Jogging excluded)
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
        for filename in files:
            try:
                df = pd.read_csv(os.path.join(folder_path, filename), header=None, 
                                 names=['user', 'activity', 'ts', 'x', 'y', 'z'])
                df['z'] = df['z'].astype(str).str.replace(';', '').astype(float)
                data = df[['x', 'y', 'z']].values / 9.81
                new_len = int(len(data) * (30 / 20))
                data = signal.resample(data, new_len)
                labels = df['activity'].values
                indices = np.linspace(0, len(labels)-1, new_len).astype(int)
                resampled_labels = labels[indices]
                step = window_size // 2
                for i in range(0, len(data) - window_size + 1, step):
                    self.samples.append(data[i : i + window_size].T)
                    maj_label = pd.Series(resampled_labels[i : i + window_size]).mode()[0]
                    self.labels.append(1 if maj_label in GAIT_CODES else 0)
                    self.activity_names.append(maj_label)
            except: continue

class HMPLoader(GaitDataset):
    # HMP is low-res; needs sensitive gates
    CONF_THRESH = 0.10   # Lowered to capture muffled steps
    ENERGY_THRESH = 0.005 # Widened to detect low-amplitude movement
    def __init__(self, folder_path, window_size=300):
        super().__init__()
        GAIT_CLASSES = {'Walk', 'Climb_stairs', 'Descend_stairs'}
        categories = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d)) and "_MODEL" not in d])
        for cat in categories:
            is_gait = 1 if cat in GAIT_CLASSES else 0
            for f in glob.glob(os.path.join(folder_path, cat, '*.txt')):
                try:
                    raw = np.loadtxt(f)
                    data = (raw.astype(float) - 32.0) * (1.5 / 32.0)
                    # Tight 5Hz filter to smooth 6-bit jaggedness
                    nyq = 0.5 * 32.0
                    b, a = signal.butter(4, 5.0/nyq, btype='low') 
                    data = signal.filtfilt(b, a, data, axis=0)
                    new_len = int(len(data) * (30 / 32))
                    data = signal.resample(data, new_len)
                    if len(data) < window_size:
                        data = np.pad(data, ((0, window_size - len(data)), (0, 0)), mode='edge')
                    step = window_size // 6 # High overlap helps recall
                    for i in range(0, len(data) - window_size + 1, step):
                        # Multiply by 1.3 to compensate for the +/- 1.5G cap
                        self.samples.append((data[i : i + window_size] * 1.3).T)
                        self.labels.append(is_gait)
                        self.activity_names.append(cat)
                except: continue

# --- 2. EVALUATION ---

def run_evaluation(dataset_type="WISDM"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if dataset_type == "WISDM":
        path = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\wisdm-dataset\raw\watch\accel'
        dataset = WISDMLoader(path)
        activity_map = {'A':'Walk', 'B':'Jog', 'C':'Stairs', 'D':'Sit', 'E':'Stand', 
        'F':'Type', 'G':'Teeth', 'H':'Soup', 'I':'Chair', 'J':'Pasta', 
        'K':'Drink', 'L':'Sandwich', 'M':'Kick', 'O':'Catch', 
        'P':'Dribble', 'Q':'Write', 'R':'Clap', 'S':'Fold'}
    else:
        path = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\HMP_Dataset'
        dataset = HMPLoader(path)
        activity_map = {}

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = torch.hub.load('yonbrand/ElderNet', 'eldernet_ft').to(device)
    model.eval()

    all_y_true, all_y_pred, all_names = [], [], []

    print(f"\n--- Running Evaluation: {dataset_type} ---")
    with torch.no_grad():
        for x, y, names in loader:
            probs = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu().numpy()
            mags = torch.sqrt(torch.sum(x**2, dim=1)).numpy()
            energies = np.std(mags, axis=1)
            
            # Use the dataset-specific thresholds defined in the class
            preds = ((probs > dataset.CONF_THRESH) & (energies > dataset.ENERGY_THRESH)).astype(int)
            
            all_y_true.extend(y.numpy())
            all_y_pred.extend(preds)
            all_names.extend(names)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_names = np.array(all_names, dtype=object)
    
    p, r, f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='binary', zero_division=0)
    print(f"Final Global Metrics | F1: {f1:.4f} | Prec: {p:.4f} | Rec: {r:.4f}")
    
    report = []
    unique_activities = np.unique(all_names)
    for act in unique_activities:
        idx = (all_names == act)
        if np.any(idx):
            pa, ra, f1a, _ = precision_recall_fscore_support(all_y_true[idx], all_y_pred[idx], labels=[1], average='binary', zero_division=0)
            report.append({'Activity': activity_map.get(act, act), 'Precision': pa, 'Recall': ra, 'F1': f1a})
    
    print("\nDetailed Breakdown:")
    print(pd.DataFrame(report).sort_values(by='F1', ascending=False).to_string(index=False))

if __name__ == "__main__":
    run_evaluation("WISDM")
    run_evaluation("HMP")