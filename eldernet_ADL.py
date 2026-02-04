import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import random
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

class HMPDatasetDebug(Dataset):
    def __init__(self, root_dir):
        self.samples, self.labels, self.activity_names = [], [], []
        self.categories = sorted([d for d in os.listdir(root_dir) 
                                if os.path.isdir(os.path.join(root_dir, d)) 
                                and "_MODEL" not in d])
        
        for cat in self.categories:
            is_gait = 1 if cat in GAIT_CLASSES else 0
            for f in glob.glob(os.path.join(root_dir, cat, '*.txt')):
                try:
                    raw = np.loadtxt(f)
                    if len(raw) < 150: continue 
                    
                    # Unit Conversion & Filtering
                    data = (raw.astype(float) - 32.0) * (1.5 / 32.0)
                    nyq = 0.5 * 32.0
                    b, a = signal.butter(4, 10.0/nyq, btype='low')
                    data = signal.filtfilt(b, a, data, axis=0)
                    
                    # Resample 32Hz -> 30Hz
                    new_len = int(len(data) * (30 / 32))
                    data = signal.resample(data, new_len)
                    
                    for i in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_SIZE):
                        win = data[i : i + WINDOW_SIZE]
                        self.samples.append(win.T)
                        self.labels.append(is_gait)
                        self.activity_names.append(cat)
                except: continue

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): 
        return torch.FloatTensor(self.samples[i]), self.labels[i], self.activity_names[i]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HMPDatasetDebug(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = torch.hub.load(REPO_NAME, 'eldernet_ft').to(device)
    model.eval()
    
    # Thresholds
    CONF_THRESH = 0.60  
    ENERGY_THRESH = 0.10 
    
    # Collectors for Metrics
    all_y_true = []
    all_y_pred = []
    all_names = []

    print("Evaluating HMP Dataset...")
    with torch.no_grad():
        for x, y, names in loader:
            x_dev = x.to(device)
            out = model(x_dev)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            
            # Energy Calculation
            mags = torch.sqrt(torch.sum(x**2, dim=1)).numpy() 
            batch_energy = np.std(mags, axis=1) 
            
            # Decision Logic
            preds = ((probs > CONF_THRESH) & (batch_energy > ENERGY_THRESH)).astype(int)
            
            all_y_true.extend(y.numpy())
            all_y_pred.extend(preds)
            all_names.extend(names)

    # Convert to arrays for sklearn
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_names = np.array(all_names)

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

if __name__ == "__main__":
    main()