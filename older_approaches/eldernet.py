import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- CONFIGURATION ---
DATASET_PATH = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\HMP_Dataset'
REPO_NAME = 'yonbrand/ElderNet'
WINDOW_SIZE = 300 
GAIT_CLASSES = {'Walk', 'Climb_stairs', 'Descend_stairs'}

class HMPDatasetBalanced(Dataset):
    def __init__(self, root_dir):
        self.samples, self.labels, self.activity_names = [], [], []
        self.categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and "_MODEL" not in d])
        
        for cat in self.categories:
            is_gait = 1 if cat in GAIT_CLASSES else 0
            for f in glob.glob(os.path.join(root_dir, cat, '*.txt')):
                try:
                    raw = np.loadtxt(f)
                    if len(raw) < WINDOW_SIZE: continue
                    
                    # 1. Basic G Conversion (Keep the gravity offset!)
                    data = (raw.astype(float) - 32.0) * (1.5 / 32.0)
                    
                    # 2. Gentle Low-pass (10Hz instead of 3Hz) to keep the 'snap' of footsteps
                    nyq = 0.5 * 32.0
                    b, a = signal.butter(4, 10.0/nyq, btype='low')
                    data = signal.filtfilt(b, a, data, axis=0)
                    
                    # 3. Resample
                    new_len = int(len(data) * (30 / 32))
                    data = signal.resample(data, new_len)
                    
                    # 4. Windows (Non-overlapping for clarity first)
                    for i in range(0, len(data) - WINDOW_SIZE + 1, WINDOW_SIZE):
                        win = data[i : i + WINDOW_SIZE]
                        
                        # --- THE AXIS FIX ---
                        # Some models expect the 'Vertical' axis in a specific slot.
                        # HMP is [X, Y, Z]. If this fails, we will swap these.
                        self.samples.append(win.T) 
                        self.labels.append(is_gait)
                        self.activity_names.append(cat)
                except: continue

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return torch.FloatTensor(self.samples[i]), self.labels[i], self.activity_names[i]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HMPDatasetBalanced(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model = torch.hub.load(REPO_NAME, 'eldernet_ft').to(device)
    model.eval()
    
    results = {cat: {'total': 0, 'gait_pred': 0} for cat in dataset.categories}
    
    print("Running balanced inference...")
    with torch.no_grad():
        for x, y, names in loader:
            out = model(x.to(device))
            # Revert to standard 0.5 threshold
            preds = torch.argmax(out, dim=1).cpu().numpy()
            
            for i in range(len(preds)):
                results[names[i]]['total'] += 1
                if preds[i] == 1:
                    results[names[i]]['gait_pred'] += 1

    print(f"\n{'ACTIVITY':<20} | {'GAIT %':<10} | {'STATUS'}")
    print("-" * 45)
    for cat in dataset.categories:
        res = results[cat]
        if res['total'] == 0: continue
        pct = (res['gait_pred'] / res['total']) * 100
        status = "OK" if ((cat in GAIT_CLASSES) == (pct > 50)) else "FAIL"
        print(f"{cat:<20} | {pct:>8.1f}% | {status}")

if __name__ == "__main__":
    main()