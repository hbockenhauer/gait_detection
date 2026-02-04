import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import signal

# --- CONFIGURATION ---
DATASET_PATH = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\HMP_Dataset'
REPO_NAME = 'yonbrand/ElderNet'
WINDOW_SIZE = 300  # 10 seconds

# Define which HMP folders count as "Gait"
GAIT_CLASSES = {'Walk', 'Climb_stairs', 'Descend_stairs'}

class HMPDatasetDetailed(Dataset):
    def __init__(self, root_dir, window_size=300):
        self.samples = []
        self.binary_labels = []   # 0 or 1
        self.activity_names = []  # "Brush_teeth", "Walk", etc.
        
        # Store unique activities to map them to integers for the DataLoader
        self.class_lookup = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_lookup)}

        print(f"Found {len(self.class_lookup)} activity categories.")
        
        for category in self.class_lookup:
            is_gait = 1 if category in GAIT_CLASSES else 0
            files = glob.glob(os.path.join(root_dir, category, '*.txt'))
            
            for f_path in files:
                try:
                    raw_data = np.loadtxt(f_path)
                except:
                    continue 

                # Conversion: 0..63 -> -1.5g..+1.5g
                data_g = (raw_data - 32) * (1.5 / 32)
                
                # Resample 32Hz -> 30Hz
                new_len = int(len(data_g) * (30 / 32))
                data_resampled = signal.resample(data_g, new_len)
                
                # Create Windows
                if len(data_resampled) < window_size:
                    # Pad short files
                    pad_amt = window_size - len(data_resampled)
                    padded = np.pad(data_resampled, ((0, pad_amt), (0,0)), mode='edge')
                    self.samples.append(padded.T) 
                    self.binary_labels.append(is_gait)
                    self.activity_names.append(category)
                else:
                    # Slide over file
                    for i in range(0, len(data_resampled) - window_size + 1, window_size):
                        window = data_resampled[i : i + window_size]
                        self.samples.append(window.T)
                        self.binary_labels.append(is_gait)
                        self.activity_names.append(category)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.samples[idx])
        y_binary = torch.tensor(self.binary_labels[idx], dtype=torch.long)
        
        # We also return the index of the specific activity name
        act_name = self.activity_names[idx]
        act_idx = self.class_to_idx[act_name]
        
        return x, y_binary, act_idx

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = HMPDatasetDetailed(DATASET_PATH, WINDOW_SIZE)
    if len(dataset) == 0: return

    # No shuffle so we can easily group output, though shuffle=True is fine too
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    print("Loading ElderNet...")
    model = torch.hub.load(REPO_NAME, 'eldernet_ft').to(device)
    model.eval()
    
    # Statistics containers
    # Keys: Activity Name, Values: [Correct_Predictions, Total_Samples, Predicted_As_Gait_Count]
    stats = {name: {'total': 0, 'pred_gait': 0} for name in dataset.class_lookup}
    
    print("Running inference...")
    
    with torch.no_grad():
        for inputs, binary_labels, act_indices in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Prediction: 1 = Gait, 0 = Non-Gait
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Update stats
            current_act_indices = act_indices.numpy()
            
            for i in range(len(predictions)):
                act_name = dataset.class_lookup[current_act_indices[i]]
                pred = predictions[i]
                
                stats[act_name]['total'] += 1
                if pred == 1:
                    stats[act_name]['pred_gait'] += 1

    # --- PRINT REPORT ---
    print(f"\n{'ACTIVITY':<20} | {'TYPE':<10} | {'SAMPLES':<8} | {'PREDICTED GAIT %':<18} | {'STATUS'}")
    print("-" * 85)
    
    overall_correct = 0
    overall_total = 0

    for name in sorted(dataset.class_lookup):
        data = stats[name]
        if data['total'] == 0: continue
        
        is_gait_class = name in GAIT_CLASSES
        type_str = "GAIT" if is_gait_class else "Non-Gait"
        
        # Calculate Percentage classified as Gait
        gait_pct = (data['pred_gait'] / data['total']) * 100
        
        # Determine if this result is "Good" or "Bad"
        # Good: Gait class > 50% OR Non-Gait class < 50%
        if is_gait_class:
            success = gait_pct > 50
            overall_correct += data['pred_gait']
        else:
            success = gait_pct <= 50
            overall_correct += (data['total'] - data['pred_gait'])
            
        overall_total += data['total']
        status = "OK" if success else "FAIL"
        
        print(f"{name:<20} | {type_str:<10} | {data['total']:<8} | {gait_pct:>17.1f}% | {status}")

    print("-" * 85)
    print(f"Overall Accuracy: {100 * overall_correct / overall_total:.2f}%")

if __name__ == "__main__":
    main()