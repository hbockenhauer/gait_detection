import torch
from ptflops import get_model_complexity_info
import os
import sys
import time
import psutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.hub_utils import safe_hub_load

def print_memory():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"Current System RAM Usage: {mem_mb:.2f} MB")

# Run this before and after model load
print("Before model load:")
print_memory()

# --- Load ElderNet model ---
REPO_NAME = 'yonbrand/ElderNet'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = safe_hub_load(REPO_NAME, 'eldernet_ft', trust_repo=True).to(device)
model.eval()

print("After model load:")
print_memory()

# --- 1. Number of parameters ---
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# --- 2. Model size on disk ---
torch.save(model.state_dict(), "temp_model.pth")
model_size = os.path.getsize("temp_model.pth") / 1e6
os.remove("temp_model.pth")
print(f"Model size on disk: {model_size:.2f} MB")

# --- 3. FLOPs / Multiply-Adds ---
# Input shape: 3 channels (accX, accY, accZ) x 300 timesteps per window
macs, params = get_model_complexity_info(model, (3, 300), as_strings=True, print_per_layer_stat=False)
print(f"MACs (Multiply-Adds) per window: {macs}")
print(f"Parameters (from ptflops): {params}")

# --- 4. Inference time per window ---
x = torch.randn(1, 3, 300).to(device)  # batch of 1 window
# Warm-up (important for GPU)
for _ in range(10):
    _ = model(x)

# Timing
iterations = 100
start = time.time()
for _ in range(iterations):
    _ = model(x)
end = time.time()
avg_time = (end - start) / iterations
print(f"Average inference time per window: {avg_time:.6f} s on {device}")

