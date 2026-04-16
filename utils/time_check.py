"""
Produces the plots of timestamp vs row index to detect any faulty data. 
"""


import csv
import os
import matplotlib.pyplot as plt
from datetime import time

from config.paths import (
    QSENSE_CLINIC, 
    QSENSE_DATA, 
    QSENSE_EDGE, 
    QSENSE_MIXED, 
    QSENSE_TEST,
    PLOTS_DIR
)

###############################################
DATA_ROOT = QSENSE_TEST
save_folder = "\faulty_data_plots\tests"

SAVE_PATH = os.path.join(PLOTS_DIR, save_folder)
################################################

FILES = [
    "s0_Hub.txt",
    "s1_1RW.txt",
    "s2_2LW.txt",
    "s3_3RL.txt",
]

def parse_time(t_str):
    h, m, s_ms = t_str.strip().split(':')
    s, ms = s_ms.split('.')
    return time(int(h), int(m), int(s), int(ms) * 1000)


def read_timestamps(file_path):
    timestamps = []
    with open(file_path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            t = parse_time(row['HH:mm:ss.fff'])
            timestamps.append(t)
    return timestamps

def plot_folder(folder_path, txt_files, out_folder):
    n = len(txt_files)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=False)

    if n == 1:
        axes = [axes]

    for ax, file_name in zip(axes, txt_files):
        file_path = os.path.join(folder_path, file_name)
        try:
            timestamps = read_timestamps(file_path)
        except Exception as e:
            ax.set_title(f"{file_name} — ERROR: {e}")
            continue

        ts_seconds = [
            t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
            for t in timestamps
        ]
        ax.plot(range(len(ts_seconds)), ts_seconds, linewidth=0.8)
        ax.set_title(file_name)
        ax.set_ylabel('Seconds since midnight')

    folder_name = os.path.basename(folder_path)
    plt.suptitle(folder_name, fontsize=13)
    plt.tight_layout()

    out_path = os.path.join(out_folder, f"{folder_name}_timecheck.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == '__main__':
    subfolders = sorted([
        entry.path
        for entry in os.scandir(DATA_ROOT)
        if entry.is_dir()
    ])
    print(f"Found {len(subfolders)} folder(s) in {DATA_ROOT}\n")

    for folder_path in subfolders:
        # Only keep FILES that actually exist in this folder
        present_files = [f for f in FILES if os.path.exists(os.path.join(folder_path, f))]

        if not present_files:
            print(f"Skipping {os.path.basename(folder_path)} — none of the FILES found")
            continue

        print(f"Processing: {os.path.basename(folder_path)} ({len(present_files)} file(s))")
        plot_folder(folder_path, present_files, out_folder=SAVE_PATH)

    print("\nDone.")