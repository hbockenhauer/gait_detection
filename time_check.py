import csv
import os
import matplotlib.pyplot as plt
from datetime import time


DATA_PATH = r"C:\Users\orlov\intern\gait_detection\QSense_data_mixed\test3_Tanya"

FILES = [
    "s1_1RW.txt",
    "s2_2LW.txt",
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


if __name__ == '__main__':
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    for ax, file_name in zip(axes, FILES):
        file_path = os.path.join(DATA_PATH, file_name)
        timestamps = read_timestamps(file_path)
        ts_seconds = [t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
                      for t in timestamps]
        ax.plot(range(len(ts_seconds)), ts_seconds, linewidth=0.8)
        ax.set_title(file_name)
        #ax.set_xlabel('Row index')
        ax.set_ylabel('Seconds since midnight')

    plt.suptitle(f"{os.path.basename(DATA_PATH)}", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(DATA_PATH, f"{os.path.basename(DATA_PATH)}_timecheck.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")
    plt.show()