import os
import sys
import pandas as pd
import numpy as np
import csv
from datetime import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.paths import QSENSE_MIXED, QSENSE_CLINIC, QSENSE_TEST

# EDIT THIS BEFORE RUNNING
# DATA_PATH = os.path.join(QSENSE_MIXED, 'Test2')
DATA_PATH = os.path.join(QSENSE_TEST, 'Arm_use')
#THRESHOLD = time(14, 57, 0)

# The entire file for orientation is walking so label every timestamp as 1 (walk)
LABELS = np.array([0, 
                   1, 
                   0, 
                   2, 
                   0, 
                   1, 
                   0, 
                   2, 
                   0, 
                   1]) # walk
TIMESTAMPS = [time(10, 38,  0),
              time(10, 44,  0), 
              time(11,  2,  0), 
              time(11,  4,  0), 
              time(11, 27, 50),
              time(11, 35,  5), 
              time(12,  3,  0), 
              time(12,  5,  0),
              time(12, 28, 35)] 
TIME_RANGE = (time(10, 20, 50), time(12, 33, 35))

def get_label(row_time, timestamps, labels):
    """
    Segments:   [start, ts[0]), [ts[0], ts[1]), ..., [ts[-1], end]
    Labels:      labels[0]       labels[1]              labels[-1]
    """
    for i, ts in enumerate(timestamps):
        if row_time < ts:
            return labels[i]
    return labels[-1]

def annotate(data_path, timestamps, labels, time_range=None):
    if len(labels) != len(timestamps) + 1:
        raise ValueError(
            f"LABELS must have exactly len(TIMESTAMPS) + 1 entries, "
            f"got {len(labels)} labels and {len(timestamps)} timestamps."
        )

    def parse_time(t_str):
        h, m, s_ms = t_str.strip().split(':')
        s, ms = s_ms.split('.')
        return time(int(h), int(m), int(s), int(ms) * 1000)

    def in_range(row_time):
        if time_range is None:
            return True
        return time_range[0] <= row_time <= time_range[1]

    # for file in os.listdir(data_path):

    rw_path = os.path.join(data_path, 's1_1RW.txt')
    lw_path = os.path.join(data_path, 's2_2LW.txt')
    rl_path = os.path.join(data_path, 's3_3RL.txt')

    print(f"Looking in: {data_path}")
    print(f"RW exists: {os.path.exists(rw_path)} -> {rw_path}")
    print(f"LW exists: {os.path.exists(lw_path)} -> {lw_path}")
    print(f"RL exists: {os.path.exists(rl_path)} -> {rl_path}")

    if os.path.exists(rw_path):
        new_file = os.path.join(data_path, 's1_1RW_tmp.txt')
        with open(rw_path, newline='') as infile, open(new_file, 'w', newline='') as outfile:
            reader = csv.DictReader(infile, delimiter='\t')
            fieldnames = reader.fieldnames + ['Label']

            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()

            for row in reader:
                row_time = parse_time(row['HH:mm:ss.fff'])
                if not in_range(row_time):
                    continue
                row['Label'] = get_label(row_time, timestamps, labels)
                writer.writerow(row)

        original_backup_rw = rw_path.replace('.txt', '_old.txt')
        os.rename(rw_path, original_backup_rw)
        os.rename(new_file, rw_path)

    if os.path.exists(lw_path):
        new_file = os.path.join(data_path, 's2_2LW_tmp.txt')
        with open(lw_path, newline='') as infile, open(new_file, 'w', newline='') as outfile:
            reader = csv.DictReader(infile, delimiter='\t')
            fieldnames = reader.fieldnames + ['Label']

            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()

            for row in reader:
                row_time = parse_time(row['HH:mm:ss.fff'])
                if not in_range(row_time):
                    continue
                row['Label'] = get_label(row_time, timestamps, labels)
                writer.writerow(row)

        original_backup_lw = lw_path.replace('.txt', '_old.txt')
        os.rename(lw_path, original_backup_lw)
        os.rename(new_file, lw_path)

    if os.path.exists(rl_path):
        new_file = os.path.join(data_path, 's3_3RL_tmp.txt')
        with open(rl_path, newline='') as infile, open(new_file, 'w', newline='') as outfile:
            reader = csv.DictReader(infile, delimiter='\t')
            fieldnames = reader.fieldnames + ['Label']

            writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()

            for row in reader:
                row_time = parse_time(row['HH:mm:ss.fff'])
                if not in_range(row_time):
                    continue
                row['Label'] = get_label(row_time, timestamps, labels)
                writer.writerow(row)

        original_backup_rl = rl_path.replace('.txt', '_old.txt')
        os.rename(rl_path, original_backup_rl)
        os.rename(new_file, rl_path)


if __name__ == "__main__":

    annotate(DATA_PATH, TIMESTAMPS, LABELS, time_range=TIME_RANGE)