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

from config.paths import QSENSE_MIXED, QSENSE_CLINIC

# EDIT THIS BEFORE RUNNING
# DATA_PATH = os.path.join(QSENSE_MIXED, 'Test2')
DATA_PATH = os.path.join(QSENSE_CLINIC, 'sub3')
#THRESHOLD = time(14, 57, 0)


LABELS = np.array([0, # still
                   1, # walk
                   0, # kitchen
                   1, # walk back 
                   0, # dominoes 
                   1, # walking around the building 
                   0, # magazine 
                   1, # walking to the execrise room
                   0, # exercise
                   1, # walk to the plants 
                   0, # reaching 
                   1, # keep walking
                   0, # watering 
                   1]) # walk from the plants 
TIMESTAMPS = [time(10, 29,  8), # walk to kitchen start   
              time(10, 29, 21), # start washing hands
              time(10, 29, 59), # walk back   
              time(10, 30, 12), # start dominoes 
              time(10, 32, 34), # start walking arounf the building 
              time(10, 34,  3), # start magazine 
              time(10, 36,  6), # start walking to the exercise room 
              time(10, 36, 44), # exercise 
              time(10, 37, 48), # walking to the plants 
              time(10, 38, 17), # reach for the jug  
              time(10, 38, 21), # keep walking 
              time(10, 38, 28), # start watering 
              time(10, 39,  9)] # start walking after the plants 
              
TIME_RANGE = (time(10, 28, 35), time(10, 39, 23))

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

    if os.path.exists(rw_path):
        new_file = os.path.join(data_path, 's1_1RW_ed.txt')
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
        new_file = os.path.join(data_path, 's2_2LW_ed.txt')
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
        new_file = os.path.join(data_path, 's3_3RL_ed.txt')
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