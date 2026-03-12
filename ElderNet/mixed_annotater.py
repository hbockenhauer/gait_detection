import os
import pandas as pd
import numpy as np
import csv
from datetime import time

# EDIT THIS BEFORE RUNNING
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = (os.path.dirname(SCRIPT_DIR)
                if os.path.basename(SCRIPT_DIR) in {'ElderNet', 'StrokeNet'}
                else SCRIPT_DIR)
# DATA_PATH = os.path.join(PROJECT_ROOT, 'Datasets', 'QSense_data_mixed', 'Test2')
DATA_PATH = os.path.join(PROJECT_ROOT, 'Datasets', 'QSense_data_mixed', 'test11_Hendrik')
#THRESHOLD = time(14, 57, 0)


LABELS = np.array([1, 0])
# np.array([0, # stand still
#                    1, # walk natural
#                    0, # kitchen
#                    1, # walk, slow
#                    0, # sit and eat smth
#                    1, # walk slow, stroke 
#                    0, # pretend to vacuum - right hand 
#                    1, # walk with various speeds 
#                    0, # stand with conve
#                    1, # walk up and down
#                    0]) 
TIMESTAMPS = [time(15, 27, 56)]
# [time(14, 54, 59), 
#               time(14, 55, 39), 
#               time(14, 56, 20), 
#               time(14, 57,  6), 
#               time(14, 57, 52), 
#               time(14, 58, 35),
#               time(14, 59, 18), 
#               time(15, 00, 10), 
#               time(15, 00, 54),
#               time(15,  1, 57)]
              
TIME_RANGE = (time(15, 14, 25), time(15, 39, 00))

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

    for file in os.listdir(data_path):

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