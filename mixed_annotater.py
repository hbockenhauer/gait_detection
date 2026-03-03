import os
import pandas as pd
import numpy as np
import csv
from datetime import time

# EDIT THIS BEFORE RUNNING
#DATA_PATH = r"C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data_mixed\Test2"
DATA_PATH = r"C:\Users\orlov\intern\gait_detection\QSense_data_mixed\test9_Hendrik"
#THRESHOLD = time(14, 57, 0)
## test 3
# TIMESTAMPS = [time(10, 36, 26), 
#               time(10, 39, 18), 
#               time(10, 40,  5),
#               time(10, 44, 20), 
#               time(10, 45, 54),
#               time(10, 48, 30)]
# LABELS = np.array([1, 0, 1, 0, 1, 0, 1])

## test4 
# TIMESTAMPS = [time(11,  4, 10), 
#               time(11,  4, 52), 
#               time(11,  5, 53), 
#               time(11,  6, 47), 
#               time(11,  8, 15), 
#               time(11,  9, 00),
#               time(11, 10, 00), 
#               time(11, 10, 44), 
#               time(11, 11, 37), 
#               time(11, 12, 48)]
LABELS = np.array([0, # stand still
                   1, # walk natural
                   0, # kitchen
                   1, # walk, slow
                   0, # sit and eat smth
                   1, # walk slow, stroke 
                   0, # pretend to vacuum - right hand 
                   1, # walk with various speeds 
                   0, # stand with conve
                   1, # walk up and down
                   0, 
                   1]) 
TIMESTAMPS = [time(13, 42,  5), 
              time(13, 42,  8), 
              time(13, 42, 35), 
              time(13, 42, 40), 
              time(13, 43,  0), 
              time(13, 43, 10),
              time(13, 43, 35), 
              time(13, 43, 50), 
              time(13, 44, 20),
              time(13, 44, 40), 
              time(13, 45,  0)]
              
TIME_RANGE = (time(13, 41, 50), time(13, 45, 30))

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