import os
import pandas as pd
import numpy as np
import csv
from datetime import time

# EDIT THIS BEFORE RUNNING
DATA_PATH = r"C:\Users\orlov\intern\gait_detection\QSense_data_mixed"
THRESHOLD = time(14, 25, 0)

def annotate(data_path, time_switch):
    for folder in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue

        rw_path = os.path.join(folder_path, 's1_1RW.txt')
        lw_path = os.path.join(folder_path, 's2_2LW.txt')

        if os.path.exists(rw_path):
            new_file = os.path.join(folder_path, 's1_1RW_ed.txt')
            with open(rw_path, newline='') as infile, open(new_file, 'w', newline='') as outfile:
                reader = csv.DictReader(infile, delimiter='\t')
                fieldnames = reader.fieldnames + ['Label']

                writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',')
                writer.writeheader()

                for row in reader:
                    t_str = row['HH:mm:ss.fff'].strip()
                    h, m, s_ms = t_str.split(':')
                    s, ms = s_ms.split('.')
                    row_time = time(int(h), int(m), int(s), int(ms) * 1000)  # microseconds

                    row['Label'] = 1 if row_time <= THRESHOLD else 0
                    writer.writerow(row)

        if os.path.exists(lw_path):
            new_file = os.path.join(folder_path, 's2_2LW_ed.txt')
            with open(lw_path, newline='') as infile, open(new_file, 'w', newline='') as outfile:
                reader = csv.DictReader(infile, delimiter='\t')
                fieldnames = reader.fieldnames + ['Label']

                writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter=',')
                writer.writeheader()

                for row in reader:
                    t_str = row['HH:mm:ss.fff'].strip()
                    h, m, s_ms = t_str.split(':')
                    s, ms = s_ms.split('.')
                    row_time = time(int(h), int(m), int(s), int(ms) * 1000)  # microseconds

                    row['Label'] = 1 if row_time <= THRESHOLD else 0
                    writer.writerow(row)


if __name__ == "__main__":
    annotate(DATA_PATH, THRESHOLD)