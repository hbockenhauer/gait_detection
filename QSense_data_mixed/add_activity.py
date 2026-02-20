## Add activity labels to mixed data files by appending extra column to the end of the dataframe and saving as new CSV files.
## User inputs a timestamp and activity label, then appends this to each file in the mixed data directory. Every file before timestamp
## gets the activity label, every file after timestamp gets "Free" label. Each folder within the mixed data directory is processed separately, so the same timestamp can be used for multiple folders.

import os
import pandas as pd
import numpy as np

MIXED_DATA_DIR = r'C:\Users\hendr\OneDrive\Documents\TU Delft\MSc Robotics\Internship at Erasmus MC\gait_detection\QSense_data_mixed'

def add_activity_labels(mixed_data_dir, folder, timestamp):

    folder_path = os.path.join(mixed_data_dir, folder)

    if not os.path.isdir(folder_path):
        print(f"Warning: Folder {folder_path} does not exist.")
        return

    for file in os.listdir(folder_path):

        # Only process sensor files
        if not (file.startswith("s1") or file.startswith("s2")):
            continue

        file_path = os.path.join(folder_path, file)

        df = pd.read_csv(file_path, sep="\t", engine="c")

        if df.empty:
            print(f"Skipping empty file: {file_path}")
            continue

        if 'HH:mm:ss.fff' not in df.columns:
            print(f"Missing timestamp column in {file_path}")
            continue

        df['label'] = 0

        first_timestamp = pd.to_datetime(
            df['yyyy-MM-dd'].iloc[0] + ' ' + df['HH:mm:ss.fff'].iloc[0],
            format='%Y-%m-%d %H:%M:%S.%f')

        if first_timestamp <= timestamp:
            df['label'] = 1

        df.to_csv(file_path, sep="\t", index=False)


if __name__ == "__main__":
    # Example usage
    folder = "test2"  # Specify the folder within the mixed data directory to process
    input_timestamp = pd.to_datetime("2026-02-19 14:57:00.000")  # User input timestamp
    add_activity_labels(MIXED_DATA_DIR, folder, input_timestamp)
    print(f"Activity labels added to files in {folder} based on timestamp {input_timestamp}.")