import os
import sys
import glob
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.paths import QSENSE_MIXED

DATASET_PATH = QSENSE_MIXED

def compare_leg_class(filepath):
    df = pd.read_csv(filepath, sep=None, engine="python")
    df = df.reset_index(drop=True)

    parent_folder = os.path.basename(os.path.dirname(filepath))

    x1_0 = 0
    x2_0 = 0
    x3_0 = 0
    x1_1 = 0
    x2_1 = 0
    x3_1 = 0

    if 'label' in df.columns or 'Label' in df.columns:
        label_col = 'label' if 'label' in df.columns else 'Label'

    if 'classification' in df.columns or 'Classification' in df.columns:
        class_col = 'classification' if 'classification' in df.columns else 'Classification'


    conf = pd.crosstab(df[class_col], df[label_col])

    # Raw counts
    x3_1 = conf.loc[3, 1] if (3 in conf.index and 1 in conf.columns) else 0
    x3_0 = conf.loc[3, 0] if (3 in conf.index and 0 in conf.columns) else 0

    total_walking = conf[1].sum() if 1 in conf.columns else 0
    total_predicted_3 = conf.loc[3].sum() if 3 in conf.index else 0
    total_samples = conf.values.sum()

    # Percentages
    pct_correct_walking_as_3 = ((x3_1 / total_walking * 100) if total_walking > 0 else 0).round(2)
    pct_predicted_3_correct = ((x3_1 / total_predicted_3 * 100) if total_predicted_3 > 0 else 0).round(2)
    pct_dataset_is_x3_1 = ((x3_1 / total_samples * 100) if total_samples > 0 else 0).round(2)

    total_nonwalking = conf[0].sum() if 0 in conf.columns else 0

    results = {}

    for cls in [1, 2, 3]:
        TP = conf.loc[cls, 1] if (cls in conf.index and 1 in conf.columns) else 0
        FP = conf.loc[cls, 0] if (cls in conf.index and 0 in conf.columns) else 0
        FN = total_walking - TP
        TN = total_nonwalking - FP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / total_walking if total_walking > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[cls] = {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TN": TN,
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100,
            "percent_of_dataset": (TP + FP) / total_samples * 100 if total_samples > 0 else 0
        }

    return conf, pct_correct_walking_as_3, pct_predicted_3_correct, pct_dataset_is_x3_1, results

if __name__ == "__main__":
    for folder in os.listdir(DATASET_PATH):
        if not os.path.isdir(os.path.join(DATASET_PATH, folder)):
            continue

        # Extract subject name and activity type from folder name

        files = [
            os.path.join(DATASET_PATH, folder, 's3_3RL.txt'),  # Right leg
        ]

        for file in files:
            if not os.path.exists(file):
                continue
            comparison = compare_leg_class(file)
            print(f"Comparison for {file}: {comparison[0]}")
            print(f"Percentage of walking samples correctly classified as class 3: {comparison[1]}%")
            print(f"Percentage of samples predicted as class 3 that are actually walking: {comparison[2]}%")
            print(f"Percentage of the dataset that is class 3 and walking: {comparison[3]}%")
            #print(f"Detailed results: {comparison[4]}")

            