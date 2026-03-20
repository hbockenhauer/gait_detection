import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.paths import QSENSE_MIXED, QSENSE_CLINIC

DATASET_PATH = QSENSE_CLINIC  # Change this to QSENSE_MIXED or QSENSE_TEST as needed

def compare_leg_class(filepath):
    df = pd.read_csv(filepath, sep=None, engine="python")
    df = df.reset_index(drop=True)

    parent_folder = os.path.basename(os.path.dirname(filepath))
    dataset_folder = os.path.basename(os.path.dirname(os.path.dirname(filepath)))

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

    #smooth the predicted class to remove noise for 2s windows (100 samples at 50Hz) and take median
    # df[class_col] = df[class_col].rolling(window=100, center=True).median()

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

        # Plot y_pred & y_true vs time for this class
        plot_df = pd.DataFrame({
            'y_true': [1.1 if label == 1 else 0 for label in df[label_col]],
            'y_pred': [0.9 if pred == 3 else 0 for pred in df[class_col]],
            'time': df.index/50  # Assuming index represents sample order
        })

        plot_path = os.path.join(PROJECT_ROOT, 'outputs', 'plots', dataset_folder, 'Leg Classification', f'{parent_folder}_leg_comparison.png')
        plot_df.to_csv(plot_path, index=False)
        plt.figure(figsize=(12, 6))
        plt.plot(plot_df['time'], plot_df['y_true'], linewidth=1.5, label='True Gait')
        plt.plot(plot_df['time'], plot_df['y_pred'], linewidth=1.5, label='Predicted Gait')
        plt.title(f'{parent_folder} - Comparison of True vs Predicted Gait using Leg Sensor {cls}')
        plt.xlabel('Time (s)')
        plt.ylabel('Class Label')
        plt.legend()
        plt.savefig(plot_path)
        plt.close()


    return conf, pct_correct_walking_as_3, pct_predicted_3_correct, pct_dataset_is_x3_1, results

if __name__ == "__main__":
    f1_scores =[]
    total_samples = []
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
            conf, pct_correct_walking_as_3, pct_predicted_3_correct, pct_dataset_is_x3_1, results = compare_leg_class(file)
            print(f"Comparison for {file}: {conf}")
            print(f"Recall: {results[3]['recall'].round(2)}%")
            print(f"Precision: {results[3]['precision'].round(2)}%")
            print(f"F1 Score: {results[3]['f1'].round(2)}%")
            #print(f"Detailed results: {results}")

        f1_scores.append(results[3]['f1'].round(2))
        total_samples.append(conf.values.sum())
            
    sum_f1 = 0
    for i in range(len(f1_scores)):
        sum_f1 += f1_scores[i] * total_samples[i]
        global_f1 = sum_f1 / sum(total_samples) if sum(total_samples) > 0 else 0
    print(f"\nGlobal average F1 score for class 3 (walking) across all subjects: {global_f1:.4f}%")