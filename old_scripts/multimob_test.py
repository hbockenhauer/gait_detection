"""initial attempt used for loading WISDM dataset """
from multimob.utils.data_loader import load_imu_data
from multimob.GSD.GSD3 import KheirkhahanGSD
from multimob.GSD.GSD2 import HickeyGSD
    
import pandas as pd
from importlib import resources


def load_imu_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    csv_path = resources.files("WearGait_ctrl") / "WHC014_FreeWalk.csv"
    df = pd.read_csv(csv_path)

    # ---- Select IMU signals (Left Wrist) ----
    imu_data = df[
        [
            "L_Wrist_Acc_X",
            "L_Wrist_Acc_Y",
            "L_Wrist_Acc_Z",
            "L_Wrist_Gyr_X",
            "L_Wrist_Gyr_Y",
            "L_Wrist_Gyr_Z",
        ]
    ].copy()

    imu_data.columns = [
        "acc_x", "acc_y", "acc_z",
        "gyr_x", "gyr_y", "gyr_z"
    ]

    # ---- Build walking label ----
    # Binary label: True = walking
    labels = pd.DataFrame({
        "Walk": df["GeneralEvent"].str.lower() == "walk"
    })

    # ---- Ensure alignment ----
    imu_data.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)

    return imu_data, labels



def window_gt_label_from_label_df(
    label_df: pd.DataFrame,
    start: int,
    end: int
) -> str:
    window_labels = label_df[
        (label_df.index >= start) & (label_df.index <= end)
    ]["gt"]

    if window_labels.empty:
        return "null"

    return window_labels.mode().iloc[0]


def walking_accuracy_from_df(
    label_df: pd.DataFrame,
    windows: pd.DataFrame
) -> float:
    if len(windows) == 0:
        return 0.0

    correct = 0

    for _, row in windows.iterrows():
        start = row["start"]
        end = row["end"]

        gt_label = window_gt_label_from_label_df(label_df, start, end)

        if gt_label == "walk":
            correct += 1

    return correct / len(windows)

imu_data, labels = load_imu_data()

M = 10000000
imu_data = imu_data.iloc[:M]


# Preprocess and detect gait events
GSDs = KheirkhahanGSD().detect(imu_data, sampling_rate_hz=100)

print(GSDs.gs_list_)

acc =  walking_accuracy_from_df(labels, GSDs.gs_list_)
print(f"Walking accuracy: {acc:.3f}")