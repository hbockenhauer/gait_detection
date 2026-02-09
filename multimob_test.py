from multimob.utils.data_loader import load_imu_data
from multimob.GSD.GSD3 import KheirkhahanGSD
from multimob.GSD.GSD2 import HickeyGSD

    
import pandas as pd
from importlib import resources

'''
def load_imu_data() -> pd.DataFrame:
    # Load accelerometer data
    with resources.open_text("HAR_data", "Watch_accelerometer.csv") as f:
        acc = pd.read_csv(f)

    # Load gyroscope data
    with resources.open_text("HAR_data", "Watch_gyroscope.csv") as f:
        gyr = pd.read_csv(f)

    # Rename axes to match IMU convention
    acc = acc.rename(columns={
        "x": "acc_is",
        "y": "acc_ml",
        "z": "acc_pa"
    })

    gyr = gyr.rename(columns={
        "x": "gyr_is",
        "y": "gyr_ml",
        "z": "gyr_pa"
    })

    label = acc[["Index", "gt"]].set_index("Index")
    # Keep only needed columns (+ Index for merge)
    acc = acc[["Index", "acc_is", "acc_ml", "acc_pa"]]
    gyr = gyr[["Index", "gyr_is", "gyr_ml", "gyr_pa"]]

    # Merge accelerometer and gyroscope data
    imu_data = pd.merge(acc, gyr, on="Index", how="inner")
    

    return imu_data, label
'''

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

#N = 100000
M = 10000000
imu_data = imu_data.iloc[:M]


# Preprocess and detect gait events
GSDs = KheirkhahanGSD().detect(imu_data, sampling_rate_hz=100)

print(GSDs.gs_list_)

acc =  walking_accuracy_from_df(labels, GSDs.gs_list_)
print(f"Walking accuracy: {acc:.3f}")