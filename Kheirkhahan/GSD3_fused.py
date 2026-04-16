"""
Fusing the detection of the two wrists at the activity level. 
"""

from typing_extensions import Self, Literal
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from multimob.GSD.utils.GSD3_utils import window, sum_partial_overlapping_windows, remove_outliers, calc_activity_parameter, resample_to_orginal_data_length, generate_gs_list
from realtime.ActivityCounts import ActivityCounts
# from multimob.GSD.utils.ActivityCounts import ActivityCounts
from multimob.GSD.utils.cwb import cwb
# from mobgap.data_transform import (
#     chain_transformers,
#     ButterworthFilter
# )


class KheirkhahanGSD:
    """
    Implementation of the Gait Sequence Detection algorithm by Kheirkhahan et al. (2017) [1].

    This implementation is adapted and fine tuned for wrist-worn accelerometer data.
    The algorithm detects gait sequences from the data by following these steps:

    1. Preprocessing: The input accelerometer data is preprocessed by calculating the norm of the three axes.
    2. Activity Counts Calculation: The norm signal is used to calculate activity counts per second.
    3. Windowing and Outlier Removal: The activity counts are divided into overlapping windows, and outliers are removed.
    4. Activity Parameter Calculation: An inactivity parameter is calculated for each window.
    5. Walking Detection: Windows with inactivity parameters below a threshold are marked as walking.
    6. Sequence Generation: The detected walking windows are interpolated to the original data length, and gait sequences are generated.

    [1] Kheirkhahan, M., et al. Adaptive walk detection algorithm using activity counts.
        In 2017 IEEE EMBS International Conference on Biomedical & Health Informatics (BHI). 2017.

    Attributes
    ----------
    gs_list_ : pd.DataFrame
        The detected gait sequences.

    Notes
    -----
    - Implementation is wrist-only (other axis-based versions are not included).
    - Uses a 9-second sliding window with 1-second shift for detection.
    - Algorithm works with sampling rate ≥ 30 Hz.
    - Data are converted from m/s² to g-units before activity counts calculation.
    - Optionally, detected micro walking bouts can be merged into Continuous Walking Bouts (CWB).
    """


    def __init__(self, *, version: Literal["wrist"] = "wrist", cwb: bool=True, threshold_still: float = 0.0):
        """
        Initialize the class.

        Parameters
        ----------
        version : str, optional
            The version of the algorithm to use. For this release the only option is "wrist".
        cwb : bool, optional
            Whether to create Continuous Walking Bouts from micro walking bouts (default is True).
        """

        self.version = version
        self.axis = "norm"
        self.lower_percentile = 20
        self.upper_percentile = 90
        self.win_size_s = 9
        self.win_shift_s = 1
        self.threshold = 0.62
        self.cwb = cwb
        # self.visual = visual
        self.threshold_still = threshold_still

    # def detect(self, data1, data2, *, sampling_rate_hz: float = 100) -> Self:
    #     """
    #     Detect gait sequences in wrist-worn accelerometer data.

    #     Parameters
    #     ----------
    #     data : pd.DataFrame
    #         Input data containing the three accelerometer axes (x, y, z).
    #         The algorithm uses the vector norm of these axes.
    #     sampling_rate_hz : float, optional
    #         The sampling rate of the input data in Hz (default: 100).

    #     Returns
    #     -------
    #     Self
    #         The instance of the class with detected gait sequences stored in the `gs_list_` attribute.
    #     """

    #     self.sampling_rate_hz = sampling_rate_hz
    #     self.data = data
    #     self.data_len = len(data)

    #     # In the current implementation for wrist worn sensors we use the norm
    #     acc = self.data.iloc[:, 0:3]
    #     norm_acc = np.linalg.norm(acc, axis=1)

    #     # Finds the activity counts per second
    #     # turning acc to g-units for activity counts calculation
    #     norm_acc = norm_acc / 9.81

    #     activity_counts = ActivityCounts().calculate(data=norm_acc.copy(),
    #                                                  sampling_rate=self.sampling_rate_hz).activity_counts_
        
    #     # shortcut if all activity counts are 0 no gait can be detected
    #     if np.all(activity_counts == 0):
    #         self.gs_list_ = pd.DataFrame(columns=["start", "end"])
    #         self.gs_list_.index.name = 'gs_id'
    #         return self

    #     # Checks if activity counts are shorter than the window size
    #     if len(activity_counts) < self.win_size_s:
    #         raise ValueError(
    #             'The provided data stream is too short. It must be at least {}s long'.format(self.win_size_s))

    #     # Creates overlapping windows of activity counts data (activity counts are expressed in seconds)
    #     windows = window(activity_counts, self.win_size_s, self.win_shift_s, copy=True)


    #     # Outlier removal only when window size is 5 or higher otherwise this method might remove regular values
    #     if self.win_size_s > 4:
    #         # Removes outliers from the windows, the limits are configurable
    #         filtered_activity_counts = remove_outliers(windows.copy(), lower_percentile=self.lower_percentile, upper_percentile=self.upper_percentile)
    #     else:
    #         filtered_activity_counts = windows.copy()

    #     # Calculates the ratio of inactive data in each window
    #     inactivity_parameter = calc_activity_parameter(filtered_activity_counts)

    #     # ---------- ADDITION FROM HICKEY ---------------------------------------
    #     # taking the std of the norm acc per window and make a threshold of how noisy the data needs to be        
    #     win_num = len(windows) 
    #     n = int(self.win_size_s * self.sampling_rate_hz)
    #     shift_samples = int(self.win_shift_s * self.sampling_rate_hz)
    #     # calculate std of each window 
    #     std_acc = np.zeros(win_num)
    #     for i in range(win_num):
    #         start_idx = int(i * shift_samples)
    #         end_idx = start_idx + n

    #         # Ensure we don't exceed array bounds due to rounding
    #         if end_idx > len(norm_acc):
    #             end_idx = len(norm_acc)
            
    #         # Only calculate if we have data to avoid RuntimeWarnings
    #         if len(norm_acc[start_idx:end_idx]) > 0:
    #             std_acc[i] = np.std(norm_acc[start_idx:end_idx])
    #         else:
    #             std_acc[i] = 0

    #     # Assigns 1 to the windows where the inactivity parameter is below the walking threshold
    #     walking_windows = np.zeros(len(windows))
    #     for i in range(win_num):
    #         if std_acc[i] >= self.threshold_still \
    #             and inactivity_parameter[i] <= self.threshold:
    #             walking_windows[i] = 1

    #     # Shows how many times each second's activity counts are included in the moving window
    #     detected_walking = sum_partial_overlapping_windows(walking_windows, activity_counts, self.win_size_s, self.win_shift_s)
    #     # Interpolates the walking windows to the original data length (True or False for all data points)
    #     detected_walking = resample_to_orginal_data_length(detected_walking, len(norm_acc)).astype(bool)

    #     gs = generate_gs_list(detected_walking)
    #     # Clipping start and end to be within limits of file
    #     gs[['start', 'end']] = np.clip(gs[['start', 'end']], 0, len(self.data))

    #     # Creating Continuous Walking Bouts from micro walking bouts
    #     if self.cwb:
    #         gs = cwb(gs, max_break_seconds=3, sampling_rate=self.sampling_rate_hz)

    #     self.gs_list_ = gs

    #     return self
 
    def _detect_single(self, data, sampling_rate_hz):
        acc = data.iloc[:, 0:3]
        norm_acc = np.linalg.norm(acc, axis=1)
        norm_acc = norm_acc / 9.81

        activity_counts = ActivityCounts().calculate(
            data=norm_acc.copy(),
            sampling_rate=sampling_rate_hz
        ).activity_counts_

        if np.all(activity_counts == 0):
            return pd.DataFrame(columns=["start", "end"])

        if len(activity_counts) < self.win_size_s:
            raise ValueError(
                f'The provided data stream is too short. It must be at least {self.win_size_s}s long'
            )

        windows = window(activity_counts, self.win_size_s, self.win_shift_s, copy=True)

        if self.win_size_s > 4:
            filtered_activity_counts = remove_outliers(
                windows.copy(),
                lower_percentile=self.lower_percentile,
                upper_percentile=self.upper_percentile
            )
        else:
            filtered_activity_counts = windows.copy()

        inactivity_parameter = calc_activity_parameter(filtered_activity_counts)

        win_num = len(windows)
        n = int(self.win_size_s * sampling_rate_hz)
        shift_samples = int(self.win_shift_s * sampling_rate_hz)

        std_acc = np.zeros(win_num)
        for i in range(win_num):
            start_idx = int(i * shift_samples)
            end_idx = min(start_idx + n, len(norm_acc))
            if len(norm_acc[start_idx:end_idx]) > 0:
                std_acc[i] = np.std(norm_acc[start_idx:end_idx])

        walking_windows = np.zeros(win_num)
        for i in range(win_num):
            if std_acc[i] >= self.threshold_still and inactivity_parameter[i] <= self.threshold:
                walking_windows[i] = 1

        detected_walking = sum_partial_overlapping_windows(
            walking_windows, activity_counts,
            self.win_size_s, self.win_shift_s
        )

        detected_walking = resample_to_orginal_data_length(
            detected_walking, len(norm_acc)
        ).astype(bool)

        gs = generate_gs_list(detected_walking)
        gs[['start', 'end']] = np.clip(gs[['start', 'end']], 0, len(data))

        if self.cwb:
            gs = cwb(gs, max_break_seconds=3, sampling_rate=sampling_rate_hz)

        return gs
    
    def detect(self, data1=None, data2=None, *, sampling_rate_hz: float = 100) -> Self:
        self.sampling_rate_hz = sampling_rate_hz

        # Check availability
        has_data1 = data1 is not None and not data1.empty
        has_data2 = data2 is not None and not data2.empty

        # CASE 1: both datasets present
        if has_data1 and has_data2:
            # first dataset to activity count
            if len(data1) != len(data2):
                print("different lengths!!!!")
            acc1 = data1.iloc[:, 0:3]
            norm_acc1 = np.linalg.norm(acc1, axis=1)
            norm_acc1 = norm_acc1 / 9.81


            activity_counts1 = ActivityCounts().calculate(
                data=norm_acc1.copy(),
                sampling_rate=sampling_rate_hz
            ).activity_counts_

            # second dataset to activity count 
            acc2 = data2.iloc[:, 0:3]
            norm_acc2 = np.linalg.norm(acc2, axis=1)
            norm_acc2 = norm_acc2 / 9.81

            activity_counts2 = ActivityCounts().calculate(
                data=norm_acc2.copy(),
                sampling_rate=sampling_rate_hz
            ).activity_counts_

            # fuse the activities 
            activity_counts = activity_counts1 + activity_counts2


            if np.all(activity_counts == 0):
                return pd.DataFrame(columns=["start", "end"])

            if len(activity_counts) < self.win_size_s:
                raise ValueError(
                    f'The provided data stream is too short. It must be at least {self.win_size_s}s long'
                )

            windows = window(activity_counts, self.win_size_s, self.win_shift_s, copy=True)

            if self.win_size_s > 4:
                filtered_activity_counts = remove_outliers(
                    windows.copy(),
                    lower_percentile=self.lower_percentile,
                    upper_percentile=self.upper_percentile
                )
            else:
                filtered_activity_counts = windows.copy()

            inactivity_parameter = calc_activity_parameter(filtered_activity_counts)

            win_num = len(windows)
            n = int(self.win_size_s * sampling_rate_hz)
            shift_samples = int(self.win_shift_s * sampling_rate_hz)

            std_acc1 = np.zeros(win_num)
            std_acc2 = np.zeros(win_num)
            for i in range(win_num):
                start_idx = int(i * shift_samples)
                end_idx1 = min(start_idx + n, len(norm_acc1))
                if len(norm_acc1[start_idx:end_idx1]) > 0:
                    std_acc1[i] = np.std(norm_acc1[start_idx:end_idx1])

                # start_idx = int(i * shift_samples)
                end_idx2 = min(start_idx + n, len(norm_acc2))
                if len(norm_acc2[start_idx:end_idx2]) > 0:
                    std_acc2[i] = np.std(norm_acc1[start_idx:end_idx2])

            walking_windows = np.zeros(win_num)
            for i in range(win_num):
                if std_acc1[i] >= self.threshold_still and std_acc2[i] >= self.threshold_still \
                    and inactivity_parameter[i] <= self.threshold:
                    walking_windows[i] = 1

            detected_walking = sum_partial_overlapping_windows(
                walking_windows, activity_counts,
                self.win_size_s, self.win_shift_s
            )

            detected_walking = resample_to_orginal_data_length(
                detected_walking, len(norm_acc1)
            ).astype(bool)

            gs = generate_gs_list(detected_walking)
            gs[['start', 'end']] = np.clip(gs[['start', 'end']], 0, len(data1))

            if self.cwb:
                gs = cwb(gs, max_break_seconds=3, sampling_rate=sampling_rate_hz)

            self.gs_list_ = gs

        # CASE 2: only data1
        elif has_data1:
            self.gs_list_ = self._detect_single(data1, sampling_rate_hz)

        # CASE 3: only data2
        elif has_data2:
            self.gs_list_ = self._detect_single(data2, sampling_rate_hz)

        # CASE 4: no valid data
        else:
            return self

        self.gs_list_.index.name = 'gs_id'
        return self