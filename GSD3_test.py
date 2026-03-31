from typing_extensions import Self, Literal
import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
from multimob.GSD.utils.GSD3_utils import window, sum_partial_overlapping_windows, remove_outliers, calc_activity_parameter, resample_to_orginal_data_length, generate_gs_list
from ActivityCounts import ActivityCounts
# from multimob.GSD.utils.ActivityCounts import ActivityCounts
from multimob.GSD.utils.cwb import cwb
from mobgap.data_transform import (
    chain_transformers,
    ButterworthFilter
)


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
    '''
    def plot_acceleration_data(self, data: pd.DataFrame, sampling_rate_hz: float, 
                               title: str = "3-Axis Acceleration", 
                               xaxis: str = None,
                               yaxis: str = None,
                               vertical_lines=None, 
                               scale: float = 1.0) -> None:
        """
        Plot 3-axis acceleration data over time.

        Parameters
        ----------
        data : pd.DataFrame
            Input acceleration data with three axes.
        sampling_rate_hz : float
            Original sampling rate of the data.
        """
        if type(data) is np.ndarray:
            #names = 
            data = pd.DataFrame(data, columns=[title])

        time = np.arange(len(data)) / sampling_rate_hz
        cols = list(data.columns[:3])
        print('cols', cols)

        fig, ax = plt.subplots(figsize=(10, 4))
        for col in cols:
            ax.plot(time, data[col], label=col)

        if xaxis is not None:
            ax.set_xlabel(xaxis) 
        if yaxis is not None:
            ax.set_ylabel(yaxis)
        ax.set_title(title)
        ax.legend()
        # if self.switch is not None:
        #     sr = switch_sampling_rate if switch_sampling_rate is not None else sampling_rate_hz
        #     for sw in self.switch: 
        #         if 0 <= sw < len(data) * (sr / sampling_rate_hz):  # bounds check in original samples
        #             time_of_change = sw / sr  # convert switch index (original samples) → seconds
        #             ax.axvline(x=time_of_change, color='red', linestyle='--', alpha=0.7, linewidth=1)


        if vertical_lines is not None:
            for idx in vertical_lines:
                if 0 <= idx < len(data):
                    time_at_idx = time[idx]
                    time_at_idx = scale * time_at_idx
                    ax.axvline(x=time_at_idx, color='blue', linestyle='--', alpha=0.7, linewidth=1)

        #print("ploting ", len(data))
        #print('plot done')
        fig.tight_layout()
        fig.show()
        return fig
    '''
    def detect(self, data, *, sampling_rate_hz: float = 100) -> Self:
        """
        Detect gait sequences in wrist-worn accelerometer data.

        Parameters
        ----------
        data : pd.DataFrame
            Input data containing the three accelerometer axes (x, y, z).
            The algorithm uses the vector norm of these axes.
        sampling_rate_hz : float, optional
            The sampling rate of the input data in Hz (default: 100).

        Returns
        -------
        Self
            The instance of the class with detected gait sequences stored in the `gs_list_` attribute.
        """

        self.sampling_rate_hz = sampling_rate_hz
        self.data = data
        self.data_len = len(data)

        # In the current implementation for wrist worn sensors we use the norm
        acc = self.data.iloc[:, 0:3]
        norm_acc = np.linalg.norm(acc, axis=1)

        # Finds the activity counts per second
        # turning acc to g-units for activity counts calculation
        '''
        ideas: 
        - using a lowpass filter with cutoff of 0.25Hz was assumed to approximate the graity contribution from Hickey
        - the output of that lowpass is the gravity, so the output of the highpass is the dynamic motion? 
        '''
        norm_acc = norm_acc / 9.81

        activity_counts = ActivityCounts().calculate(data=norm_acc.copy(),
                                                     sampling_rate=self.sampling_rate_hz).activity_counts_
        
        # shortcut if all activity counts are 0 no gait can be detected
        if np.all(activity_counts == 0):
            self.gs_list_ = pd.DataFrame(columns=["start", "end"])
            self.gs_list_.index.name = 'gs_id'
            return self

        # Checks if activity counts are shorter than the window size
        if len(activity_counts) < self.win_size_s:
            raise ValueError(
                'The provided data stream is too short. It must be at least {}s long'.format(self.win_size_s))

        # Creates overlapping windows of activity counts data (activity counts are expressed in seconds)
        windows = window(activity_counts, self.win_size_s, self.win_shift_s, copy=True)
        # for 265 1s bin, windows are of size (257, 9) (265-9+1)


        # Outlier removal only when window size is 5 or higher otherwise this method might remove regular values
        if self.win_size_s > 4:
            # Removes outliers from the windows, the limits are configurable
            filtered_activity_counts = remove_outliers(windows.copy(), lower_percentile=self.lower_percentile, upper_percentile=self.upper_percentile)
        else:
            filtered_activity_counts = windows.copy()

        # Calculates the ratio of inactive data in each window
        inactivity_parameter = calc_activity_parameter(filtered_activity_counts)
        # inactivity_parameter has the size of the number of windows 

        # ---------- ADDITION FROM HICKEY ---------------------------------------
        # taking the std of the norm acc per window and make a threshold of how noisy the data needs to be        
        win_num = len(windows) 
        n = int(self.win_size_s * self.sampling_rate_hz)
        shift_samples = int(self.win_shift_s * self.sampling_rate_hz)
        # calculate std of each window 
        std_acc = np.zeros(win_num)
        # mean_acc = np.zeros(win_num)
        for i in range(win_num):
            start_idx = int(i * shift_samples)
            end_idx = start_idx + n

            # Ensure we don't exceed array bounds due to rounding
            if end_idx > len(norm_acc):
                end_idx = len(norm_acc)
            
            # Only calculate if we have data to avoid RuntimeWarnings
            if len(norm_acc[start_idx:end_idx]) > 0:
                std_acc[i] = np.std(norm_acc[start_idx:end_idx])
            else:
                std_acc[i] = 0

        # print("last i", i)
        # print("std_acc", std_acc)
        # print("std_acc len", len(std_acc))

        # self.threshold_still = 0.1
        th3 = 0.45
        # Assigns 1 to the windows where the inactivity parameter is below the walking threshold
        walking_windows = np.zeros(len(windows))
        for i in range(win_num):
            if std_acc[i] >= self.threshold_still \
                and inactivity_parameter[i] <= self.threshold \
                and inactivity_parameter[i] >= th3:
                walking_windows[i] = 1

        # Shows how many times each second's activity counts are included in the moving window
        detected_walking = sum_partial_overlapping_windows(walking_windows, activity_counts, self.win_size_s, self.win_shift_s)
        # Interpolates the walking windows to the original data length (True or False for all data points)
        detected_walking = resample_to_orginal_data_length(detected_walking, len(norm_acc)).astype(bool)
        """
        here the .astype(bool) taked any detected window to be true; 
        could add a check of detected_walking needs to be >1 for example 
        not sure if that makes sense ?

        didnt seem helpful i think? 
        """
        # print("len activity",len(activity_counts))
        # print("len detected_walking",len(detected_walking))
        # for i in range(activity_counts):
            
        #     if activity_counts[i] > 200:
                
        #         detected_walking[i] = 0 
        gs = generate_gs_list(detected_walking)
        # Clipping start and end to be within limits of file
        gs[['start', 'end']] = np.clip(gs[['start', 'end']], 0, len(self.data))

        # Creating Continuous Walking Bouts from micro walking bouts
        if self.cwb:
            gs = cwb(gs, max_break_seconds=3, sampling_rate=self.sampling_rate_hz)

        self.gs_list_ = gs

        return self


    def filters(self, data, *, sampling_rate_hz: float = 100) -> Self:

        self.sampling_rate_hz = sampling_rate_hz
        self.data = data
        self.data_len = len(data)

        # In the current implementation for wrist worn sensors we use the norm
        # from gsd 3 
        acc = self.data.iloc[:, 0:3]
        norm_acc = np.linalg.norm(acc, axis=1)
        norm_acc = norm_acc / 9.81
        ###############

        # from gsd2 
        cutoff = 1.75 #0.25
        # class instance
        filter_chain = [("butter", ButterworthFilter(order=1, cutoff_freq_hz=cutoff, filter_type='lowpass'))]

        acc_is = self.data.iloc[:, 0]
        acc_ml = self.data.iloc[:, 1]
        acc_pa = self.data.iloc[:, 2]
        # application to all corrected axes
        acc_is_filt = np.asarray(chain_transformers(acc_is, filter_chain, sampling_rate_hz=sampling_rate_hz))
        acc_ml_filt = np.asarray(chain_transformers(acc_ml, filter_chain, sampling_rate_hz=sampling_rate_hz))
        acc_pa_filt = np.asarray(chain_transformers(acc_pa, filter_chain, sampling_rate_hz=sampling_rate_hz))

        acc_is_no_grav = self.data.iloc[:, 0] - acc_is_filt
        acc_ml_no_grav = self.data.iloc[:, 1] - acc_ml_filt
        acc_pa_no_grav = self.data.iloc[:, 2] - acc_pa_filt

        acc_no_grav = pd.concat([acc_is_no_grav, acc_ml_no_grav, acc_pa_no_grav], axis=1)
        acc_norm = np.linalg.norm(acc_no_grav, axis=1)

    
        time = np.arange(len(data)) / sampling_rate_hz
        time_sec = np.arange(len(data)/ sampling_rate_hz)
        # fig, ax = plt.subplots(figsize=(10, 4))
        fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        ax1.plot(time, acc_is, label = 'acc_is')
        ax1.plot(time, acc_ml, label = 'acc_ml')
        ax1.plot(time, acc_pa, label = 'acc_pa')
 
        ax1.set_ylabel("Raw acceleration")
        ax1.set_title("Accelerations")
        ax1.legend()
        # ------------------------
        ax2.plot(time, acc_is_filt, label = 'acc_is')
        ax2.plot(time, acc_ml_filt, label = 'acc_ml')
        ax2.plot(time, acc_pa_filt, label = 'acc_pa')
 
        ax2.set_ylabel("Grav component")
        ax2.legend()
        # --------------------------------
        ax3.plot(time, acc_is_no_grav, label = 'acc_is')
        ax3.plot(time, acc_ml_no_grav, label = 'acc_ml')
        ax3.plot(time, acc_pa_no_grav, label = 'acc_pa')
 
        ax3.set_ylabel("No grav")
        ax3.set_xlabel("Time(s)")
        ax3.legend()

        fig1.tight_layout()
        fig1.show()

        ######################################
        fig2, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
        ax.plot(time, acc_norm, label = 'norm from GSD2')
        ax.plot(time, norm_acc, label = 'norm from GSD3')
        ax.set_ylabel("acc norm")
        ax.set_xlabel("Time(s)")
        ax.legend()

        fig2.tight_layout()
        fig2.show()

        #######################################
        activity_counts_3 = ActivityCounts().calculate(data=norm_acc.copy(),
                                                     sampling_rate=self.sampling_rate_hz).activity_counts_
        activity_counts_2 = ActivityCounts().calculate(data=acc_norm.copy(),
                                                     sampling_rate=self.sampling_rate_hz).activity_counts_
        
        fig3, ax = plt.subplots(1, 1, figsize=(10, 4), sharex=True)
        ax.plot(time_sec, activity_counts_2, label = 'Activity from GSD2')
        ax.plot(time_sec, activity_counts_3, label = 'Activity from GSD3')
        ax.set_ylabel("Activity count")
        ax.set_xlabel("Time(s)")
        ax.legend()

        fig3.tight_layout()
        fig3.show()



        '''
        ideas: 
        - using a lowpass filter with cutoff of 0.25Hz was assumed to approximate the graity contribution from Hickey
        - the output of that lowpass is the gravity, so the output of the highpass is the dynamic motion? 
        '''

        return self
     
    def get_activity(self, data, *, sampling_rate_hz: float = 100):
        self.sampling_rate_hz = sampling_rate_hz
        self.data = data
        self.data_len = len(data)

        # In the current implementation for wrist worn sensors we use the norm
        acc = self.data.iloc[:, 0:3]
        norm_acc = np.linalg.norm(acc, axis=1)

        # Finds the activity counts per second
        # turning acc to g-units for activity counts calculation
        norm_acc = norm_acc / 9.81

        activity_counts = ActivityCounts().calculate(data=norm_acc.copy(),
                                                     sampling_rate=self.sampling_rate_hz).activity_counts_

        if len(activity_counts) < self.win_size_s:
            raise ValueError(
                'The provided data stream is too short. It must be at least {}s long'.format(self.win_size_s))

        # Creates overlapping windows of activity counts data (activity counts are expressed in seconds)
        windows = window(activity_counts, self.win_size_s, self.win_shift_s, copy=True)
        # for 265 1s bin, windows are of size (257, 9) (265-9+1)


        # Outlier removal only when window size is 5 or higher otherwise this method might remove regular values
        if self.win_size_s > 4:
            # Removes outliers from the windows, the limits are configurable
            filtered_activity_counts = remove_outliers(windows.copy(), lower_percentile=self.lower_percentile, upper_percentile=self.upper_percentile)
        else:
            filtered_activity_counts = windows.copy()

        # Calculates the ratio of inactive data in each window
        inactivity_parameter = calc_activity_parameter(filtered_activity_counts)
        # print()
        # # print("inactivity_parameter is", inactivity_parameter)
        # print("len inactivity_parameter of", len(inactivity_parameter))
        # print()

        # Assigns 1 to the windows where the inactivity parameter is below the walking threshold
        walking_windows = np.zeros(len(windows))
        walking_windows[inactivity_parameter < self.threshold] = 1
        detected_walking = sum_partial_overlapping_windows(walking_windows, activity_counts, self.win_size_s, self.win_shift_s)
        detected_walking = resample_to_orginal_data_length(detected_walking, len(norm_acc)) #.astype(bool)

        
        return activity_counts, walking_windows
    
    def get_std_norm(self, data, *, sampling_rate_hz: float = 100):
        self.sampling_rate_hz = sampling_rate_hz
        self.data = data
        self.data_len = len(data)

        # In the current implementation for wrist worn sensors we use the norm
        acc = self.data.iloc[:, 0:3]
        norm_acc = np.linalg.norm(acc, axis=1)

        # Finds the activity counts per second
        # turning acc to g-units for activity counts calculation
        norm_acc = norm_acc / 9.81

        activity_counts = ActivityCounts().calculate(data=norm_acc.copy(),
                                                     sampling_rate=self.sampling_rate_hz).activity_counts_
        
        # Checks if activity counts are shorter than the window size
        if len(activity_counts) < self.win_size_s:
            raise ValueError(
                'The provided data stream is too short. It must be at least {}s long'.format(self.win_size_s))

        # Creates overlapping windows of activity counts data (activity counts are expressed in seconds)
        windows = window(activity_counts, self.win_size_s, self.win_shift_s, copy=True)

        # ---------- ADDITION FROM HICKEY ---------------------------------------
        # taking the std of the norm acc per window and make a threshold of how noisy the data needs to be        
        win_num = len(windows) 
        n = int(self.win_size_s * self.sampling_rate_hz)
        shift_samples = int(self.win_shift_s * self.sampling_rate_hz)
        # calculate std of each window 
        std_acc = np.zeros(win_num)
        # mean_acc = np.zeros(win_num)
        for i in range(win_num):
            start_idx = int(i * shift_samples)
            end_idx = start_idx + n

            # Ensure we don't exceed array bounds due to rounding
            if end_idx > len(norm_acc):
                end_idx = len(norm_acc)
            
            # Only calculate if we have data to avoid RuntimeWarnings
            if len(norm_acc[start_idx:end_idx]) > 0:
                std_acc[i] = np.std(norm_acc[start_idx:end_idx])
            else:
                std_acc[i] = 0
        return std_acc

    