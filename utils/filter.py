"""Filter comparison"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy import signal

fs=30
# def actigraph_matched(data, fs):
#     lowcut = 0.5127
#     highcut = 2.7130

#     nyq = fs / 2

#     wp = [lowcut / nyq, highcut / nyq]      # passband
#     ws = [0.3 / nyq, 5.0 / nyq]             # stopband (tune these!)

#     sos = signal.iirdesign(
#         wp=wp,
#         ws=ws,
#         gpass=1,    # passband ripple (dB)
#         gstop=40,   # stopband attenuation (dB)
#         ftype='ellip', 
#         output='sos'
#     )

#     return signal.sosfiltfilt(sos, data)

# def _actigraph_filter(data: np.ndarray) -> np.ndarray:
#         """
#         Apply the ActiGraph-specific filter to the data.

#         Parameters
#         ----------
#         data : np.ndarray
#                 The input signal data.

#         Returns
#         -------
#         np.ndarray
#                 The filtered data.
#         """
#         b = [0.04910898, -0.12284184, 0.14355788, -0.11269399, 0.05380374, -0.02023027,
#                 0.00637785, 0.01851254, -0.03815411, 0.04872652, -0.05257721, 0.04784714,
#                 -0.04601483, 0.03628334, -0.01297681, -0.00462621, 0.01283540, -0.00937622,
#                 0.00344850, -0.00080972, -0.00019623]
#         a = [1.00000000, -4.16372603, 7.57115309, -7.98046903, 5.38501191, -2.46356271,
#                 0.89238142, 0.06360999, -1.34810513, 2.47338133, -2.92571736, 2.92983230,
#                 -2.78159063, 2.47767354, -1.68473849, 0.46482863, 0.46565289, -0.67311897,
#                 0.41620323, -0.13832322, 0.01985172]
#         return signal.filtfilt(b, a, data)


b = [0.04910898, -0.12284184, 0.14355788, -0.11269399, 0.05380374, -0.02023027,
        0.00637785, 0.01851254, -0.03815411, 0.04872652, -0.05257721, 0.04784714,
        -0.04601483, 0.03628334, -0.01297681, -0.00462621, 0.01283540, -0.00937622,
        0.00344850, -0.00080972, -0.00019623]
a = [1.00000000, -4.16372603, 7.57115309, -7.98046903, 5.38501191, -2.46356271,
        0.89238142, 0.06360999, -1.34810513, 2.47338133, -2.92571736, 2.92983230,
        -2.78159063, 2.47767354, -1.68473849, 0.46482863, 0.46565289, -0.67311897,
        0.41620323, -0.13832322, 0.01985172]

lowcut = 0.3076171875
highcut = 1.6278076171875

nyq = fs / 2
wp = [lowcut / nyq, highcut / nyq]
ws = [0.3 / nyq, 5.0 / nyq]

## attempt 1 
# sos = signal.iirdesign(
#     wp=wp,
#     ws=ws,
#     gpass=1,
#     gstop=40,
#     ftype='ellip',
#     output='sos'
# )
## attempt 2
sos = signal.ellip(
    N=6,
    rp=1,      # passband ripple
    rs=40,     # stopband attenuation
    Wn=wp,
    btype='bandpass',
    output='sos'
)


# --- Frequency response ---
w, h_orig = signal.freqz(b, a, worN=8192)
w2, h_new = signal.sosfreqz(sos, worN=8192)

# Convert to Hz
freq = (w / (2 * np.pi)) * fs

# Magnitude in dB (normalized)
mag_orig = 20 * np.log10(np.abs(h_orig))
mag_new = 20 * np.log10(np.abs(h_new))

mag_orig -= np.max(mag_orig)
mag_new -= np.max(mag_new)

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(freq, mag_orig, label="Original ActiGraph filter")
plt.plot(freq, mag_new, linestyle='--', label="Elliptic approximation")

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Filter Frequency Response Comparison")
plt.axhline(-3, linestyle=':', label='-3 dB')
plt.legend()
plt.grid()

plt.xlim(0, 6)  # zoom into relevant region
plt.ylim(-80, 5)

plt.show()
