import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from collections import deque

# ----------------------------
# Serial & EEG config
# ----------------------------
ser = serial.Serial('COM7', 115200)  # CHANGE COM PORT
fs = 250  # Hz
window_size = 1000  # ~4 seconds

buffer = deque(maxlen=window_size)

# ----------------------------
# Band-pass filter
# ----------------------------
def bandpass_filter(data, lowcut=0.5, highcut=40, fs=250, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

# ----------------------------
# Plot setup
# ----------------------------
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

# ----------------------------
# Main loop
# ----------------------------
while True:
    try:
        value = int(ser.readline().decode().strip())
        buffer.append(value)

        if len(buffer) == window_size:
            eeg = np.array(buffer)
            eeg = eeg - np.mean(eeg)              # DC removal
            eeg = bandpass_filter(eeg)            # Filter

            # ---------- FFT ----------
            fft_vals = np.fft.rfft(eeg)
            fft_power = np.abs(fft_vals) ** 2
            freqs = np.fft.rfftfreq(len(eeg), 1/fs)

            # ---------- Time-domain plot ----------
            ax1.clear()
            ax1.plot(eeg)
            ax1.set_title("Filtered EEG (Time Domain)")
            ax1.set_ylabel("Amplitude")

            # ---------- Frequency-domain plot ----------
            ax2.clear()
            ax2.plot(freqs, fft_power)
            ax2.set_xlim(0, 40)
            ax2.set_title("EEG Frequency Spectrum (FFT)")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Power")

            plt.pause(0.01)

    except:
        pass
