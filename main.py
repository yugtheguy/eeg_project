
import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from collections import deque

# ----------------------------
# Serial config
# ----------------------------
ser = serial.Serial('COM7', 115200)  # CHANGE COM PORT
fs = 250  # Sampling frequency (Hz)

# ----------------------------
# Buffers (for left and right channels)
# ----------------------------
raw_left_buffer = deque(maxlen=1000)
raw_right_buffer = deque(maxlen=1000)
filt_left_buffer = deque(maxlen=1000)
filt_right_buffer = deque(maxlen=1000)

# ----------------------------
# Bandpass filter design
# ----------------------------
def bandpass_filter(data, lowcut=0.5, highcut=40, fs=250, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# ----------------------------
# Plot setup
# ----------------------------
plt.ion()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8))
plt.show(block=False)  # Show the figure window
fig.canvas.draw()
fig.canvas.flush_events()

# ----------------------------
# Main loop
# ----------------------------
while True:
    try:
        # Read and parse CSV line: timestamp,left,right
        line = ser.readline().decode().strip()
        parts = line.split(',')
        
        if len(parts) == 3:
            timestamp = int(parts[0])
            left_value = int(parts[1])
            right_value = int(parts[2])
            
            raw_left_buffer.append(left_value)
            raw_right_buffer.append(right_value)

            if len(raw_left_buffer) > 100:
                # Convert to arrays
                raw_left = np.array(raw_left_buffer)
                raw_right = np.array(raw_right_buffer)

                # Remove DC offset
                raw_left = raw_left - np.mean(raw_left)
                raw_right = raw_right - np.mean(raw_right)

                # Filter both channels
                filt_left = bandpass_filter(raw_left)
                filt_right = bandpass_filter(raw_right)

                filt_left_buffer.clear()
                filt_left_buffer.extend(filt_left)
                filt_right_buffer.clear()
                filt_right_buffer.extend(filt_right)

                # Plot raw left
                ax1.clear()
                ax1.plot(raw_left, 'b')
                ax1.set_title("Raw LEFT Channel (DC Removed)")
                ax1.set_ylabel("Amplitude")

                # Plot raw right
                ax2.clear()
                ax2.plot(raw_right, 'r')
                ax2.set_title("Raw RIGHT Channel (DC Removed)")
                ax2.set_ylabel("Amplitude")

                # Plot filtered left
                ax3.clear()
                ax3.plot(filt_left_buffer, 'b')
                ax3.set_title("Filtered LEFT (0.5–40 Hz)")
                ax3.set_ylabel("Amplitude")
                ax3.set_xlabel("Samples")

                # Plot filtered right
                ax4.clear()
                ax4.plot(filt_right_buffer, 'r')
                ax4.set_title("Filtered RIGHT (0.5–40 Hz)")
                ax4.set_ylabel("Amplitude")
                ax4.set_xlabel("Samples")

                plt.tight_layout()
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)

    except KeyboardInterrupt:
        print("\nStopping...")
        break
    except Exception as e:
        # Silently skip malformed lines
        pass

ser.close()
plt.close()
