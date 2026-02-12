"""
Real-Time EEG Alpha Band Monitor
Reads dual-channel EEG from Arduino, filters alpha band (8-12 Hz), 
and displays real-time signal and power analysis.
"""

import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch
from collections import deque
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
SERIAL_PORT = 'COM7'
BAUD_RATE = 115200
SAMPLING_RATE = 250  # Hz
BUFFER_DURATION = 5  # seconds
BUFFER_SIZE = SAMPLING_RATE * BUFFER_DURATION  # 1250 samples

# Filter parameters
ALPHA_LOW = 8   # Hz
ALPHA_HIGH = 12 # Hz
FILTER_ORDER = 4

# FFT parameters
FFT_WINDOW = 1.0  # seconds
FFT_SAMPLES = int(SAMPLING_RATE * FFT_WINDOW)  # 250 samples
ALPHA_POWER_BUFFER = 200  # Keep 200 power estimates for plotting

# ============================================================================
# FILTER DESIGN
# ============================================================================
def design_alpha_filter(lowcut, highcut, fs, order=4):
    """Design Butterworth bandpass filter for alpha band."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Pre-compute filter coefficients
b_alpha, a_alpha = design_alpha_filter(ALPHA_LOW, ALPHA_HIGH, SAMPLING_RATE, FILTER_ORDER)

# ============================================================================
# SIGNAL PROCESSING FUNCTIONS
# ============================================================================
def remove_dc(signal):
    """Remove DC offset by subtracting mean."""
    return signal - np.mean(signal)

def apply_alpha_filter(signal, b, a):
    """Apply pre-computed alpha bandpass filter."""
    if len(signal) < 3 * max(len(a), len(b)):
        return signal  # Not enough samples for stable filtering
    return filtfilt(b, a, signal)

def compute_alpha_power(signal, fs):
    """
    Compute average power in alpha band (8-12 Hz) using Welch's method.
    Returns power in dB.
    """
    if len(signal) < FFT_SAMPLES:
        return 0.0
    
    # Use last FFT_SAMPLES for power calculation
    segment = signal[-FFT_SAMPLES:]
    
    # Welch's method for PSD
    freqs, psd = welch(segment, fs=fs, nperseg=min(128, len(segment)), 
                       scaling='density', detrend='constant')
    
    # Find indices for alpha band
    alpha_idx = (freqs >= ALPHA_LOW) & (freqs <= ALPHA_HIGH)
    
    if not np.any(alpha_idx):
        return 0.0
    
    # Integrate power in alpha band
    alpha_power = np.trapezoid(psd[alpha_idx], freqs[alpha_idx])
    
    # Convert to dB
    alpha_power_db = 10 * np.log10(alpha_power + 1e-12)
    
    return alpha_power_db

# ============================================================================
# DATA BUFFERS
# ============================================================================
left_buffer = deque(maxlen=BUFFER_SIZE)
alpha_power_history = deque(maxlen=ALPHA_POWER_BUFFER)
time_axis = np.linspace(0, BUFFER_DURATION, BUFFER_SIZE)

# ============================================================================
# SERIAL CONNECTION
# ============================================================================
print(f"Connecting to {SERIAL_PORT} at {BAUD_RATE} baud...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino to reset
    print("✓ Connected")
    
    # Flush initial garbage data
    for _ in range(10):
        ser.readline()
    
except Exception as e:
    print(f"✗ Failed to connect: {e}")
    exit(1)

# ============================================================================
# MATPLOTLIB SETUP
# ============================================================================
plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('Real-Time EEG Alpha Monitor (LEFT Channel)', fontsize=14, fontweight='bold')

# Initialize plot lines
line_raw, = ax1.plot([], [], 'b-', linewidth=0.8, label='Raw (DC removed)')
line_alpha, = ax2.plot([], [], 'g-', linewidth=1.0, label='Alpha filtered (8-12 Hz)')
line_power, = ax3.plot([], [], 'r-', linewidth=1.5, label='Alpha power')

# Configure axes
ax1.set_xlim(0, BUFFER_DURATION)
ax1.set_ylim(-200, 200)
ax1.set_ylabel('Amplitude (ADC units)')
ax1.set_title('Raw EEG Signal (DC Removed)')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

ax2.set_xlim(0, BUFFER_DURATION)
ax2.set_ylim(-100, 100)
ax2.set_ylabel('Amplitude (ADC units)')
ax2.set_title('Alpha Band (8-12 Hz) Filtered Signal')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

ax3.set_xlim(0, ALPHA_POWER_BUFFER)
ax3.set_ylim(-40, 20)
ax3.set_xlabel('Time (samples)')
ax3.set_ylabel('Power (dB)')
ax3.set_title('Alpha Band Power Over Time')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right')

plt.tight_layout()
plt.show(block=False)
fig.canvas.draw()
fig.canvas.flush_events()

# ============================================================================
# MAIN LOOP
# ============================================================================
print("\nStreaming data... Press Ctrl+C to stop\n")
sample_count = 0
update_interval = 10  # Update plot every N samples

try:
    while True:
        try:
            # Read and parse serial line
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if not line:
                continue
            
            parts = line.split(',')
            
            # Parse: timestamp,left,right
            if len(parts) == 3:
                timestamp = int(parts[0])
                left_value = int(parts[1])
                right_value = int(parts[2])
                
                # Store left channel only
                left_buffer.append(left_value)
                sample_count += 1
                
                # Update plot periodically
                if sample_count % update_interval == 0 and len(left_buffer) >= FFT_SAMPLES:
                    
                    # Convert buffer to array
                    raw_signal = np.array(left_buffer)
                    
                    # Remove DC offset
                    raw_dc_removed = remove_dc(raw_signal)
                    
                    # Apply alpha filter
                    alpha_filtered = apply_alpha_filter(raw_dc_removed, b_alpha, a_alpha)
                    
                    # Compute alpha power
                    alpha_power = compute_alpha_power(raw_dc_removed, SAMPLING_RATE)
                    alpha_power_history.append(alpha_power)
                    
                    # Update raw signal plot
                    current_time = time_axis[:len(raw_dc_removed)]
                    line_raw.set_data(current_time, raw_dc_removed)
                    ax1.set_ylim(np.min(raw_dc_removed) - 20, np.max(raw_dc_removed) + 20)
                    
                    # Update alpha filtered plot
                    line_alpha.set_data(current_time, alpha_filtered)
                    ax2.set_ylim(np.min(alpha_filtered) - 10, np.max(alpha_filtered) + 10)
                    
                    # Update alpha power plot
                    power_array = np.array(alpha_power_history)
                    power_time = np.arange(len(power_array))
                    line_power.set_data(power_time, power_array)
                    ax3.set_xlim(0, max(ALPHA_POWER_BUFFER, len(power_array)))
                    ax3.set_ylim(np.min(power_array) - 5, np.max(power_array) + 5)
                    
                    # Refresh canvas
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.001)
        
        except (ValueError, IndexError):
            # Skip malformed lines
            continue
        
        except UnicodeDecodeError:
            # Skip unreadable lines
            continue

except KeyboardInterrupt:
    print("\n\n✓ Stopped by user")

finally:
    # Cleanup
    ser.close()
    plt.close()
    print("✓ Serial port closed")
    print(f"✓ Total samples processed: {sample_count}")
