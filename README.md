# Professional Real-time EEG Signal Processing & Attention Decoder

A production-ready, modular Python system for real-time EEG signal acquisition, preprocessing, feature extraction, and attention direction detection based on hemispheric alpha power asymmetry.

## Features

- ✅ **Real-time serial acquisition** from Arduino (~250 Hz)
- ✅ **Robust signal preprocessing** (50Hz notch, 1-40 Hz bandpass, zero-phase filtering)
- ✅ **Alpha band extraction** (8-12 Hz) with Hilbert transform
- ✅ **Lateralization Index** computation for attention direction
- ✅ **Artifact detection** (muscle artifacts, saturation, high variance)
- ✅ **Decision smoothing** with confidence estimation
- ✅ **Adaptive threshold calibration** for individual differences
- ✅ **CSV logging** of all metrics and decisions
- ✅ **Automatic reconnection** on serial disconnects
- ✅ **Signal quality metrics** (SNR, power, artifacts)

## System Architecture

```
┌─────────────────┐      ┌──────────────┐      ┌──────────────┐
│   Arduino EEG   │─────▶│ acquisition  │─────▶│   filters    │
│   Hardware      │      │   (Serial)   │      │  (Notch+BP)  │
└─────────────────┘      └──────────────┘      └──────────────┘
                                                        │
                         ┌──────────────┐              │
                         │   metrics    │◀─────────────┘
                         │  (Quality)   │
                         └──────────────┘
                                │
                                ▼
                         ┌──────────────┐      ┌──────────────┐
                         │  features    │─────▶│   decision   │
                         │ (Alpha Power)│      │     (LI)     │
                         └──────────────┘      └──────────────┘
                                                        │
                                                        ▼
                                                ┌──────────────┐
                                                │   Output:    │
                                                │ LEFT/RIGHT/  │
                                                │   NEUTRAL    │
                                                └──────────────┘
```

## Module Overview

### 1. **config.py**
Configuration management using dataclasses:
- Serial communication parameters
- Signal processing parameters (sampling rate, filter specs)
- Decision thresholds (lateralization index)
- Artifact detection settings
- Logging and visualization options

### 2. **acquisition.py**
Serial interface with Arduino:
- Automatic port detection
- Non-blocking data acquisition
- Robust reconnection with exponential backoff
- Data validation and parsing
- Dual-channel buffering (left/right hemispheres)

### 3. **filters.py**
Signal preprocessing using scipy.signal:
- 50Hz notch filter (power line interference)
- 1-40 Hz bandpass filter (valid EEG range)
- Alpha band extraction (8-12 Hz)
- Beta band extraction (13-30 Hz, for artifact detection)
- Zero-phase filtering (filtfilt) to preserve waveform
- Power spectral density computation (Welch's method)

### 4. **features.py**
Feature extraction for attention detection:
- Alpha band power computation (left/right)
- Alpha envelope via Hilbert transform
- Spectral features (band powers, peak frequency)
- Time-domain features (variance, RMS, skewness)
- Cross-hemisphere features (asymmetry, correlation)

### 5. **metrics.py**
Signal quality assessment:
- SNR computation (signal-to-noise ratio in dB)
- Artifact detection:
  - Muscle artifacts (high beta power)
  - Saturation/clipping detection
  - High variance outliers
  - Power line noise
  - Low signal detection
- Quality score (0-100)

### 6. **decision.py**
Attention direction classification:
- **Lateralization Index (LI)**: `(Right_Alpha - Left_Alpha) / (Right_Alpha + Left_Alpha)`
- Direction classification: LEFT, RIGHT, or NEUTRAL
- Confidence estimation
- Decision smoothing (majority voting)
- Adaptive threshold calibration (auto-adjusts to individual differences)

### 7. **realtime_engine.py**
Main processing loop:
- Integrates all components
- Sliding window processing (2-second windows, 50% overlap)
- Real-time console output
- CSV logging with timestamps
- Performance monitoring
- Graceful error handling

## Installation

### Requirements
- Python 3.8+
- Arduino with EEG hardware (2-channel output)

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Arduino Setup

Your Arduino should output data in this format via serial:

```
timestamp,left_channel,right_channel
1234.567,512.3,498.1
1238.567,515.2,501.4
...
```

- **Timestamp**: Milliseconds since start
- **Left channel**: Raw ADC value (0-1023 for 10-bit, 0-4095 for 12-bit)
- **Right channel**: Raw ADC value

### Example Arduino Code Snippet

```cpp
unsigned long startTime = millis();

void loop() {
  int leftChannel = analogRead(A0);
  int rightChannel = analogRead(A1);
  
  unsigned long timestamp = millis() - startTime;
  
  Serial.print(timestamp);
  Serial.print(",");
  Serial.print(leftChannel);
  Serial.print(",");
  Serial.println(rightChannel);
  
  delay(4);  // ~250 Hz sampling rate
}
```

## Usage

### Basic Usage

```python
from realtime_engine import RealtimeEEGEngine

# Create and run engine
engine = RealtimeEEGEngine()
engine.run(duration=60)  # Run for 60 seconds
```

### Command Line

```bash
# Auto-detect Arduino port
python realtime_engine.py

# Specify port
python realtime_engine.py COM3
```

### Advanced Usage with Configuration

```python
from config import SystemConfig, SerialConfig, DecisionConfig
from realtime_engine import RealtimeEEGEngine

# Create custom configuration
config = SystemConfig()
config.serial.port = "COM4"
config.serial.baudrate = 115200
config.signal.sampling_rate = 250.0
config.signal.window_size = 2.0
config.decision.li_left_threshold = -0.15
config.decision.li_right_threshold = 0.15

# Run with custom config
engine = RealtimeEEGEngine(config)
engine.run()
```

### Context Manager Pattern

```python
from realtime_engine import RealtimeEEGEngine

with RealtimeEEGEngine() as engine:
    engine.run(duration=120)
```

## Configuration Parameters

### Key Parameters to Tune

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sampling_rate` | 250.0 Hz | EEG sampling rate |
| `window_size` | 2.0 s | Processing window duration |
| `window_overlap` | 0.5 (50%) | Overlap between windows |
| `li_left_threshold` | -0.15 | LI threshold for LEFT attention |
| `li_right_threshold` | 0.15 | LI threshold for RIGHT attention |
| `adaptive_threshold` | True | Enable auto-calibration |
| `decision_smoothing_window` | 5 | Number of decisions to smooth |

### Modifying Configuration

```python
from config import get_config, update_config

# Method 1: Direct modification
config = get_config()
config.signal.sampling_rate = 500.0

# Method 2: Update function
update_config(
    signal__sampling_rate=500.0,
    decision__li_left_threshold=-0.2
)
```

## Output

### Console Output

```
2026-02-13 10:30:15 - INFO - Attention: ← LEFT | LI: -0.234 | Conf: 0.85 | Quality: 87.3/100 | L-Alpha: 145.23 | R-Alpha: 198.45
2026-02-13 10:30:16 - INFO - Attention: ← LEFT | LI: -0.187 | Conf: 0.78 | Quality: 89.1/100 | L-Alpha: 152.11 | R-Alpha: 187.33
2026-02-13 10:30:17 - INFO - Attention: • NEUTRAL | LI: -0.045 | Conf: 0.62 | Quality: 85.7/100 | L-Alpha: 163.45 | R-Alpha: 171.22
```

### CSV Log Format

The system logs data to `eeg_data_log.csv` with the following columns:

- `timestamp`: Unix timestamp
- `sample_count`: Total samples processed
- `left_alpha_power`: Alpha power from left hemisphere
- `right_alpha_power`: Alpha power from right hemisphere
- `lateralization_index`: Computed LI value
- `attention_direction`: Instantaneous direction (LEFT/RIGHT/NEUTRAL)
- `confidence`: Decision confidence (0-1)
- `smoothed_direction`: Smoothed direction after majority voting
- `quality_score`: Overall signal quality (0-100)
- `left_snr_db`: Left channel SNR
- `right_snr_db`: Right channel SNR
- `left_artifact`: Left channel artifact flag
- `right_artifact`: Right channel artifact flag

## Understanding the Science

### Alpha Power and Attention

**Key Principle**: Alpha power (8-12 Hz) reflects cortical inhibition:
- High alpha = Low cortical activity = Suppressed processing
- Low alpha = High cortical activity = Active processing

### Hemispheric Lateralization

**Attention Direction Detection**:
1. **Left Attention** (looking left):
   - Right hemisphere: LOW alpha (active)
   - Left hemisphere: HIGH alpha (suppressed)
   - **LI < -0.15** (negative)

2. **Right Attention** (looking right):
   - Left hemisphere: LOW alpha (active)
   - Right hemisphere: HIGH alpha (suppressed)
   - **LI > +0.15** (positive)

3. **Neutral** (centered attention):
   - Both hemispheres balanced
   - **-0.15 ≤ LI ≤ +0.15**

### Lateralization Index Formula

```
LI = (Right_Alpha - Left_Alpha) / (Right_Alpha + Left_Alpha)
```

- Range: -1.0 to +1.0
- Normalized measure independent of absolute power

## Troubleshooting

### Connection Issues

**Problem**: "Failed to connect to Arduino"

**Solutions**:
1. Check USB cable connection
2. Verify Arduino is powered on
3. Check Device Manager (Windows) for COM port
4. Try different USB port
5. Close other programs using the serial port

### Poor Signal Quality

**Problem**: Low SNR or frequent artifacts

**Solutions**:
1. Check electrode placement and contact
2. Verify electrode impedance (<10 kΩ)
3. Apply conductive gel
4. Reduce muscle tension
5. Shield from electrical interference
6. Check Arduino ADC reference voltage

### No Direction Detection

**Problem**: Always shows NEUTRAL

**Solutions**:
1. Wait for calibration (100 samples)
2. Adjust LI thresholds in config
3. Verify alpha power is being detected
4. Check that electrode placement differentiates hemispheres
5. Increase signal amplitude at hardware level

### High Corruption Rate

**Problem**: Many corrupted packets

**Solutions**:
1. Reduce baud rate (try 9600 or 57600)
2. Check serial cable quality
3. Add error checking on Arduino side
4. Verify Arduino is sending correct format
5. Check for electromagnetic interference

## Performance

### Processing Speed
- Typical: 10-20 ms per window
- Supports real-time processing at 250 Hz with 2-second windows
- Window hop: 1 second (50% overlap)

### Memory Usage
- Minimal buffering (2x window size)
- Typical RAM: <100 MB

### Latency
- End-to-end: ~2 seconds (window duration)
- Can be reduced by using smaller windows (trade-off with frequency resolution)

## Future Enhancements

Potential improvements:
- [ ] Real-time matplotlib visualization
- [ ] Multi-channel support (>2 channels)
- [ ] Machine learning classifier (replace threshold-based detection)
- [ ] Frequency band power visualization
- [ ] Web dashboard for monitoring
- [ ] Bluetooth support
- [ ] Mobile app integration

## Scientific References

1. **Alpha Asymmetry**: Worden, M. S., et al. (2000). "Anticipatory biasing of visuospatial attention indexed by retinotopically specific α-band electroencephalography increases over occipital cortex." *Journal of Neuroscience*.

2. **EEG Preprocessing**: Delorme, A., & Makeig, S. (2004). "EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics including independent component analysis." *Journal of Neuroscience Methods*.

3. **Lateralization Index**: Thut, G., et al. (2006). "Alpha-band electroencephalographic activity over occipital cortex indexes visuospatial attention bias and predicts visual target detection." *Journal of Neuroscience*.

## License

This code is provided for educational and research purposes.

## Authors

Professional EEG Signal Processing System
Created: 2026

---

## Quick Start Checklist

- [ ] Install Python 3.8+
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Connect Arduino with EEG hardware
- [ ] Verify Arduino serial output format
- [ ] Run: `python realtime_engine.py`
- [ ] Wait for calibration (100 samples)
- [ ] Observe attention direction output
- [ ] Check CSV log for detailed data

**For support or questions, review the troubleshooting section above.**
