# EEG Attention Decoder — Client Setup Guide

A complete step-by-step guide to set up, calibrate, train, and use the EEG Attention Decoder from scratch.

---

## Table of Contents

1. [What This System Does](#1-what-this-system-does)
2. [Hardware You Need](#2-hardware-you-need)
3. [Hardware Assembly](#3-hardware-assembly)
4. [Electrode Placement](#4-electrode-placement)
5. [Software Installation](#5-software-installation)
6. [Configuration](#6-configuration)
7. [Before Every Session — Checklist](#7-before-every-session--checklist)
8. [Step 1 — Hardware Verification](#8-step-1--hardware-verification)
9. [Step 2 — Baseline & IAF Measurement](#9-step-2--baseline--iaf-measurement)
10. [Step 3 — Generate Protocol](#10-step-3--generate-protocol)
11. [Step 4 — Attention Task Recording](#11-step-4--attention-task-recording)
12. [Step 5 — Train Your Personal Model](#12-step-5--train-your-personal-model)
13. [Step 6 — Real-Time Decoding](#13-step-6--real-time-decoding)
14. [Interpreting Results](#14-interpreting-results)
15. [Troubleshooting](#15-troubleshooting)
16. [Re-Training & Maintenance](#16-re-training--maintenance)

---

## 1. What This System Does

This system reads your brainwave signals (EEG) from two electrodes on the sides of your head and decodes **which direction you are paying auditory attention to** — left or right — in real time.

It works on the well-established principle that when you actively listen to audio on your left side, your **right hemisphere alpha power increases** (a phenomenon called alpha lateralization). The system measures this change and classifies your attention direction with a trained machine learning model.

**Important:** The system requires you to wear headphones and have actual audio playing to a specific ear during both training and real-time use. Imagined attention without real audio does not produce a strong enough signal.

---

## 2. Hardware You Need

| Item | Quantity | Notes |
|---|---|---|
| Arduino UNO R4 WiFi or Minima | 1 | The microcontroller that reads the EEG |
| UpsideDownLabs BioAmp EXG Pill | 2 | EEG amplifier modules |
| Ag/AgCl EEG electrodes (snap-type) | 6 minimum | T7, T8, and mastoid references |
| Electrode paste or Ten20 conductive gel | 1 tube | For good skin contact |
| USB-A to USB-C cable | 1 | Arduino to PC |
| Stereo headphones | 1 pair | For dichotic listening during training |
| PC running Windows 10/11 | 1 | With Python 3.11+ installed |

---

## 3. Hardware Assembly

### Wiring the BioAmp EXG Pills to Arduino

```
BioAmp Pill #1  (LEFT channel — T7)
  OUT  →  Arduino A0
  VCC  →  Arduino 3.3V
  GND  →  Arduino GND

BioAmp Pill #2  (RIGHT channel — T8)
  OUT  →  Arduino A1
  VCC  →  Arduino 3.3V
  GND  →  Arduino GND
```

### Upload firmware to Arduino

1. Open `arduino_eeg_acquisition.ino` in the Arduino IDE
2. Select board: **Arduino UNO R4 Minima** (or WiFi)
3. Select the correct COM port
4. Click **Upload**
5. Once uploaded, close the Arduino IDE (**important**: do not have the Serial Monitor open during data collection)

---

## 4. Electrode Placement

Use the **10-20 International EEG System** placement. You only need 4 electrode sites:

```
         FRONT
    _______________
   /               \
  |  T7         T8  |   ← Place here (above ears, temporal lobe)
  |                 |
   \    Left Right  /
    \               /
          |
     Left     Right
    mastoid   mastoid   ← Behind and below each ear
```

### Step-by-step placement

1. **Clean the skin** at each site with an alcohol wipe and let dry
2. **Apply a small amount of electrode gel** to each electrode
3. **T7** — place firmly on the LEFT side of the head, directly above the left ear (roughly where a headphone speaker sits)
4. **T8** — same on the RIGHT side above the right ear
5. **Left mastoid (REF)** — bony bump directly behind the left ear
6. **Right mastoid (REF)** — bony bump directly behind the right ear

> **Check contact:** Press each electrode firmly for 5 seconds after placing. Poor contact is the #1 cause of bad signal.

### Connecting to BioAmp Pills

```
BioAmp Pill #1 (LEFT):
  IN+  →  T7 electrode
  IN−  →  Left mastoid electrode
  GND  →  Body ground (wrist or ankle electrode)

BioAmp Pill #2 (RIGHT):
  IN+  →  T8 electrode
  IN−  →  Right mastoid electrode
  GND  →  Same body ground
```

---

## 5. Software Installation

### Prerequisites

- Python 3.11 or higher: https://python.org/downloads
- Git (optional): https://git-scm.com

### Install

Open a terminal (PowerShell on Windows) and run:

```powershell
# Navigate to the project folder
cd path\to\eeg_attention_decoder

# Create a virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

> If `.ps1` scripts are blocked, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Verify installation

```powershell
python -m pytest tests/ -v
```

All **46 tests should pass**. If any fail, re-run `pip install -r requirements.txt`.

---

## 6. Configuration

Open `configs/default.yaml` in any text editor. The only setting you may need to change is the **serial port**:

```yaml
hardware:
  serial_port: "COM7"    # Change this to match your Arduino's COM port
```

### Finding your COM port (Windows)

1. Plug in the Arduino via USB
2. Open **Device Manager** (`Win + X` → Device Manager)
3. Expand **Ports (COM & LPT)**
4. Look for **Arduino** or **USB Serial Device** — note the COM number (e.g. COM3, COM7)
5. Update `serial_port` in the config to match

> **North America only:** Also change `notch_freq: 50.0` to `notch_freq: 60.0` in the config (60 Hz mains frequency).

### Choose your subject ID

Every person needs a unique subject ID. Use the format `sub-XX` (e.g. `sub-01`, `sub-02`). You will use this in every command. All your personal data and model are stored under this ID.

---

## 7. Before Every Session — Checklist

Run through this before starting any data collection:

- [ ] Arduino connected via USB, firmware uploaded
- [ ] Arduino IDE and Serial Monitor **closed**
- [ ] All 4 electrodes firmly placed with gel
- [ ] Headphones ready (stereo, left/right separated)
- [ ] Virtual environment activated (`.\.venv\Scripts\Activate.ps1`)
- [ ] Working directory is `eeg_attention_decoder\` — NOT the parent folder
- [ ] Sitting comfortably in a quiet room, muscles relaxed

---

## 8. Step 1 — Hardware Verification

This checks that the hardware is working correctly before any real data collection.

**Duration:** ~2 minutes

**Command:**
```powershell
python main.py acquire --subject sub-XX --session ses-01 --duration 60 --validation-seconds 30
```

Replace `sub-XX` with your subject ID.

**What to do:** Sit still, relax, look at a fixed point. Do not blink excessively or move your jaw.

**Expected output:**
```
[INFO] hardware_validation: sampling_stability ... PASS
[INFO] hardware_validation: packet_loss       ... PASS
[INFO] hardware_validation: noise_floor       ... PASS
[INFO] hardware_validation: alpha_snr         ... PASS
[INFO] hardware_validation: All 4 checks PASSED
```

### If a check FAILS

| Failure | Likely cause | Fix |
|---|---|---|
| `sampling_stability FAIL` | Arduino not sending at 250Hz | Re-upload firmware, check baud rate is 115200 |
| `packet_loss FAIL` | USB cable issue or overloaded port | Try a different USB port or cable |
| `noise_floor FAIL` | Electrode not making contact | Re-apply gel to failing electrode, press firmly |
| `alpha_snr FAIL` | Very poor signal or wrong COM port | Check COM port in config, re-seat all electrodes |

**Do not proceed past this step until all 4 checks PASS.**

---

## 9. Step 2 — Baseline & IAF Measurement

This 2-minute recording measures your Individual Alpha Frequency (IAF) — the exact brainwave speed unique to you. This is critical for accurate classification.

**Duration:** 2 minutes (120 seconds)

**Command:**
```powershell
python main.py baseline --subject sub-XX --session ses-01
```

**What to do:** Close your eyes and sit completely still for the full 2 minutes. Keep your body relaxed and your jaw unclenched.

**Expected output:**
```
[INFO] Baseline complete. IAF estimated at X.X Hz (peak_power_db=XX.XX, used_default=false)
```

> `used_default=false` means your personal alpha peak was found. If you see `used_default=true`, the signal was too noisy — re-check electrode contact and redo this step.

---

## 10. Step 3 — Generate Protocol

This creates the randomised sequence of left/right attention blocks for your training session. It only needs to be run **once per subject per session**.

**Command:**
```powershell
python main.py acquire --subject sub-XX --session ses-01 --duration 1 --validation-seconds 0
```

Actually, the protocol is generated automatically during training preparation. Run this instead to generate and preview it:

```powershell
python -c "
from experiments.protocol import generate_session_protocol, save_protocol
from pathlib import Path
p = generate_session_protocol(30, 'sub-XX', 'ses-01', block_min_sec=10.0, block_max_sec=15.0)
save_protocol(p, Path('data/raw'))
print(f'Total: {p.total_duration_sec:.0f}s | Seed: {p.seed}')
for b in p.blocks:
    m0,s0=divmod(int(b.onset_sec),60); m1,s1=divmod(int(b.offset_sec),60)
    print(f'  Block {b.block_id:2d}  {b.trial_type:10s}  {m0}:{s0:02d} -> {m1}:{s1:02d}  ({b.duration_sec:.1f}s)')
"
```

**Print or write down the output.** You will follow this cue sheet during recording in Step 4.

---

## 11. Step 4 — Attention Task Recording

This is the most important step. You record ~7 minutes of EEG while switching attention between left and right audio.

### Before starting

1. Put on your stereo headphones
2. Open a second device (phone or tablet) ready to play audio
3. Have the printed cue sheet from Step 3 visible
4. Start a stopwatch — you will press Enter and start the stopwatch simultaneously

### Audio setup

Use any application that lets you pan audio fully left or fully right:
- **Phone:** use an equaliser or audio tool app with L/R pan control
- **YouTube:** search "audio pan test left right" — use a video that lets you test one ear at a time
- **Spotify/Music:** on desktop, use the system sound balance slider

You need to be able to switch audio quickly from full-left to full-right within 2–3 seconds between blocks.

### What to do during each block type

| Block type | What to do |
|---|---|
| **LEFT** | Pan audio fully to LEFT ear. Actively listen to it. Tilt head slightly left if it helps. |
| **RIGHT** | Pan audio fully to RIGHT ear. Actively listen to it. Tilt head slightly right. |
| **neutral** | Centre the audio. Just sit still, relaxed. |
| **catch** | Silence. Sit still. Do nothing. |
| **bilateral** | Audio playing equally in both ears. No preference. |

> **Key:** The audio must actually play. Do not try to do imagined attention without sound — it does not produce measurable signal.

### Run the recording

```powershell
python main.py acquire --subject sub-XX --session ses-01 --duration 420 --validation-seconds 415
```

Press **Enter** and immediately start your stopwatch. Follow your cue sheet precisely.

**During recording:**
- Follow the cue sheet timing as closely as possible
- Do not move your head, chew, or blink rapidly during LEFT/RIGHT blocks
- You may blink and swallow during catch/neutral blocks
- Sit motionless for the entire recording

**What you will see every few seconds:**
```
[INFO] SerialReader: 500 packets received, 0 dropped (0.00% loss)
```

When it ends:
```
[INFO] SerialReader stopped. CSV: data/raw/sub-XX/ses-01/raw_eeg_<timestamp>.csv
[INFO] hardware_validation: All 4 checks PASSED
```

---

## 12. Step 5 — Train Your Personal Model

This analyses the recording, extracts brainwave features, and trains a personalised LDA classifier. It runs automatically and takes ~3 minutes (1000 permutation tests).

**Command:**
```powershell
python main.py train --subject sub-XX --session ses-01 --data-dir data/raw
```

### Understanding the output

At the end you will see results like:

```
[INFO] CV summary: {'mean_balanced_accuracy': 0.68, 'std_balanced_accuracy': 0.12}
[INFO] Permutation test: p=0.021 | H₀ REJECTED (significant) | observed_bacc=0.68
[INFO] Model saved: models/saved/lda_sub-XX_ses-01_<timestamp>.joblib
```

### Pass / Fail criteria

| Metric | Pass | Action if fail |
|---|---|---|
| `mean_balanced_accuracy` | > 0.55 | Re-do Step 4 with better audio compliance |
| `perm_p_value` | < 0.05 | Re-do Step 4 — signal not above chance |
| `perm_reject_null` | `true` | Re-do Step 4 |

> If accuracy is 0.35–0.50 but `roc_auc` is also below 0.5, the audio direction may have been reversed (right audio during left blocks). Redo Step 4 with careful attention to which ear the audio is in.

**Note the model path printed at the end** — you will need it for Step 6.

---

## 13. Step 6 — Real-Time Decoding

Once you have a trained model, you can run the live attention decoder. This streams EEG from the Arduino, runs the full processing pipeline, and displays your attention direction in real time.

**Command:**
```powershell
python main.py decode --subject sub-XX --model models/saved/lda_sub-XX_ses-01_<timestamp>.joblib
```

Replace `<timestamp>` with the actual filename printed at the end of Step 5.

### What the dashboard shows

A window opens with:
- **Live EEG waveforms** for T7 (left) and T8 (right) scrolling in real time
- **Probability bars** for Left and Right attention (updated every 4ms)
- **Lateralization Index** — positive = currently attending left, negative = attending right

### During real-time use

1. Put on headphones with audio playing
2. Actively listen to the left or right channel
3. Watch the probability bar shift toward the direction you are attending to
4. Expect 1–3 second latency — the model uses 1-second windows

**Close the window** when done. The terminal will print latency statistics:
```
[INFO] Realtime latency: {'mean_ms': 2.4, 'p95_ms': 2.9, 'max_ms': 3.8}
```

Latency should be well below the 250ms threshold.

---

## 14. Interpreting Results

### Training metrics explained

| Metric | What it means | Good value |
|---|---|---|
| `mean_balanced_accuracy` | Average correct classification rate, Left vs Right | > 0.65 |
| `std_balanced_accuracy` | Consistency across folds — lower is better | < 0.15 |
| `roc_auc` | Area under ROC curve — 0.5 = chance, 1.0 = perfect | > 0.60 |
| `perm_p_value` | Probability the result is due to chance | < 0.05 |
| `perm_reject_null` | Whether the model is statistically significant | `true` |
| `block_accuracy` | Block-level majority-vote accuracy | > 0.60 |

### Why accuracy might be lower than expected

- **Task compliance:** Did you actually pan the audio to the correct ear each block?
- **Electrode contact:** Any loose electrode adds noise to both alpha channels equally, masking the asymmetry
- **Movement:** Jaw clenching, head movement, or swallowing during LEFT/RIGHT blocks injects muscle artefact
- **Alpha suppression:** Some people have naturally weak alpha. This is not a fault — collect more sessions

---

## 15. Troubleshooting

### "Serial port not found" or "Could not open COM7"

- Check the Arduino is plugged in
- Find your correct COM port in Device Manager
- Update `serial_port` in `configs/default.yaml`
- Make sure Arduino IDE Serial Monitor is closed

### "Consumer queue full — dropping packets"

This is a display warning only. It means the live dashboard rendering briefly slowed the queue consumer. It does **not** affect recorded data or model quality. Ignore it.

### "used_default=true" during baseline

The baseline could not find a clear alpha peak. This means:
- Electrode contact is poor (most likely)
- Subject was not sitting still with eyes closed
- Try again after re-applying electrode gel

### Hardware check fails "noise_floor"

The 100–120 Hz noise band is too high, meaning a muscle or electronics artefact. Check:
- Both mastoid IN− connections are secure
- Subject is not tense — relax shoulders and jaw
- No electrical devices (phone charger, fluorescent light) touching the chair

### Model accuracy below 0.50

First check: was the audio actually in the correct ear for each block? If unsure, redo the recording with a helper reading the cue sheet aloud while you control the audio pan.

Also verify electrode placement — T7 must be on the **left** side.

### "Protocol file not found"

You must run Step 3 (protocol generation) before Step 5 (training). They must use the same subject and session ID.

---

## 16. Re-Training & Maintenance

### When to re-train

| Situation | Action |
|---|---|
| First time using system | Full workflow Steps 1–6 |
| Accuracy drops noticeably after weeks of use | Repeat Steps 4–5 only (new session ID e.g. `ses-02`) |
| Electrodes placed in different positions | Repeat Steps 1–5 |
| Not used for more than 2 months | Repeat Steps 2–5 |
| Switching to a different user | Full workflow Steps 1–6 with new `sub-XX` ID |

### Adding more training sessions

Each additional session improves accuracy. Use a new session ID:

```powershell
# New session
python main.py baseline --subject sub-XX --session ses-02
python main.py acquire  --subject sub-XX --session ses-02 --duration 420 --validation-seconds 415
python main.py train    --subject sub-XX --session ses-02 --data-dir data/raw
```

### File structure

All your data is stored under:
```
eeg_attention_decoder/
  data/raw/
    sub-XX/
      ses-01/
        iaf.json              ← Your personal alpha frequency
        protocol.json         ← Attention task block sequence
        raw_eeg_<time>.csv    ← Raw EEG recording
  models/saved/
    lda_sub-XX_ses-01_<time>.joblib      ← Your trained model
    lda_sub-XX_ses-01_<time>_meta.json   ← Model metadata
  logs/
    train_sub-XX_ses-01_<time>.json      ← Full training report
```

---

## Quick Reference Card

```
FIRST TIME SETUP (one-off):
  pip install -r requirements.txt
  python -m pytest tests/ -v          # all 46 tests should pass

EVERY SESSION (in order):
  1. python main.py acquire  --subject sub-XX --session ses-01 --duration 60  --validation-seconds 30
  2. python main.py baseline --subject sub-XX --session ses-01
  3. [generate protocol — see Step 3]
  4. python main.py acquire  --subject sub-XX --session ses-01 --duration 420 --validation-seconds 415
  5. python main.py train    --subject sub-XX --session ses-01 --data-dir data/raw
  6. python main.py decode   --subject sub-XX --model models/saved/lda_sub-XX_ses-01_<timestamp>.joblib

TESTS:
  python main.py validate
```

---

*For technical support, provide the full log file from `logs/train_sub-XX_ses-01_<timestamp>.json` and the output of `python main.py validate`.*
