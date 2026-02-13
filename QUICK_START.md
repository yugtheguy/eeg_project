# 🚀 EEG Alpha Monitor - Quick Start Guide

**One-Page Setup Reference for USCAPES**

---

## ⚡ 5-Minute Setup

### 1️⃣ Hardware Connection (2 min)

```
EEG Electrodes → EEG Amplifier → Arduino → Computer
    │                 │             │         │
  C3, C4           Output         A0, A1    USB
  (scalp)          0-5V          pins      Python
```

**Wiring:**
- LEFT electrode → Amplifier → Arduino **A0**
- RIGHT electrode → Amplifier → Arduino **A1**
- REF electrode (earlobe) → Amplifier **GND**
- Amplifier GND → Arduino **GND**
- Arduino USB → Computer

### 2️⃣ Software Setup (3 min)

```bash
# Install Python packages
pip install numpy scipy pyserial matplotlib

# Upload Arduino code
# (Use arduino_eeg_acquisition.ino file)

# Update COM port in Python code
# Edit line 16 in eeg_alpha_monitor.py:
SERIAL_PORT = 'COM7'  # Change to your port

# Run the monitor
python eeg_alpha_monitor.py
```

---

## 📍 Electrode Positions

```
    LEFT               RIGHT
     (C3)    👤HEAD    (C4)
      ●                 ●
      
      
   A1/A2 (Earlobe) → Reference
```

**Quick placement:**
- LEFT: Above left ear, or 10% of head width from midline
- RIGHT: Above right ear, or 10% of head width from midline
- REF: Earlobe or behind ear (mastoid)

---

## ✅ Verification Checklist

### Hardware
- [ ] Arduino LED is on
- [ ] Electrodes have gel applied
- [ ] Electrodes firmly on scalp
- [ ] Serial Monitor shows streaming data: `1234567,512,0`

### Software
- [ ] No Python errors when running
- [ ] Three graphs appear
- [ ] Raw signal varies (not flat)
- [ ] Alpha filter shows oscillations
- [ ] Alpha power changes when eyes open/close

---

## 🧪 Quick Test: Eyes Open/Close

**Best way to verify system works:**

1. Run: `python eeg_alpha_monitor.py`
2. Subject: Close eyes, relax for 10 seconds
3. **OBSERVE**: Alpha power (graph 3) should **INCREASE** ↗
4. Subject: Open eyes
5. **OBSERVE**: Alpha power should **DECREASE** ↘

✅ **If this works → System is functioning correctly!**

---

## 📊 What You'll See

### Graph 1: Raw EEG
- Wiggly line centered around 200-400
- Should continuously vary

### Graph 2: Alpha Band (8-12 Hz)
- Regular wave pattern ~10 Hz
- Looks like smooth sine wave
- **Stronger when relaxed/eyes closed**

### Graph 3: Alpha Power
- Number in dB (usually -40 to +20)
- Higher = more relaxed, less attention
- Lower = more alert, focused attention

---

## 🔧 Common Issues → Quick Fixes

| Problem | Quick Fix |
|---------|-----------|
| **No serial connection** | Check COM port, close Arduino IDE Serial Monitor |
| **Flat signal** | Check electrode connections, add more gel |
| **Very noisy** | Subject should relax, check electrode contact |
| **Stuck at 0 or 1023** | Reduce amplifier gain |
| **No alpha waves** | Check electrode position (should be parietal/temporal) |

---

## 💡 Pro Tips

1. **Good gel = Good signal**: Don't skimp on conductive gel
2. **Relax to see alpha**: Alpha waves strongest when relaxed
3. **Wait 60 seconds**: Signal stabilizes after 30-60 seconds
4. **Close Serial Monitor**: Python can't connect if Arduino IDE has the port open
5. **Eyes closed test**: Most reliable way to see alpha increase

---

## 📁 Key Files

| File | Use Case |
|------|----------|
| `arduino_eeg_acquisition.ino` | Upload to Arduino |
| `eeg_alpha_monitor.py` | Simple real-time visualization |
| `main.py` | Full system with attention detection |
| `README.md` | Complete documentation |
| `HARDWARE_SETUP_GUIDE.md` | Detailed wiring & troubleshooting |

---

## 🎯 Expected Results

**Good Signal Quality:**
- SNR: >15 dB
- Raw amplitude: ±50-100 ADC units
- Alpha frequency: 8-12 Hz visible
- Eyes open/closed test: 2-3x power change

**Attention Detection:**
- LI (Lateralization Index): -1.0 to +1.0
- LEFT attention: LI < -0.2
- RIGHT attention: LI > +0.2
- NEUTRAL: -0.2 to +0.2

---

## 📞 Need Help?

1. **Serial issues?** → Run `python check_port.py`
2. **Signal quality?** → Run `python quick_diagnostics.py`
3. **Complete guide?** → See `README.md`
4. **Hardware help?** → See `HARDWARE_SETUP_GUIDE.md`

---

## 🧠 Remember

**Alpha Waves (8-12 Hz) = Relaxation/Reduced Attention**

- 😌 **HIGH alpha** → Relaxed, drowsy, not processing information
- 🎯 **LOW alpha** → Alert, attentive, actively processing
- 🔀 **Asymmetric alpha** → Directional attention (left/right)

---

**🎉 You're ready to monitor alpha waves! Good luck with your research!**

---

*USCAPES EEG Alpha Band Monitoring System v1.0*  
*February 2026*
