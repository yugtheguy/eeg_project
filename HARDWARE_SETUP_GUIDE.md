# EEG Alpha Band Monitor - Hardware Setup Guide

**For USCAPES Client Implementation**  
Version 1.0 - February 2026

---

## 📦 Bill of Materials (BOM)

### Essential Components

| Item | Specification | Quantity | Purpose | Example Part |
|------|---------------|----------|---------|--------------|
| Arduino Board | Uno/Nano/Mega | 1 | Signal acquisition | Arduino Uno R3 |
| EEG Amplifier | Gain: 1000-5000x, Input: High-Z | 1 | Signal amplification | INA128, AD620, TI ADS1299 |
| EEG Electrodes | Ag/AgCl, 10mm cup | 3-4 | Contact with scalp | Disposable cup electrodes |
| Electrode Gel | Conductive paste | 1 tube | Impedance reduction | Ten20 paste, SignaGel |
| USB Cable | Type A to B (Arduino) | 1 | Data transmission | Standard Arduino USB cable |
| Jumper Wires | Male-to-male | 5-10 | Connections | Dupont wires |
| Breadboard | Half/Full size | 1 | Prototyping (optional) | Standard breadboard |

### Optional Components

| Item | Purpose | Notes |
|------|---------|-------|
| Electrode Cap/Headband | Secure placement | Easier than manual positioning |
| Isopropyl Alcohol | Skin cleaning | 70% concentration |
| Medical Tape | Electrode securing | Hypoallergenic |
| Shielded Cable | Noise reduction | For amplifier-Arduino connection |
| Ground Strap | Additional grounding | Reduces 50/60 Hz noise |

---

## 🧠 Electrode Placement Guide

### Standard EEG 10-20 System Positions

```
                  TOP VIEW OF HEAD
                        
                      [FRONT/Nose]
                           │
                           
              Fp1 ●    ●   Fpz   ●    ● Fp2
                           
                  F3 ●       ●  Fz  ●       ● F4
                          
         LEFT                                    RIGHT
         (C3) ●             ● Cz             ● (C4)
         
                  P3 ●       ●  Pz  ●       ● P4
                  
                       O1 ●  ●  Oz  ●  ● O2
                           
                      [BACK/Occipital]


                   SIDE VIEW OF HEAD
                   
              [FRONT]                [BACK]
                 👁                     
          ┌────────────────────────────┐
          │      Fz                    │
          │              Cz            │
          │                      Pz    │
          └─────┬──────────────────────┘
              Fp1/2                 O1/2
                ↓                     ↓
           (Forehead)           (Back of head)
              
              
         Ear → A1/A2 (Reference/Ground)
```

### Recommended Electrode Positions for Alpha Monitoring

#### **Configuration 1: Temporal Focus (RECOMMENDED)**
Use this for speech/hearing attention detection:

- **LEFT Channel**: T3 (left temporal lobe, above left ear)
- **RIGHT Channel**: T4 (right temporal lobe, above right ear)
- **REFERENCE**: A1 or A2 (earlobe) or mastoid process (bone behind ear)
- **GROUND**: Forehead (Fpz) or alternative earlobe

#### **Configuration 2: Parietal/Sensorimotor**
Use this for general attention/motor tasks:

- **LEFT Channel**: C3 (left sensorimotor cortex)
- **RIGHT Channel**: C4 (right sensorimotor cortex)
- **REFERENCE**: Cz (central, top of head) or earlobe
- **GROUND**: Forehead or mastoid

#### **Configuration 3: Occipital (Visual Tasks)**
Use this for visual attention tasks:

- **LEFT Channel**: O1 (left occipital, back-left of head)
- **RIGHT Channel**: O2 (right occipital, back-right of head)
- **REFERENCE**: Oz (central occipital) or earlobe
- **GROUND**: Forehead

---

## 🔌 Wiring Diagram

### Complete System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         SUBJECT'S HEAD                          │
│                                                                 │
│   LEFT      TOP OF HEAD          RIGHT                          │
│  Electrode      REF             Electrode                       │
│   (C3)      (Earlobe)            (C4)                          │
│     │           │                  │                            │
└─────┼───────────┼──────────────────┼────────────────────────────┘
      │           │                  │
      ↓           ↓                  ↓
  ┌───────────────────────────────────────┐
  │       EEG AMPLIFIER MODULE            │
  │  (e.g., INA128, AD620, ADS1299)      │
  │                                       │
  │  LEFT+  REF/GND  RIGHT+               │
  │    │      │        │                  │
  │    │      │        │   Output: 0-5V   │
  └────┼──────┼────────┼───────────────────┘
       │      │        │
       │      └────────┼──────┐
       │               │      │
       ↓               ↓      ↓
  ┌────────────────────────────────────────┐
  │         ARDUINO BOARD                  │
  │                                        │
  │  Digital Pins:                         │
  │  [0] [1] [2] ... [13]                 │
  │                                        │
  │  Analog Pins:                          │
  │  [A0]←─── LEFT Channel (0-5V)         │
  │  [A1]←─── RIGHT Channel (Optional)    │
  │  [A2] [A3] [A4] [A5]                  │
  │                                        │
  │  Power:                                │
  │  [5V] [3.3V]                           │
  │  [GND]←──── Amplifier Ground           │
  │  [VIN]                                 │
  │                                        │
  │  [USB]──────────────────────┐          │
  └─────────────────────────────┼──────────┘
                                │
                                ↓
                        ┌───────────────┐
                        │   COMPUTER    │
                        │               │
                        │  • USB Port   │
                        │  • Python SW  │
                        └───────────────┘
```

### Detailed Pin Connections

```
EEG AMPLIFIER                    ARDUINO
─────────────                    ───────

LEFT Output    ───red─wire───→   A0 (Analog Input 0)
RIGHT Output   ───blue─wire──→   A1 (Analog Input 1)
GND/Reference  ───black─wire─→   GND (Ground)
+5V Power      ←──red─wire───    5V (if amplifier needs power)
```

### Breadboard Layout (Optional)

```
              BREADBOARD
    ┌────────────────────────┐
    │  + + + + + + + + +    │  ← Power Rail (+5V)
    │  - - - - - - - - -    │  ← Ground Rail (GND)
    │                       │
    │  [INA128]             │  ← EEG Amplifier IC
    │    ││││                │
    │ LEFT│REF│RIGHT│GND│5V  ←─ Connections
    │     │   │     │   │    │
    │     ↓   ↓     ↓   ↓    │
    └─────┼───┼─────┼───┼────┘
          │   │     │   │
          └───┼─────┼───┼───→ To Arduino
              └─────┘   └───→ Power from Arduino
```

---

## 🔧 Step-by-Step Assembly

### Step 1: Prepare the Electrodes

1. **Open electrode packaging** (use sterile electrodes)
2. **Apply conductive gel/paste** to electrode cup
   - Amount: Fill cup ~75% full
   - Gel should be fresh and moist
3. **Label electrodes** (LEFT, RIGHT, REF) with tape

### Step 2: Prepare Subject

1. **Position subject** in comfortable chair
2. **Identify electrode sites** using 10-20 system
3. **Part hair** at electrode sites (if applicable)
4. **Clean skin** with alcohol wipe
   - Rub gently to remove oils
   - Allow to dry completely (30 seconds)
5. **Optional**: Light abrasion with prep gel for lower impedance

### Step 3: Apply Electrodes

1. **Apply reference electrode** first (earlobe or mastoid)
   - Press firmly for 5-10 seconds
   - Secure with medical tape if needed
   
2. **Apply LEFT electrode** (C3 or T3)
   - Center over target location
   - Press firmly, ensure gel makes contact
   - Check impedance <10kΩ if meter available
   
3. **Apply RIGHT electrode** (C4 or T4)
   - Mirror position of left electrode
   - Press firmly, ensure gel makes contact
   
4. **Verify placement**
   - All electrodes firmly attached
   - Subject comfortable
   - No pulling or tension

### Step 4: Connect to Amplifier

```
Step-by-step connection:

1. Electrode Wires → Amplifier Inputs
   ┌──────────────┬─────────────────────┐
   │ Electrode    │ Amplifier Input     │
   ├──────────────┼─────────────────────┤
   │ LEFT (C3)    │ Channel 1 IN+ (Red) │
   │ RIGHT (C4)   │ Channel 2 IN+ (Blu) │
   │ REF (Ear)    │ GND/REF (Black)     │
   └──────────────┴─────────────────────┘

2. Amplifier Outputs → Arduino Inputs
   ┌──────────────┬─────────────────────┐
   │ Amplifier    │ Arduino Pin         │
   ├──────────────┼─────────────────────┤
   │ CH1 OUT      │ A0                  │
   │ CH2 OUT      │ A1                  │
   │ GND          │ GND                 │
   │ VCC (if req) │ 5V                  │
   └──────────────┴─────────────────────┘
```

### Step 5: Connect Arduino to Computer

1. **Insert USB cable** into Arduino USB port
2. **Connect to computer** USB port
3. **Wait for driver installation** (first time only)
4. **Verify connection**:
   - Windows: Device Manager → Ports → Arduino (COM#)
   - Mac: Terminal → `ls /dev/tty.*` → `/dev/tty.usbmodem*`
   - Linux: Terminal → `ls /dev/ttyACM*` or `/dev/ttyUSB*`

### Step 6: Power On & Test

1. **Power on EEG amplifier** (if separate power)
2. **LED should light on Arduino** (power indicator)
3. **Upload Arduino code** (arduino_eeg_acquisition.ino)
4. **Open Serial Monitor** (Tools → Serial Monitor, 115200 baud)
5. **Verify data stream**:
   ```
   1234567,512,0
   1238567,515,0
   1242567,518,0
   ```
   - Values should change continuously
   - If stuck at 0 or 1023, check connections
   - If around 512 with variation ±50, good signal!

---

## ⚡ Signal Quality Verification

### Expected Signal Characteristics

| Condition | Raw Value (ADC) | Alpha Filtered | Notes |
|-----------|----------------|----------------|-------|
| **No electrode** | ~512 (stable) | Flatline | No signal input |
| **Poor contact** | Erratic, large spikes | Noisy | Check gel/placement |
| **Good contact, eyes open** | 512 ± 30-50 | Small oscillations | Active state |
| **Good contact, eyes closed** | 512 ± 50-100 | Strong 10 Hz waves | Alpha prominent |
| **Muscle artifact** | Large spikes >200 | Saturated | Subject should relax |
| **Saturation** | 0 or 1023 | Clipping | Reduce amplifier gain |

### Quick Quality Tests

1. **Baseline Test** (subject relaxed, eyes closed)
   - Should see ~10 Hz oscillations in alpha band
   - Amplitude: moderate and stable
   
2. **Eyes Open/Closed Test**
   - CLOSE EYES: Alpha power increases (alpha up 2-3x)
   - OPEN EYES: Alpha power decreases
   - This is the most reliable EEG test!
   
3. **Attention Test**
   - Look far LEFT: Right hemisphere alpha may increase
   - Look far RIGHT: Left hemisphere alpha may increase
   - Effect is subtle, may need practice
   
4. **Artifact Test**
   - Blink eyes: Should see large spikes (normal)
   - Clench jaw: Should see high-frequency noise
   - These confirm electrode connection is working

---

## 🔍 Troubleshooting Hardware Issues

### Problem: Flat Signal (Value stuck at 512)

**Diagnosis**: No EEG signal reaching Arduino

**Solutions**:
1. ✓ Check amplifier is powered on
2. ✓ Verify output cable from amplifier to Arduino A0
3. ✓ Test amplifier output with multimeter (should vary 0-5V)
4. ✓ Check electrode connections to amplifier inputs
5. ✓ Verify electrodes are on skin with gel
6. ✓ Try touching A0 pin directly - should see changes

### Problem: Saturated Signal (Value at 0 or 1023)

**Diagnosis**: Amplifier gain too high or wrong voltage reference

**Solutions**:
1. ✓ Reduce amplifier gain (if adjustable)
2. ✓ Check amplifier is single-supply (0-5V) not ±5V
3. ✓ Verify reference electrode is connected
4. ✓ Check for DC offset in amplifier
5. ✓ Add voltage divider if output exceeds 5V

### Problem: Very Noisy Signal (large random spikes)

**Diagnosis**: Poor signal quality or electrical interference

**Solutions**:
1. ✓ **Electrode impedance**: Clean skin, add more gel
2. ✓ **Muscle artifacts**: Subject should relax face/neck/jaw
3. ✓ **50/60 Hz noise**: Use notch filter in software (already implemented)
4. ✓ **Cable shielding**: Use shielded cable from amp to Arduino
5. ✓ **Grounding**: Ensure reference electrode is secure
6. ✓ **Interference**: Move away from power lines, monitors, fluorescent lights

### Problem: Signal Too Small (barely varies)

**Diagnosis**: Insufficient amplification

**Solutions**:
1. ✓ Increase amplifier gain (target gain: 1000-5000x)
2. ✓ Check electrode contact quality
3. ✓ Verify electrodes on correct scalp positions
4. ✓ Brain signal is tiny (~50 µV), needs high gain
5. ✓ Use differential amplifier (common mode rejection)

### Problem: Arduino Not Detected by Computer

**Diagnosis**: USB connection or driver issue

**Solutions**:
1. ✓ Try different USB cable (data cable, not power-only)
2. ✓ Try different USB port on computer
3. ✓ Install Arduino drivers from arduino.cc
4. ✓ Check Device Manager (Windows) for errors
5. ✓ Try different Arduino board if possible

---

## 📊 Expected Signal Examples

### Good Quality EEG Signal

```
Time Series (Raw):
     
  600 |     *  *        *   *
      |   *      *    *       *
  512 | *          * *          *     ← Centered around 512 (2.5V)
      |                           *  *
  400 |                             *
      |
      └─────────────────────────────── Time
      
      Characteristics:
      • Centered near 512 (ADC midpoint)
      • Variation: ±50-100 units
      • Continuous, not erratic
      • No clipping (0 or 1023)
```

### Alpha Band Filtered (Eyes Closed)

```
Amplitude:
      
  100 |    **        **        **
      |   *  *      *  *      *  *
    0 | **    **  **    **  **    **   ← Strong 10 Hz rhythm
      |         **        **        **
 -100 |
      |
      └──────────────────────────────── Time
         0.1s   0.2s   0.3s   0.4s
      
      Characteristics:
      • Frequency: ~10 Hz (100ms period)
      • Regular oscillations
      • Moderate to high amplitude
      • Smooth sinusoidal waveform
```

---

## 🛡️ Safety Considerations

### Electrical Safety

❗ **IMPORTANT**: This system is for RESEARCH ONLY, not medical diagnosis

1. **Isolation**: Use USB isolation if subject safety is critical
2. **Voltage limits**: Ensure amplifier output is <5V
3. **Battery power**: Consider battery-powered amplifier for complete isolation
4. **No modifications**: Don't modify Arduino power supply
5. **Inspection**: Check all cables for damage before use

### Subject Safety

1. ✓ Use medical-grade electrodes (Ag/AgCl)
2. ✓ Don't use on damaged skin
3. ✓ Remove electrodes gently to avoid skin irritation
4. ✓ Clean electrodes between subjects (70% alcohol)
5. ✓ Allow breaks, don't exceed 30-minute sessions without rest

### Data Safety

1. ✓ Anonymize subject data
2. ✓ Secure storage of recordings
3. ✓ Obtain informed consent for research
4. ✓ Follow institutional review board (IRB) guidelines

---

## 📞 Hardware Support Checklist

Before contacting support, verify:

- [ ] Arduino powers on (LED lights up)
- [ ] USB cable is data-capable (not power-only)
- [ ] Arduino code uploaded successfully (no errors)
- [ ] Serial Monitor shows data streaming (115200 baud)
- [ ] Amplifier is powered and connected
- [ ] Electrodes have conductive gel applied
- [ ] Electrodes are on correct scalp positions
- [ ] Amplifier output is 0-5V range (check with multimeter)
- [ ] All ground connections are secure
- [ ] Serial port is not in use by other software

**If all checked and still issues → Review SOFTWARE troubleshooting in main README.md**

---

## 📚 Additional Resources

**Arduino Resources:**
- Official documentation: docs.arduino.cc
- Forums: forum.arduino.cc
- ADC reference: arduino.cc/reference/en/language/functions/analog-io/analogread/

**EEG Hardware:**
- OpenBCI community: openbci.com
- NeuroSky reference: neurosky.com/biosensors
- DIY EEG resources: instructables.com (search "EEG")

**Amplifier ICs:**
- INA128 datasheet (Texas Instruments)
- AD620 datasheet (Analog Devices)
- ADS1299 datasheet (TI - Medical-grade 8-channel)

---

**Document Version**: 1.0  
**Last Updated**: February 13, 2026  
**For**: USCAPES Client Implementation  

*Ensuring reliable EEG signal acquisition for alpha band monitoring* 🔧⚡
