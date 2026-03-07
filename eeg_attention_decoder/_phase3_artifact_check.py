"""
Phase 3 — Artifact validation.
Checks that the raw signal contains clear amplitude transients
at the times when blink / jaw / movement artifacts are expected.
"""
import glob
import numpy as np
import pandas as pd

# ── Load most recent CSV ─────────────────────────────────────────────────────
csvs = sorted(glob.glob("data/raw/sub-01/ses-01/raw_eeg_*.csv"))
path = csvs[-1]
print(f"CSV: {path}\n")

df = pd.read_csv(path, comment="#")
t  = df["wall_time_s"].to_numpy()
t  = t - t[0]                         # relative seconds from session start
T7 = df["T7"].to_numpy(dtype=float)
T8 = df["T8"].to_numpy(dtype=float)

fs_est = (len(t) - 1) / (t[-1] - t[0])
print(f"Recording duration : {t[-1]:.1f} s")
print(f"Estimated fs       : {fs_est:.1f} Hz\n")

# ── Broad statistics ─────────────────────────────────────────────────────────
for name, ch in [("T7", T7), ("T8", T8)]:
    print(f"{name}  mean={ch.mean():.1f}  std={ch.std():.1f}  "
          f"min={ch.min():.0f}  max={ch.max():.0f}  "
          f"range={ch.max()-ch.min():.0f}")
print()

# ── RMS in 1-second windows ──────────────────────────────────────────────────
win_sec  = 1.0
win_samp = int(win_sec * fs_est)
n_wins   = int(t[-1] / win_sec)

print("RMS per 1-s window (T7 | T8):")
print(f"  {'Window':>8}  {'T7 RMS':>8}  {'T8 RMS':>8}  {'Peak?':>8}")
rms_t7, rms_t8, peaks = [], [], []
for i in range(n_wins):
    sl = slice(i * win_samp, (i + 1) * win_samp)
    r7 = float(np.std(T7[sl]))
    r8 = float(np.std(T8[sl]))
    rms_t7.append(r7)
    rms_t8.append(r8)

# Per-channel baseline: median of the quietest half of the first 5 windows
# (skip window 0 which may include settling transients)
b7 = float(np.median(rms_t7[1:5])) if n_wins >= 5 else float(np.median(rms_t7))
b8 = float(np.median(rms_t8[1:5])) if n_wins >= 5 else float(np.median(rms_t8))
# Temporal sites (T7/T8) produce small blink artifacts; use 3× baseline,
# with a conservative floor of 15 ADC (≈2× electronics noise, not 60).
thr7 = max(b7 * 3.0, 15.0)
thr8 = max(b8 * 3.0, 15.0)
artifact_threshold = (thr7 + thr8) / 2  # for reporting only

for i, (r7, r8) in enumerate(zip(rms_t7, rms_t8)):
    is_peak = (r7 > thr7) or (r8 > thr8)
    marker = " ← ARTIFACT" if is_peak else ""
    print(f"  {i:>4d}-{i+1:>2d}s  {r7:>8.1f}  {r8:>8.1f}{marker}")
    peaks.append(is_peak)

# ── Summary ──────────────────────────────────────────────────────────────────
n_artifact_windows = sum(peaks)

print(f"\nPer-channel baselines   T7={b7:.1f}  T8={b8:.1f} ADC RMS")
print(f"Artifact thresholds     T7>{thr7:.1f}  T8>{thr8:.1f} ADC RMS")
print(f"Windows with artifact   : {n_artifact_windows}/{n_wins}")
print()

# ADC floor samples (electrode contact check)
t8_floor  = int((T8 <= 2).sum())
t7_floor  = int((T7 <= 2).sum())
floor_pct = max(t7_floor, t8_floor) / len(T7) * 100

# ── PASS/FAIL ────────────────────────────────────────────────────────────────
checks = {
    "Both channels active (std > 10 ADC)": T7.std() > 10 and T8.std() > 10,
    "T8 ADC-floor samples < 0.5%": floor_pct < 0.5,
    "Artifact windows detected (>= 2)": n_artifact_windows >= 2,
    "Signal varies between windows (not flat)": np.std(rms_t7) > 5,
}

for name, ok in checks.items():
    print(f"  {'PASS ✓' if ok else 'FAIL ✗'}  {name}")

print(f"\nPhase 3 overall: {'PASSED' if all(checks.values()) else 'NEEDS REVIEW'}")
print()
if floor_pct >= 0.5:
    print(f"⚠ T8 ADC-floor: {max(t7_floor,t8_floor)} samples hit 0 ({floor_pct:.1f}%).")
    print("  → Reseat the T8 signal electrode (right temporal) and right mastoid reference.")
    print("  → Press firmly; add saline or electrode gel if available.")
if not checks["Artifact windows detected (>= 2)"]:
    print("⚠ No artifact windows found.")
    print("  → Re-run acquisition and perform blink/jaw/movement actions more forcefully.")
if not checks["Both channels active (std > 10 ADC)"]:
    print("⚠ One or both channels appear flat — check electrode contact.")
