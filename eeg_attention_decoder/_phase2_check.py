import pandas as pd
import numpy as np
import glob, os

# Find the most recent CSV
csvs = sorted(glob.glob("data/raw/sub-01/ses-01/raw_eeg_*.csv"))
if not csvs:
    print("ERROR: No CSV found in data/raw/sub-01/ses-01/")
    raise SystemExit(1)

path = csvs[-1]
print(f"CSV: {os.path.basename(path)}\n")

df = pd.read_csv(path, comment="#")
t   = df["wall_time_s"].to_numpy()
iti = np.diff(t) * 1000          # inter-sample intervals in ms
seq = df["seq_id"].to_numpy()
seq_diff = (seq[1:] - seq[:-1]) % 65536   # handle wrap-around

print(f"N samples    : {len(t)}")
print(f"Duration     : {t[-1]-t[0]:.2f} s")
print(f"Effective fs : {(len(t)-1)/(t[-1]-t[0]):.3f} Hz  (ideal=250.000)")
print(f"ITI mean     : {iti.mean():.3f} ms  (ideal=4.000)")
print(f"ITI std      : {iti.std():.3f} ms")
print(f"ITI CV       : {iti.std()/iti.mean():.4f}  (PASS < 0.15)")
print(f"ITI p95      : {np.percentile(iti,95):.3f} ms")
print(f"ITI p99      : {np.percentile(iti,99):.3f} ms  (PASS < 8.0 ms)")
print(f"Max gap      : {iti.max():.1f} ms")
print(f"Seq gaps     : {int((seq_diff > 1).sum())}  (PASS = 0)")
print()

# Channel value range check
for ch in ["T7", "T8"]:
    col = df[ch].to_numpy()
    print(f"{ch}  min={col.min():.0f}  max={col.max():.0f}  "
          f"mean={col.mean():.1f}  std={col.std():.1f}")

print()
# PASS / FAIL summary
cv   = iti.std() / iti.mean()
p99  = np.percentile(iti, 99)
gaps = int((seq_diff > 1).sum())
fs   = (len(t)-1) / (t[-1]-t[0])

results = {
    "Effective fs 245–255 Hz": 245 <= fs <= 255,
    "ITI CV < 0.15"          : cv < 0.15,
    "ITI p99 < 8 ms"         : p99 < 8.0,
    "Zero seq gaps"          : gaps == 0,
}
for name, ok in results.items():
    status = "PASS ✓" if ok else "FAIL ✗"
    print(f"  {status}  {name}")

overall = all(results.values())
print(f"\nPhase 2 overall: {'PASSED' if overall else 'FAILED'}")
