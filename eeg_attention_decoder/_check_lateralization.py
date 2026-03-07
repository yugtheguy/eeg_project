"""Quick diagnostic: does alpha LI track LEFT vs RIGHT protocol blocks?"""
import glob, json
import numpy as np
import pandas as pd
from scipy.signal import welch

csvs = sorted(glob.glob('data/raw/sub-01/ses-01/raw_eeg_*.csv'))
df = pd.read_csv(csvs[-1], comment='#')
fs = 250
wall_times = df['wall_time_s'].values
ch0 = df['T7'].values
ch1 = df['T8'].values
session_start = wall_times[0]

with open('data/raw/sub-01/ses-01/protocol.json') as f:
    proto = json.load(f)

results = []
print(f"{'Block':>5}  {'Type':6}  {'LI':>8}  {'Dominant'}")
print('-' * 40)
for blk in proto['blocks']:
    if blk['trial_type'] not in ('left', 'right'):
        continue
    t0 = session_start + blk['onset_sec'] + 0.4
    t1 = session_start + blk['offset_sec'] - 0.5
    mask = (wall_times >= t0) & (wall_times < t1)
    if mask.sum() < 50:
        print(f"  Block {blk['block_id']:2d}: not enough samples, skipping")
        continue
    seg0 = ch0[mask]
    seg1 = ch1[mask]
    nperseg = min(256, len(seg0) // 2)
    freqs, psd0 = welch(seg0, fs=fs, nperseg=nperseg)
    freqs, psd1 = welch(seg1, fs=fs, nperseg=nperseg)
    alpha_mask = (freqs >= 6.5) & (freqs <= 10.5)
    a0 = psd0[alpha_mask].mean()
    a1 = psd1[alpha_mask].mean()
    li = float((a1 - a0) / (a1 + a0 + 1e-30))
    dom = 'T8 side' if li > 0 else 'T7 side'
    results.append((blk['block_id'], blk['trial_type'], li))
    print(f"  {blk['block_id']:3d}  {blk['trial_type']:6s}  {li:+.4f}  ({dom})")

if results:
    left_li  = np.mean([r[2] for r in results if r[1] == 'left'])
    right_li = np.mean([r[2] for r in results if r[1] == 'right'])
    effect = left_li - right_li
    print()
    print(f'Mean LI during LEFT  blocks: {left_li:+.4f}')
    print(f'Mean LI during RIGHT blocks: {right_li:+.4f}')
    print(f'Effect (LEFT_LI - RIGHT_LI): {effect:+.4f}')
    print()
    if effect > 0.005:
        print('CORRECT direction: during LEFT attention, T7 suppresses (LI positive = T8 dominant)')
        print('  => alpha is lateralizing correctly, task compliance OK')
    elif effect < -0.005:
        print('INVERTED: during LEFT attention, T8 suppresses (LI negative = T7 dominant)')
        print('  => either channel labels swapped, or user attended in wrong direction')
    else:
        print('NO EFFECT: LI does not change between LEFT and RIGHT blocks')
        print('  => task compliance issue (not actually doing spatial attention)')
