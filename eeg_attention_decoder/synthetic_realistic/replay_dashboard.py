"""
synthetic_realistic/replay_dashboard.py
========================================
Replay the synthetic EEG CSV through the full decoder + live dashboard —
no Arduino required.  Feeds samples at exactly 250 Hz (4 ms/sample) using
a Qt timer, using the identical RealtimeDecoder and EEGDashboard used
during real acquisition.

Usage
-----
From the eeg_attention_decoder/ directory:

    python synthetic_realistic/replay_dashboard.py --model models/saved/lda_sub-realistic_ses-01_<timestamp>.joblib

If --model is omitted, the most recently saved model for sub-realistic is used.

What to watch
-------------
  * The EEG waveform traces should look noisy but structured (like real EEG)
  * Left/Right probability bars should fluctuate — not lock to one side
  * During LEFT blocks in the protocol, the Left bar should trend slightly higher
  * During RIGHT blocks, the Right bar should trend slightly higher
  * Overall accuracy ≈ 50–60%, matching what the offline CV showed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
HERE     = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

SUBJECT_ID = "sub-realistic"
SESSION_ID = "ses-01"
FS         = 250   # Hz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_latest_model() -> Path:
    """Find the most recently saved model for sub-realistic."""
    saved = PKG_ROOT / "models" / "saved"
    candidates = sorted(saved.glob(f"lda_{SUBJECT_ID}_{SESSION_ID}_*.joblib"))
    if not candidates:
        raise FileNotFoundError(
            f"No model found for {SUBJECT_ID}/{SESSION_ID} in {saved}.\n"
            f"Run:  python main.py train --subject {SUBJECT_ID} --session {SESSION_ID} --data-dir data/raw"
        )
    return candidates[-1]


def _load_csv() -> pd.DataFrame:
    csv_dir = PKG_ROOT / "data" / "raw" / SUBJECT_ID / SESSION_ID
    candidates = sorted(csv_dir.glob("raw_eeg_*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV found in {csv_dir}.\n"
            f"Run:  python synthetic_realistic/generate.py"
        )
    path = candidates[-1]
    print(f"[replay] Loading CSV: {path.name}")
    return pd.read_csv(path, comment="#")


def _load_protocol() -> list:
    proto_path = PKG_ROOT / "data" / "raw" / SUBJECT_ID / SESSION_ID / "protocol.json"
    with open(proto_path) as f:
        doc = json.load(f)
    return doc["blocks"]


# ---------------------------------------------------------------------------
# Main replay
# ---------------------------------------------------------------------------

def run_replay(model_path: Path) -> None:
    from pyqtgraph.Qt import QtWidgets, QtCore

    import yaml
    from models.lda import AttentionLDA
    from processing.features import extract_features_window
    from processing.filters import build_filter_bank
    from realtime.dashboard import EEGDashboard
    from realtime.decoder import RealtimeDecoder

    # Load config
    cfg_path = PKG_ROOT / "configs" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Override channel_names to correct orientation (generator always uses T7, T8)
    cfg["hardware"]["channel_names"] = ["T7", "T8"]

    print(f"[replay] Loading model: {model_path.name}")
    decoder = RealtimeDecoder.from_config(cfg, model_path=model_path, iaf_hz=8.5)
    decoder.start()

    # Load data
    df = _load_csv()
    wall_times = df["wall_time_s"].values
    ch_t7 = df["T7"].values
    ch_t8 = df["T8"].values
    n_samples = len(wall_times)

    blocks = _load_protocol()
    # Build a per-sample label array for overlay (optional terminal printout)
    label_arr = ["—"] * n_samples
    for blk in blocks:
        i_s = max(0, int(blk["onset_sec"]  * FS))
        i_e = min(n_samples, int(blk["offset_sec"] * FS))
        if blk["trial_type"] in ("left", "right"):
            for k in range(i_s, i_e):
                label_arr[k] = blk["trial_type"].upper()

    print(f"[replay] {n_samples} samples ({n_samples/FS:.1f} s) — replaying at {FS} Hz")
    print("[replay] Close the dashboard window to stop.\n")

    # ── Build Qt app and dashboard ────────────────────────────────────────
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    dashboard = EEGDashboard(cfg, class_names=["Left", "Right"])
    dashboard._win.setWindowTitle("Synthetic EEG Replay — sub-realistic")
    dashboard.show()

    # State
    state = {"idx": 0, "correct": 0, "total": 0, "last_print_idx": 0}
    PRINT_EVERY = FS * 5   # print stats every 5 seconds

    def _tick():
        i = state["idx"]
        if i >= n_samples:
            timer.stop()
            decoder.stop()
            stats = decoder.latency_stats
            print(f"\n[replay] Finished. Latency: {stats}")
            if state["total"] > 0:
                print(f"[replay] Real-time accuracy (majority vote windows): "
                      f"{state['correct']}/{state['total']} = "
                      f"{state['correct']/state['total']*100:.1f}%")
            return

        sample = np.array([ch_t7[i], ch_t8[i]], dtype=np.float64)
        wt     = float(wall_times[i])

        result = decoder.push_sample(sample, wt)

        if result is not None:
            dashboard.update(sample, result)
            # Check if the prediction is correct vs protocol label
            true_label = label_arr[i]
            if true_label in ("LEFT", "RIGHT"):
                pred = "LEFT" if result.predicted_class == 0 else "RIGHT"
                if pred == true_label:
                    state["correct"] += 1
                state["total"] += 1

        # Progress printout every 5 s
        if i - state["last_print_idx"] >= PRINT_EVERY:
            elapsed = i / FS
            m, s = divmod(int(elapsed), 60)
            block_label = label_arr[i] if label_arr[i] != "—" else "neutral/catch"
            print(f"  {m}:{s:02d}  Current block: {block_label:10s}  |  "
                  f"Samples replayed: {i}/{n_samples}")
            state["last_print_idx"] = i

        state["idx"] += 1

    # Timer: fires every 4 ms = 250 Hz real-time replay
    timer = QtCore.QTimer()
    timer.timeout.connect(_tick)
    timer.start(4)

    try:
        app.exec()
    finally:
        timer.stop()
        decoder.stop()
        print("\n[replay] Dashboard closed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Replay synthetic EEG through decoder + dashboard (no Arduino needed)."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to .joblib model file. If omitted, uses the most recently trained sub-realistic model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model_path = args.model or _find_latest_model()
    run_replay(model_path)
