"""
main.py
========
Entry point for the EEG Attention Decoder.

Modes
-----
acquire   – Record raw EEG to CSV (hardware validation included).
baseline  – Record 2-min eyes-closed baseline for IAF estimation.
train     – Offline training pipeline with LOBO CV + permutation test.
decode    – Real-time decoding with live dashboard.
validate  – Run unit tests and display a pass/fail summary.

Usage examples
--------------
::

    # Acquire a session
    python main.py acquire --subject sub-01 --session ses-01

    # Record eyes-closed baseline
    python main.py baseline --subject sub-01 --session ses-01

    # Train offline model
    python main.py train --subject sub-01 --session ses-01 \
                         --data-dir data/raw \
                         --labels-dir data/labels

    # Real-time decode with loaded model
    python main.py decode --subject sub-01 --model models/saved/lda_sub-01_ses-01.joblib

    # Run all tests
    python main.py validate

Reproducibility
---------------
Every run logs: git commit hash, config hash, timestamp, and hardware
validation summary to ``logs/<run_id>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH  = PROJECT_ROOT / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: Path = CONFIG_PATH) -> Dict:
    """Load YAML config and inject system metadata."""
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    # Git commit hash for reproducibility
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        commit = "unknown"
    cfg["system"]["git_commit"] = commit

    # Config hash (for run logs)
    cfg_bytes = json.dumps(cfg, sort_keys=True).encode()
    cfg["system"]["config_hash"] = hashlib.sha256(cfg_bytes).hexdigest()[:12]

    return cfg


# ---------------------------------------------------------------------------
# Run logger
# ---------------------------------------------------------------------------

def log_run(
    run_id: str,
    mode: str,
    cfg: Dict,
    extra: Optional[Dict] = None,
) -> Path:
    """Write a run manifest JSON to ``logs/<run_id>.json``."""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    path = log_dir / f"{run_id}.json"

    doc = {
        "run_id": run_id,
        "mode": mode,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": cfg["system"].get("git_commit", "unknown"),
        "config_hash": cfg["system"].get("config_hash", "unknown"),
        "random_seed": cfg["system"]["random_seed"],
    }
    if extra:
        doc.update(extra)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    logger.info("Run log written: %s", path)
    return path


# ---------------------------------------------------------------------------
# Mode: acquire
# ---------------------------------------------------------------------------

def mode_acquire(args: argparse.Namespace, cfg: Dict) -> None:
    """
    Stream raw EEG from the Arduino and save to CSV.

    Runs hardware validation after ``--validation-seconds`` of data.
    """
    from acquisition.hardware_validation import run_all_validations
    from acquisition.serial_reader import SerialReader

    run_id = f"acquire_{args.subject}_{args.session}_{time.strftime('%Y%m%dT%H%M%S')}"
    hw_cfg  = cfg["hardware"]
    acq_cfg = cfg["acquisition"]

    reader = SerialReader(
        port=hw_cfg["serial_port"],
        baud_rate=hw_cfg["baud_rate"],
        n_channels=hw_cfg["n_channels"],
        output_dir=Path(acq_cfg["output_dir"]),
        subject_id=args.subject,
        session_id=args.session,
        channel_names=hw_cfg["channel_names"],
        reconnect_attempts=acq_cfg["reconnect_attempts"],
        reconnect_delay_sec=acq_cfg["reconnect_delay_sec"],
    )
    reader.start(config_dict=cfg)

    validation_sec = getattr(args, "validation_seconds", 30)
    collect_sec    = getattr(args, "duration", 300)
    packets = []

    logger.info("Acquiring for %.0f s (validation after %.0f s) …", collect_sec, validation_sec)
    t0 = time.monotonic()
    try:
        while (time.monotonic() - t0) < collect_sec:
            pkt = reader.get_packet(timeout=2.0)
            if pkt is None:
                continue
            packets.append(pkt)

            elapsed = time.monotonic() - t0
            if len(packets) == int(validation_sec * hw_cfg["sampling_rate"]):
                _run_validation(packets, cfg, args.subject, args.session)
    except KeyboardInterrupt:
        logger.info("Acquisition stopped by user.")
    finally:
        reader.stop()

    log_run(run_id, "acquire", cfg, {
        "n_packets": len(packets),
        "csv_path": str(reader.csv_path),
    })


def _run_validation(packets, cfg, subject_id, session_id):
    from acquisition.hardware_validation import run_all_validations
    hw_cfg = cfg["hardware"]

    signal_uv = np.array([p.channel_data for p in packets])
    iaf_hz = cfg["iaf"]["default_iaf_hz"]
    out_dir  = (
        PROJECT_ROOT
        / cfg["acquisition"]["output_dir"]
        / subject_id / session_id / "validation.json"
    )
    report = run_all_validations(
        packets, signal_uv, hw_cfg["sampling_rate"], iaf_hz,
        output_path=out_dir,
        subject_id=subject_id,
        session_id=session_id,
    )
    if not report["overall_passed"]:
        logger.error(
            "Hardware validation FAILED (%d/%d checks). "
            "Inspect report: %s",
            report["checks_passed"], report["checks_total"], out_dir,
        )
    return report


# ---------------------------------------------------------------------------
# Mode: baseline
# ---------------------------------------------------------------------------

def mode_baseline(args: argparse.Namespace, cfg: Dict) -> None:
    """
    Acquire eyes-closed baseline and compute IAF.
    Saves IAF to ``<data_dir>/<subject>/<session>/iaf.json``.
    """
    from acquisition.serial_reader import SerialReader
    from processing.features import estimate_iaf
    from processing.filters import build_filter_bank
    from processing.referencing import reference_from_config

    hw_cfg  = cfg["hardware"]
    acq_cfg = cfg["acquisition"]
    iaf_cfg = cfg["iaf"]

    duration_sec = iaf_cfg["baseline_duration_sec"]
    logger.info("Eyes-closed baseline: %.0f s", duration_sec)

    reader = SerialReader(
        port=hw_cfg["serial_port"],
        baud_rate=hw_cfg["baud_rate"],
        n_channels=hw_cfg["n_channels"],
        output_dir=Path(acq_cfg["output_dir"]),
        subject_id=args.subject,
        session_id=args.session,
        channel_names=hw_cfg["channel_names"],
    )
    reader.start(config_dict=cfg)

    filter_bank = build_filter_bank(cfg, n_channels=hw_cfg["n_channels"])
    samples = []
    t0 = time.monotonic()

    try:
        while (time.monotonic() - t0) < duration_sec:
            pkt = reader.get_packet(timeout=2.0)
            if pkt is None:
                continue
            filtered = filter_bank.apply(pkt.channel_data.reshape(1, -1))
            referenced = reference_from_config(cfg, filtered)
            samples.append(referenced.ravel())
    except KeyboardInterrupt:
        logger.info("Baseline collection stopped by user.")
    finally:
        reader.stop()

    if len(samples) < int(10 * hw_cfg["sampling_rate"]):
        logger.error("Insufficient baseline data (< 10 s). Cannot estimate IAF.")
        return

    data = np.array(samples)  # (n_samples, n_channels)
    result = estimate_iaf(
        data,
        fs=hw_cfg["sampling_rate"],
        search_min_hz=cfg["features"]["iaf_search_min"],
        search_max_hz=cfg["features"]["iaf_search_max"],
        default_hz=iaf_cfg["default_iaf_hz"],
    )

    out_dir = (
        Path(acq_cfg["output_dir"]) / args.subject / args.session
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    iaf_path = out_dir / "iaf.json"
    with open(iaf_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    logger.info("IAF = %.2f Hz saved to %s", result["iaf_hz"], iaf_path)


# ---------------------------------------------------------------------------
# Mode: train
# ---------------------------------------------------------------------------

def mode_train(args: argparse.Namespace, cfg: Dict) -> None:
    """
    Full offline training pipeline:
    1. Load pre-recorded CSV + protocol labels.
    2. Filter + reference + window + extract features.
    3. Run LOBO or LOSO cross-validation.
    4. Run permutation test.
    5. Save best model (final fit on all data).
    """
    import pandas as pd

    from experiments.labeling import assign_labels, filter_labeled
    from experiments.protocol import load_protocol
    from models.lda import AttentionLDA
    from processing.features import extract_features_batch, extract_windows
    from processing.filters import build_filter_bank
    from processing.referencing import reference_from_config
    from stats.cross_validation import run_cv, summarise_cv
    from stats.metrics import aggregate_cv_metrics
    from stats.permutation_tests import run_permutation_test

    run_id = f"train_{args.subject}_{args.session}_{time.strftime('%Y%m%dT%H%M%S')}"
    hw_cfg    = cfg["hardware"]
    feat_cfg  = cfg["features"]
    cv_cfg    = cfg["cross_validation"]
    seed      = cfg["system"]["random_seed"]

    # ── Load IAF ──────────────────────────────────────────────────────────
    iaf_path = Path(args.data_dir) / args.subject / args.session / "iaf.json"
    if iaf_path.exists():
        with open(iaf_path) as fh:
            iaf_hz = float(json.load(fh)["iaf_hz"])
        logger.info("Loaded IAF = %.2f Hz from %s", iaf_hz, iaf_path)
    else:
        iaf_hz = cfg["iaf"]["default_iaf_hz"]
        logger.warning("IAF file not found — using default %.1f Hz", iaf_hz)

    # ── Load raw CSV ───────────────────────────────────────────────────────
    csv_glob = list(
        (Path(args.data_dir) / args.subject / args.session).glob("raw_eeg_*.csv")
    )
    if not csv_glob:
        logger.error("No raw EEG CSV found for %s/%s", args.subject, args.session)
        return
    csv_path = sorted(csv_glob)[-1]   # take most recent
    logger.info("Loading EEG: %s", csv_path)

    raw_df = pd.read_csv(csv_path, comment="#")
    ch_cols = hw_cfg["channel_names"]
    signal_uv = raw_df[ch_cols].to_numpy(dtype=np.float64)  # (n_samples, n_ch)
    wall_times = raw_df["wall_time_s"].to_numpy(dtype=np.float64)

    # ── Filter + reference ────────────────────────────────────────────────
    filter_bank = build_filter_bank(cfg, n_channels=hw_cfg["n_channels"])
    filtered    = filter_bank.apply(signal_uv)
    referenced  = reference_from_config(cfg, filtered)

    # ── Window ────────────────────────────────────────────────────────────
    fs = float(hw_cfg["sampling_rate"])
    win_samples  = int(feat_cfg["window_size_sec"] * fs)
    step_samples = int(win_samples * feat_cfg["overlap"])

    windows, center_ts = extract_windows(referenced, wall_times, win_samples, step_samples)

    # ── Features ──────────────────────────────────────────────────────────
    # Channel index: channel_names[0] -> left, channel_names[1] -> right
    ch_names = hw_cfg.get("channel_names", ["T7", "T8"])
    left_ch_idx  = next((i for i, c in enumerate(ch_names) if c in ("T7", "C3", "F3")), 0)
    right_ch_idx = next((i for i, c in enumerate(ch_names) if c in ("T8", "C4", "F4")), 1)
    X = extract_features_batch(windows, fs, iaf_hz, cfg,
                               left_ch_idx=left_ch_idx,
                               right_ch_idx=right_ch_idx)

    # ── Labels ────────────────────────────────────────────────────────────
    protocol_path = Path(args.data_dir) / args.subject / args.session / "protocol.json"
    if not protocol_path.exists():
        logger.error("Protocol file not found: %s", protocol_path)
        return
    protocol = load_protocol(protocol_path)

    delay_sec = cfg["experiment"]["physiological_delay_ms"] / 1000.0
    session_start = float(wall_times[0])
    labels, block_ids = assign_labels(center_ts, protocol, delay_sec, session_start)

    # Filter unlabeled / catch / bilateral for binary Left-vs-Right decoding
    exclude = [
        cfg["experiment"]["label_map"]["catch"],
        cfg["experiment"]["label_map"]["bilateral"],
        cfg["experiment"]["label_map"]["neutral"],
    ]
    X_clean, y_clean, bid_clean, ts_clean = filter_labeled(
        X, labels, block_ids, center_ts, exclude_labels=exclude
    )
    logger.info("Clean dataset: %d windows, %d features", X_clean.shape[0], X_clean.shape[1])

    # ── Cross-validation ───────────────────────────────────────────────────
    strategy = cv_cfg["strategy"]

    def model_factory():
        return AttentionLDA(shrinkage="auto", random_state=seed)

    cv_results = run_cv(
        X_clean, y_clean, bid_clean, ts_clean,
        feat_cfg["window_size_sec"], model_factory, strategy=strategy,
    )
    cv_summary = summarise_cv(cv_results)
    logger.info("CV summary: %s", cv_summary)

    # Aggregated metrics with bootstrap CI
    all_metrics = aggregate_cv_metrics(
        cv_results, bid_clean,
        n_bootstrap=cv_cfg["bootstrap_n"],
        ci_level=cv_cfg["bootstrap_ci"] / 100,
        random_seed=seed,
    )

    # ── Permutation test ────────────────────────────────────────────────────
    perm_result = run_permutation_test(
        X_clean, y_clean, bid_clean, ts_clean,
        feat_cfg["window_size_sec"],
        model_factory=model_factory,
        n_permutations=cv_cfg["n_permutations"],
        alpha_level=cv_cfg["alpha_level"],
        cv_strategy=strategy,
        random_seed=seed,
        plot=True,
        plot_path=str(PROJECT_ROOT / "logs" / f"{run_id}_null_dist.png"),
    )

    # ── Save final model (fit on all clean data) ─────────────────────────
    final_model = AttentionLDA(shrinkage="auto", random_state=seed)
    final_model.fit(
        X_clean, y_clean,
        training_metadata={
            "iaf_hz": iaf_hz,
            "strategy": strategy,
            "window_size_sec": feat_cfg["window_size_sec"],
            "filter_low": cfg["processing"]["bandpass_low"],
            "filter_high": cfg["processing"]["bandpass_high"],
            "subject_id": args.subject,
            "session_id": args.session,
        },
    )
    model_dir = PROJECT_ROOT / cfg["model"]["save_dir"]
    model_path, meta_path = final_model.save(model_dir, args.subject, args.session)

    # ── Run manifest ────────────────────────────────────────────────────────
    log_run(run_id, "train", cfg, {
        "subject_id": args.subject,
        "session_id": args.session,
        "iaf_hz": iaf_hz,
        "cv_summary": cv_summary,
        "perm_p_value": perm_result["p_value"],
        "perm_reject_null": perm_result["reject_null"],
        "model_path": str(model_path),
        "n_windows": int(X_clean.shape[0]),
        "window_metrics": all_metrics["window_level"],
        "block_accuracy": all_metrics["block_level"]["block_accuracy"],
    })


# ---------------------------------------------------------------------------
# Mode: decode (realtime)
# ---------------------------------------------------------------------------

def mode_decode(args: argparse.Namespace, cfg: Dict) -> None:
    """Start real-time decoding with the live PyQtGraph dashboard."""
    import sys as _sys
    try:
        from pyqtgraph.Qt import QtWidgets
    except ImportError:
        logger.error("PyQtGraph is required for realtime mode. pip install pyqtgraph PyQt6")
        _sys.exit(1)

    from acquisition.serial_reader import SerialReader
    from realtime.dashboard import EEGDashboard
    from realtime.decoder import RealtimeDecoder

    hw_cfg  = cfg["hardware"]
    acq_cfg = cfg["acquisition"]

    # Load IAF
    iaf_path = Path(acq_cfg["output_dir"]) / args.subject / "latest" / "iaf.json"
    iaf_hz = None
    if iaf_path.exists():
        with open(iaf_path) as fh:
            iaf_hz = float(json.load(fh)["iaf_hz"])

    decoder = RealtimeDecoder.from_config(
        cfg,
        model_path=Path(args.model),
        iaf_hz=iaf_hz,
    )
    decoder.start()

    reader = SerialReader(
        port=hw_cfg["serial_port"],
        baud_rate=hw_cfg["baud_rate"],
        n_channels=hw_cfg["n_channels"],
        output_dir=Path(acq_cfg["output_dir"]) / "realtime",
        subject_id=args.subject,
        session_id="realtime",
        channel_names=hw_cfg["channel_names"],
    )
    reader.start(config_dict=cfg)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    dashboard = EEGDashboard(cfg, class_names=["Left", "Right"])
    dashboard.show()

    from pyqtgraph.Qt import QtCore

    def _update():
        pkt = reader.get_packet(block=False)
        if pkt is None:
            return
        result = decoder.push_sample(pkt.channel_data, pkt.wall_time)
        dashboard.update(pkt.channel_data, result)

    timer = QtCore.QTimer()
    timer.timeout.connect(_update)
    timer.start(4)   # ~250 Hz poll (non-blocking)

    try:
        app.exec()
    finally:
        reader.stop()
        decoder.stop()
        logger.info("Realtime latency: %s", decoder.latency_stats)


# ---------------------------------------------------------------------------
# Mode: validate (tests)
# ---------------------------------------------------------------------------

def mode_validate(_args: argparse.Namespace, _cfg: Dict) -> None:
    """Run pytest on the tests/ directory and print a summary."""
    import subprocess as sp
    result = sp.run(
        [sys.executable, "-m", "pytest", str(PROJECT_ROOT / "tests"), "-v", "--tb=short"],
        cwd=PROJECT_ROOT,
    )
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EEG Attention Decoder — Publication Grade",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH,
                        help="Path to YAML config file.")
    sub = parser.add_subparsers(dest="mode", required=True)

    # acquire
    acq = sub.add_parser("acquire", help="Record raw EEG to CSV.")
    acq.add_argument("--subject", required=True)
    acq.add_argument("--session", required=True)
    acq.add_argument("--duration", type=float, default=300.0,
                     help="Recording duration in seconds.")
    acq.add_argument("--validation-seconds", type=float, default=30.0,
                     help="Seconds of data before hardware validation.")

    # baseline
    base = sub.add_parser("baseline", help="Record eyes-closed IAF baseline.")
    base.add_argument("--subject", required=True)
    base.add_argument("--session", required=True)

    # train
    tr = sub.add_parser("train", help="Offline training + validation.")
    tr.add_argument("--subject", required=True)
    tr.add_argument("--session", required=True)
    tr.add_argument("--data-dir", default="data/raw")

    # decode
    dec = sub.add_parser("decode", help="Real-time attention decoding.")
    dec.add_argument("--subject", required=True)
    dec.add_argument("--model", required=True,
                     help="Path to .joblib model file.")

    # validate
    sub.add_parser("validate", help="Run unit tests.")

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    cfg    = load_config(args.config)

    # Seed global RNG for reproducibility
    np.random.seed(cfg["system"]["random_seed"])

    modes = {
        "acquire":  mode_acquire,
        "baseline": mode_baseline,
        "train":    mode_train,
        "decode":   mode_decode,
        "validate": mode_validate,
    }
    modes[args.mode](args, cfg)


if __name__ == "__main__":
    main()
