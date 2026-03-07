"""
realtime/decoder.py
====================
Streaming EEG attention decoder.

Architecture
------------
* Shares ALL processing code with the offline pipeline — no duplicate
  logic.  Uses ``processing.filters.FilterBank`` and
  ``processing.features.extract_features_window`` directly.
* Maintains a ``RingBuffer`` (from ``acquisition.serial_reader``) as a
  1-second sliding window.
* Emits posterior class probabilities after exponential smoothing.
* Logs per-window latency; raises a warning if >250 ms.
* The fitted model is loaded from disk — no retraining occurs at runtime.

Usage
-----
::

    decoder = RealtimeDecoder.from_config(cfg, model_path, meta_path)
    decoder.start()
    # Push samples from SerialReader:
    while running:
        packet = reader.get_packet()
        result = decoder.push_sample(packet.channel_data, packet.wall_time)
        if result is not None:
            print(result.probabilities)
    decoder.stop()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from acquisition.serial_reader import RingBuffer
from models.lda import AttentionLDA
from processing.features import extract_features_window
from processing.filters import FilterBank, build_filter_bank
from processing.referencing import apply_reference

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DecodeResult:
    """Output of one decoding cycle."""
    probabilities: np.ndarray    # shape (n_classes,), after smoothing
    predicted_class: int
    lateralization_index: float
    alpha_power_left_db: float
    alpha_power_right_db: float
    window_center_time: float    # wall-clock seconds
    latency_ms: float            # decode latency for this window


# ---------------------------------------------------------------------------
# Realtime decoder
# ---------------------------------------------------------------------------

class RealtimeDecoder:
    """
    Sliding-window EEG attention decoder for streaming data.

    Parameters
    ----------
    filter_bank : FilterBank
        Pre-built causal filter bank (identical to offline pipeline).
    model : AttentionLDA
        Fitted classification model loaded from disk.
    fs : float
        Sampling rate (Hz).
    iaf_hz : float
        Individual alpha frequency from the offline baseline.
    cfg : dict
        Full YAML config.
    smoothing_alpha : float
        Exponential smoothing weight for new probability estimate.
        Higher = more responsive; lower = more stable.
    max_latency_ms : float
        Warning threshold for decoding latency.
    """

    def __init__(
        self,
        filter_bank: FilterBank,
        model: AttentionLDA,
        fs: float,
        iaf_hz: float,
        cfg: Dict,
        smoothing_alpha: float = 0.2,
        max_latency_ms: float = 250.0,
    ) -> None:
        self._filter_bank = filter_bank
        self._model = model
        self._fs = fs
        self._iaf_hz = iaf_hz
        self._cfg = cfg
        self._smoothing_alpha = smoothing_alpha
        self._max_latency_ms = max_latency_ms

        feat_cfg = cfg["features"]
        proc_cfg = cfg["processing"]
        hw_cfg   = cfg["hardware"]
        rt_cfg   = cfg["realtime"]

        self._window_samples = int(feat_cfg["window_size_sec"] * fs)
        self._step_samples   = int(
            feat_cfg["window_size_sec"] * feat_cfg["overlap"] * fs
        )
        self._n_channels     = hw_cfg["n_channels"]
        self._ref_type       = proc_cfg["reference_type"]

        # Feature parameters
        self._iaf_bw          = feat_cfg["iaf_bandwidth"]
        self._beta_low        = feat_cfg["beta_low"]
        self._beta_high       = feat_cfg["beta_high"]
        self._broadband_low   = feat_cfg["broadband_low"]
        self._broadband_high  = feat_cfg["broadband_high"]
        self._enable_coh      = feat_cfg.get("enable_coherence", False)

        # Ring buffer holds exactly 1 sliding window
        buf_samples = int(rt_cfg["buffer_size_sec"] * fs) + self._window_samples
        self._buffer = RingBuffer(capacity_samples=buf_samples, n_channels=self._n_channels)
        self._samples_since_last_decode = 0

        # Smoothed probability state
        n_classes = len(self._model._pipeline.classes_)  # type: ignore[union-attr]
        self._smoothed_proba = np.full(n_classes, 1.0 / n_classes, dtype=np.float64)

        self._latency_log: List[float] = []
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialise the filter state and mark decoder as running."""
        self._filter_bank.reset()
        self._running = True
        logger.info("RealtimeDecoder started (fs=%.0f Hz, window=%d samples).",
                    self._fs, self._window_samples)

    def stop(self) -> None:
        """Stop the decoder and log average latency."""
        self._running = False
        if self._latency_log:
            avg_lat = float(np.mean(self._latency_log))
            p95_lat = float(np.percentile(self._latency_log, 95))
            logger.info(
                "RealtimeDecoder stopped. Latency: mean=%.1f ms, p95=%.1f ms",
                avg_lat, p95_lat,
            )

    # ------------------------------------------------------------------
    # Main entry point (called per-sample)
    # ------------------------------------------------------------------

    def push_sample(
        self,
        sample: np.ndarray,
        wall_time: float,
    ) -> Optional[DecodeResult]:
        """
        Push one EEG sample into the decoder pipeline.

        Parameters
        ----------
        sample : np.ndarray, shape (n_channels,)
            Raw ADC sample (µV) from the serial reader.
        wall_time : float
            ``time.monotonic()`` timestamp at receipt.

        Returns
        -------
        DecodeResult if a full window has elapsed, else None.
        """
        if not self._running:
            raise RuntimeError("Call decoder.start() before pushing samples.")

        # ── 1. Causal filter (stateful) ───────────────────────────────
        filtered = self._filter_bank.apply_single(sample)

        # ── 2. Referencing ────────────────────────────────────────────
        referenced = apply_reference(
            filtered.reshape(1, -1), self._ref_type, self._n_channels
        ).ravel()

        # ── 3. Buffer ─────────────────────────────────────────────────
        self._buffer.push(referenced, wall_time)
        self._samples_since_last_decode += 1

        # ── 4. Decode when step has elapsed ───────────────────────────
        if (self._samples_since_last_decode >= self._step_samples
                and self._buffer.n_samples >= self._window_samples):

            self._samples_since_last_decode = 0
            return self._decode_window(wall_time)

        return None

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _decode_window(self, wall_time: float) -> Optional[DecodeResult]:
        t_start = time.monotonic()

        try:
            data, timestamps = self._buffer.get_window(self._window_samples)
        except BufferError:
            return None

        # Transpose to (n_channels, n_samples) as expected by feature fn
        window = data.T    # (n_channels, window_samples)

        # ── Feature extraction (identical to offline) ─────────────────
        features = extract_features_window(
            window=window,
            fs=self._fs,
            iaf_hz=self._iaf_hz,
            iaf_bw=self._iaf_bw,
            beta_low=self._beta_low,
            beta_high=self._beta_high,
            broadband_low=self._broadband_low,
            broadband_high=self._broadband_high,
            left_ch_idx=0,
            right_ch_idx=1,
            enable_coherence=self._enable_coh,
        )

        # ── Classification ────────────────────────────────────────────
        proba = self._model.predict_proba(features.reshape(1, -1)).ravel()

        # ── Exponential smoothing ─────────────────────────────────────
        self._smoothed_proba = (
            self._smoothing_alpha * proba
            + (1.0 - self._smoothing_alpha) * self._smoothed_proba
        )
        predicted_class = int(np.argmax(self._smoothed_proba))

        # ── Extract diagnostic features ───────────────────────────────
        # feature vector layout: [log_al, log_ar, log_bl, log_br,
        #                          rel_al, rel_ar, LI, (coh?)]
        log_alpha_left  = float(features[0])
        log_alpha_right = float(features[1])
        li              = float(features[6])
        window_center   = float(np.mean(timestamps))

        # ── Latency logging ───────────────────────────────────────────
        latency_ms = (time.monotonic() - t_start) * 1000.0
        self._latency_log.append(latency_ms)
        if latency_ms > self._max_latency_ms:
            logger.warning("Decode latency %.1f ms exceeds limit %.0f ms",
                           latency_ms, self._max_latency_ms)

        return DecodeResult(
            probabilities=self._smoothed_proba.copy(),
            predicted_class=predicted_class,
            lateralization_index=li,
            alpha_power_left_db=log_alpha_left,
            alpha_power_right_db=log_alpha_right,
            window_center_time=window_center,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: Dict,
        model_path: Path,
        meta_path: Optional[Path] = None,
        iaf_hz: Optional[float] = None,
    ) -> "RealtimeDecoder":
        """
        Construct a decoder from YAML config and a saved model.

        Parameters
        ----------
        cfg : dict   — full YAML config
        model_path : Path  — ``.joblib`` file from ``AttentionLDA.save``
        meta_path : Path, optional — companion JSON metadata
        iaf_hz : float, optional
            Override IAF; if None, reads from model metadata or uses
            config default.
        """
        model = AttentionLDA.load(model_path, meta_path)

        from_meta = model._training_metadata.get("iaf_hz")
        if iaf_hz is None:
            iaf_hz = float(from_meta) if from_meta else cfg["iaf"]["default_iaf_hz"]

        filter_bank = build_filter_bank(cfg, n_channels=cfg["hardware"]["n_channels"])

        rt_cfg = cfg["realtime"]
        return cls(
            filter_bank=filter_bank,
            model=model,
            fs=cfg["hardware"]["sampling_rate"],
            iaf_hz=iaf_hz,
            cfg=cfg,
            smoothing_alpha=rt_cfg["smoothing_alpha"],
            max_latency_ms=rt_cfg["max_latency_ms"],
        )

    @property
    def smoothed_probabilities(self) -> np.ndarray:
        """Current smoothed posterior probability vector."""
        return self._smoothed_proba.copy()

    @property
    def latency_stats(self) -> Dict:
        """Summary statistics for decode latency (ms)."""
        if not self._latency_log:
            return {}
        arr = np.array(self._latency_log)
        return {
            "mean_ms": round(float(arr.mean()), 2),
            "std_ms":  round(float(arr.std()), 2),
            "p95_ms":  round(float(np.percentile(arr, 95)), 2),
            "max_ms":  round(float(arr.max()), 2),
            "n_windows": len(arr),
        }
