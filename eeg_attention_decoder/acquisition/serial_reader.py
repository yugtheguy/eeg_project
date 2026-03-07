"""
acquisition/serial_reader.py
============================
Thread-safe serial reader for the Arduino R4 EEG acquisition board.

Packet format (Arduino must emit):
    timestamp_ms,seq_id,left,right\\n

Guarantees
----------
* Dropped-packet detection via monotonic seq_id.
* Jitter tracking (inter-packet interval deviation).
* Effective sampling-rate logging.
* Raw CSV written with YAML-sourced metadata header.
* Multi-subject / multi-session directory layout:
      <output_dir>/<subject_id>/<session_id>/raw_eeg_<timestamp>.csv
"""

from __future__ import annotations

import csv
import json
import logging
import queue
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import serial

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class EEGPacket:
    """Single decoded packet from the Arduino."""
    timestamp_ms: int    # Arduino millis() counter
    seq_id: int          # Monotonic sequence counter (rolls over at 2^16)
    channel_data: np.ndarray  # shape (n_channels,), dtype float64, µV

    wall_time: float = field(default_factory=time.monotonic)
    """Host-side monotonic timestamp at reception (seconds)."""


@dataclass
class AcquisitionStats:
    """Runtime acquisition diagnostics, updated every second."""
    effective_fs: float = 0.0          # Hz
    jitter_ms_mean: float = 0.0        # Mean inter-packet interval deviation
    jitter_ms_std: float = 0.0
    dropped_packets_total: int = 0
    total_packets: int = 0
    loss_fraction: float = 0.0


# ---------------------------------------------------------------------------
# Serial reader
# ---------------------------------------------------------------------------

class SerialReader:
    """
    Threaded serial reader.  Produces ``EEGPacket`` objects into a
    ``queue.Queue`` that downstream consumers subscribe to.

    Parameters
    ----------
    port : str
        Serial port identifier (e.g. "COM3", "/dev/ttyUSB0").
    baud_rate : int
        Must match the Arduino sketch.
    n_channels : int
        Expected number of ADC channels in each packet.
    output_dir : Path
        Root directory for raw CSV files.
    subject_id : str
        Participant identifier (e.g. "sub-01").
    session_id : str
        Session identifier (e.g. "ses-01").
    channel_names : list[str]
        Human-readable channel labels for CSV header.
    reconnect_attempts : int
        How many times to retry a lost connection before raising.
    reconnect_delay_sec : float
        Seconds to wait between reconnect attempts.
    max_queue_size : int
        Maximum number of packets buffered in the consumer queue.
    """

    SEQ_MAX: int = 65536  # seq_id rolls over at 2^16 on Arduino

    def __init__(
        self,
        port: str,
        baud_rate: int,
        n_channels: int,
        output_dir: Path,
        subject_id: str,
        session_id: str,
        channel_names: List[str],
        reconnect_attempts: int = 3,
        reconnect_delay_sec: float = 2.0,
        max_queue_size: int = 4096,
    ) -> None:
        if len(channel_names) != n_channels:
            raise ValueError(
                f"channel_names length {len(channel_names)} != n_channels {n_channels}"
            )

        self.port = port
        self.baud_rate = baud_rate
        self.n_channels = n_channels
        self.output_dir = Path(output_dir)
        self.subject_id = subject_id
        self.session_id = session_id
        self.channel_names = channel_names
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_sec = reconnect_delay_sec

        self._queue: queue.Queue[EEGPacket] = queue.Queue(maxsize=max_queue_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._serial: Optional[serial.Serial] = None

        # Diagnostics
        self._stats = AcquisitionStats()
        self._stats_lock = threading.Lock()
        self._recent_intervals: List[float] = []   # milliseconds between packets
        self._last_seq: Optional[int] = None
        self._dropped_count: int = 0
        self._total_count: int = 0

        # CSV writer
        self._csv_path: Optional[Path] = None
        self._csv_file = None
        self._csv_writer = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, config_dict: Optional[Dict] = None) -> None:
        """
        Open the serial port, initialise the CSV file, and begin reading.

        Parameters
        ----------
        config_dict : dict, optional
            Full YAML config to embed in the CSV metadata header.
        """
        self._open_serial()
        self._init_csv(config_dict or {})
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._read_loop,
            name="SerialReader",
            daemon=True,
        )
        self._thread.start()
        logger.info("SerialReader started on %s", self.port)

    def stop(self) -> None:
        """Signal the reader thread to stop and flush the CSV."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
            self._csv_file = None
        if self._serial and self._serial.is_open:
            self._serial.close()
        logger.info("SerialReader stopped. CSV: %s", self._csv_path)

    def get_packet(self, block: bool = True, timeout: float = 1.0) -> Optional[EEGPacket]:
        """
        Pop the next packet from the queue.

        Returns ``None`` if timeout expires or reader is stopped.
        """
        try:
            return self._queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def get_stats(self) -> AcquisitionStats:
        """Return a snapshot of current acquisition diagnostics."""
        with self._stats_lock:
            import copy
            return copy.copy(self._stats)

    @property
    def csv_path(self) -> Optional[Path]:
        """Path of the raw CSV file being written (None if not started)."""
        return self._csv_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open_serial(self) -> None:
        for attempt in range(1, self.reconnect_attempts + 1):
            try:
                self._serial = serial.Serial(
                    port=self.port,
                    baudrate=self.baud_rate,
                    timeout=2.0,
                )
                self._serial.reset_input_buffer()
                logger.info("Serial port %s opened (attempt %d)", self.port, attempt)
                return
            except serial.SerialException as exc:
                logger.warning(
                    "Cannot open %s (attempt %d/%d): %s",
                    self.port, attempt, self.reconnect_attempts, exc,
                )
                if attempt < self.reconnect_attempts:
                    time.sleep(self.reconnect_delay_sec)
        raise RuntimeError(
            f"Failed to open serial port {self.port} after "
            f"{self.reconnect_attempts} attempts."
        )

    def _init_csv(self, config_dict: Dict) -> None:
        """Create the output CSV with a metadata header block."""
        session_dir = self.output_dir / self.subject_id / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = time.strftime("%Y%m%dT%H%M%S")
        self._csv_path = session_dir / f"raw_eeg_{timestamp_str}.csv"

        self._csv_file = self._csv_path.open("w", newline="", encoding="utf-8")
        writer = self._csv_file

        # Metadata header — lines begin with '#' so pandas can skip them.
        writer.write(f"# subject_id: {self.subject_id}\n")
        writer.write(f"# session_id: {self.session_id}\n")
        writer.write(f"# start_time_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        writer.write(f"# port: {self.port}\n")
        writer.write(f"# baud_rate: {self.baud_rate}\n")
        writer.write(f"# n_channels: {self.n_channels}\n")
        writer.write(f"# channel_names: {self.channel_names}\n")
        writer.write(f"# config: {json.dumps(config_dict)}\n")

        self._csv_writer = csv.writer(writer)
        header = ["wall_time_s", "arduino_timestamp_ms", "seq_id"] + self.channel_names
        self._csv_writer.writerow(header)

    def _read_loop(self) -> None:
        """Main reader loop running in the background thread."""
        assert self._serial is not None

        fs_window_start = time.monotonic()
        fs_window_count = 0
        prev_wall: Optional[float] = None

        while not self._stop_event.is_set():
            try:
                raw_line = self._serial.readline()
            except serial.SerialException as exc:
                logger.error("Serial read error: %s — attempting reconnect", exc)
                self._attempt_reconnect()
                continue

            if not raw_line:
                continue

            packet = self._parse_line(raw_line)
            if packet is None:
                continue

            wall_now = packet.wall_time
            fs_window_count += 1
            self._total_count += 1

            # ── Sequence gap detection ──────────────────────────────────
            self._check_sequence(packet.seq_id)

            # ── Jitter (inter-packet interval) ──────────────────────────
            if prev_wall is not None:
                interval_ms = (wall_now - prev_wall) * 1000.0
                self._recent_intervals.append(interval_ms)
                if len(self._recent_intervals) > 500:
                    self._recent_intervals.pop(0)
            prev_wall = wall_now

            # ── Update stats every second ───────────────────────────────
            elapsed = wall_now - fs_window_start
            if elapsed >= 1.0:
                eff_fs = fs_window_count / elapsed
                intervals = self._recent_intervals[-250:]  # last ~1 s
                jitter_mean = float(np.mean(intervals)) if intervals else 0.0
                jitter_std = float(np.std(intervals)) if intervals else 0.0

                with self._stats_lock:
                    self._stats.effective_fs = eff_fs
                    self._stats.jitter_ms_mean = jitter_mean
                    self._stats.jitter_ms_std = jitter_std
                    self._stats.dropped_packets_total = self._dropped_count
                    self._stats.total_packets = self._total_count
                    if self._total_count > 0:
                        self._stats.loss_fraction = (
                            self._dropped_count / self._total_count
                        )

                logger.debug(
                    "fs=%.1f Hz | jitter=%.2f±%.2f ms | dropped=%d",
                    eff_fs, jitter_mean, jitter_std, self._dropped_count,
                )
                fs_window_start = wall_now
                fs_window_count = 0

            # ── Write to CSV ────────────────────────────────────────────
            if self._csv_writer is not None:
                row = [
                    f"{wall_now:.6f}",
                    packet.timestamp_ms,
                    packet.seq_id,
                ] + packet.channel_data.tolist()
                self._csv_writer.writerow(row)

            # ── Enqueue for downstream consumers ────────────────────────
            try:
                self._queue.put_nowait(packet)
            except queue.Full:
                logger.warning("Consumer queue full — dropping packet seq=%d", packet.seq_id)

    def _parse_line(self, raw: bytes) -> Optional[EEGPacket]:
        """
        Decode a CSV-formatted line from the Arduino.

        Expected format: ``timestamp_ms,seq_id,ch0[,ch1,...]\n``
        Returns ``None`` on malformed input.
        """
        wall_now = time.monotonic()
        try:
            text = raw.decode("ascii", errors="replace").strip()
            if not text or text.startswith("#"):
                return None
            parts = text.split(",")
            if len(parts) < 2 + self.n_channels:
                return None
            ts_ms = int(parts[0])
            seq = int(parts[1]) % self.SEQ_MAX
            channels = np.array([float(parts[2 + i]) for i in range(self.n_channels)],
                                 dtype=np.float64)
            return EEGPacket(
                timestamp_ms=ts_ms,
                seq_id=seq,
                channel_data=channels,
                wall_time=wall_now,
            )
        except (ValueError, IndexError) as exc:
            logger.debug("Malformed packet: %r — %s", raw[:60], exc)
            return None

    def _check_sequence(self, seq: int) -> None:
        """Update dropped-packet counter based on seq_id gap."""
        if self._last_seq is None:
            self._last_seq = seq
            return
        expected = (self._last_seq + 1) % self.SEQ_MAX
        if seq != expected:
            gap = (seq - self._last_seq) % self.SEQ_MAX
            dropped = max(0, gap - 1)
            if dropped > 0:
                logger.warning(
                    "Dropped %d packet(s): expected seq=%d got seq=%d",
                    dropped, expected, seq,
                )
                self._dropped_count += dropped
        self._last_seq = seq

    def _attempt_reconnect(self) -> None:
        """Try to reopen the serial port after a read failure."""
        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass
        for attempt in range(1, self.reconnect_attempts + 1):
            time.sleep(self.reconnect_delay_sec)
            try:
                self._serial.open()
                self._serial.reset_input_buffer()
                logger.info("Reconnected to %s (attempt %d)", self.port, attempt)
                return
            except serial.SerialException as exc:
                logger.warning("Reconnect attempt %d failed: %s", attempt, exc)
        logger.error("All reconnect attempts exhausted — stopping reader.")
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Buffer helper used by both offline recorder and realtime decoder
# ---------------------------------------------------------------------------

class RingBuffer:
    """
    Fixed-length FIFO buffer for EEG samples.

    Parameters
    ----------
    capacity_samples : int
        Maximum number of samples to hold (oldest are overwritten).
    n_channels : int
        Number of EEG channels per sample.
    """

    def __init__(self, capacity_samples: int, n_channels: int) -> None:
        self._buf = np.full((capacity_samples, n_channels), np.nan, dtype=np.float64)
        self._ts = np.full(capacity_samples, np.nan, dtype=np.float64)
        self._capacity = capacity_samples
        self._n = n_channels
        self._head = 0       # next write index
        self._count = 0      # number of valid samples

    def push(self, sample: np.ndarray, timestamp: float) -> None:
        """Append one sample (shape ``(n_channels,)``) with its wall timestamp."""
        self._buf[self._head] = sample
        self._ts[self._head] = timestamp
        self._head = (self._head + 1) % self._capacity
        self._count = min(self._count + 1, self._capacity)

    def get_window(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the last ``n_samples`` samples.

        Returns
        -------
        data : np.ndarray, shape (n_samples, n_channels)
        timestamps : np.ndarray, shape (n_samples,)

        Raises
        ------
        BufferError
            If fewer than ``n_samples`` samples are available.
        """
        if self._count < n_samples:
            raise BufferError(
                f"Buffer has {self._count} samples; {n_samples} requested."
            )
        # Indices of the last n_samples in chronological order
        tail = (self._head - n_samples) % self._capacity
        if tail + n_samples <= self._capacity:
            data = self._buf[tail : tail + n_samples].copy()
            ts = self._ts[tail : tail + n_samples].copy()
        else:
            # Wraps around
            split = self._capacity - tail
            data = np.vstack([self._buf[tail:], self._buf[:n_samples - split]])
            ts = np.concatenate([self._ts[tail:], self._ts[:n_samples - split]])
        return data, ts

    @property
    def n_samples(self) -> int:
        """Number of valid samples currently in the buffer."""
        return self._count
