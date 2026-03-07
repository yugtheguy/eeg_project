"""
realtime/dashboard.py
======================
PyQtGraph-based real-time EEG monitoring dashboard.

Panels
------
1. Raw EEG scrolling traces (T7, T8)
2. Alpha power time-series (left / right channels)
3. Lateralization index (LI) bar gauge
4. Class probability gauge (posterior probabilities per class)

Usage
-----
::

    app = QtWidgets.QApplication(sys.argv)
    dashboard = EEGDashboard(cfg, class_names=["Left", "Right"])
    dashboard.show()
    # In your acquisition loop:
    dashboard.update(raw_sample, decode_result)
    app.exec()

The dashboard is designed to run in the main GUI thread; the
SerialReader and RealtimeDecoder run in background threads.
Thread-safe update is guaranteed by posting Qt signals from worker
threads.

Dependencies
------------
pip install pyqtgraph PyQt6   (or PyQt5 / PySide6 — pyqtgraph supports all)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — guard against missing PyQtGraph at import time so that
# offline-only usage of the project does not crash.
# ---------------------------------------------------------------------------
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
    _PG_AVAILABLE = True
except ImportError:
    _PG_AVAILABLE = False
    logger.warning(
        "PyQtGraph not found. Install with: pip install pyqtgraph PyQt6\n"
        "The dashboard will not be available."
    )


# ---------------------------------------------------------------------------
# Guard decorator
# ---------------------------------------------------------------------------

def _require_pyqtgraph(fn):
    def wrapper(*args, **kwargs):
        if not _PG_AVAILABLE:
            raise ImportError(
                "PyQtGraph is required for the dashboard.\n"
                "Install: pip install pyqtgraph PyQt6"
            )
        return fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Dashboard widget
# ---------------------------------------------------------------------------

class EEGDashboard:
    """
    Real-time EEG attention decoder dashboard.

    Parameters
    ----------
    cfg : dict
        Full YAML config.
    class_names : list of str
        Human-readable class labels (e.g. ["Left", "Right"]).
    history_sec : float
        Duration of raw-trace history to display.
    """

    TRACE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    ALPHA_COLORS = {"left": "#1f77b4", "right": "#ff7f0e"}

    @_require_pyqtgraph
    def __init__(
        self,
        cfg: Dict,
        class_names: Optional[List[str]] = None,
        history_sec: float = 5.0,
    ) -> None:
        self._cfg         = cfg
        self._fs          = float(cfg["hardware"]["sampling_rate"])
        self._n_channels  = int(cfg["hardware"]["n_channels"])
        self._ch_names    = cfg["hardware"]["channel_names"]
        self._class_names = class_names or ["Left", "Right", "Neutral"]
        self._history_sec = history_sec
        self._history_samples = int(history_sec * self._fs)

        # Rolling buffers for display
        self._raw_buf    = np.zeros((self._history_samples, self._n_channels))
        self._alpha_left_hist:  List[float] = []
        self._alpha_right_hist: List[float] = []
        self._li_hist:          List[float] = []
        self._time_axis: List[float] = []          # relative seconds
        self._t_elapsed = 0.0
        self._update_interval = 1.0 / cfg["realtime"].get("dashboard_update_hz", 10)

        pg.setConfigOptions(antialias=True, background="k", foreground="w")
        self._win = pg.GraphicsLayoutWidget(title="EEG Attention Decoder")
        self._win.resize(1280, 900)
        self._build_layout()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Construct all plot panels."""
        win = self._win

        # ── Row 0: Raw EEG traces ──────────────────────────────────────
        self._raw_plot = win.addPlot(row=0, col=0, colspan=2,
                                     title="Raw EEG (µV)")
        self._raw_plot.showGrid(x=True, y=True, alpha=0.3)
        self._raw_plot.setLabel("bottom", "Time (s)")
        self._raw_plot.setLabel("left", "Amplitude (µV)")
        self._raw_curves = []
        for ch in range(self._n_channels):
            curve = self._raw_plot.plot(
                pen=pg.mkPen(self.TRACE_COLORS[ch % len(self.TRACE_COLORS)], width=1),
                name=self._ch_names[ch],
            )
            self._raw_curves.append(curve)
        self._raw_plot.addLegend()

        # ── Row 1: Alpha power traces ──────────────────────────────────
        self._alpha_plot = win.addPlot(row=1, col=0, colspan=2,
                                       title="Log Alpha Power (dB)")
        self._alpha_plot.showGrid(x=True, y=True, alpha=0.3)
        self._alpha_plot.setLabel("bottom", "Time (s)")
        self._alpha_plot.setLabel("left", "Log Power (dB)")
        self._alpha_left_curve = self._alpha_plot.plot(
            pen=pg.mkPen(self.ALPHA_COLORS["left"], width=2), name="Alpha Left"
        )
        self._alpha_right_curve = self._alpha_plot.plot(
            pen=pg.mkPen(self.ALPHA_COLORS["right"], width=2), name="Alpha Right"
        )
        self._alpha_plot.addLegend()

        # ── Row 2: LI bar ─────────────────────────────────────────────
        self._li_plot = win.addPlot(row=2, col=0, title="Lateralization Index")
        self._li_plot.setYRange(-1.1, 1.1)
        self._li_plot.setXRange(-0.5, 0.5)
        self._li_plot.showGrid(x=False, y=True, alpha=0.3)
        self._li_plot.addLine(y=0, pen=pg.mkPen("w", style=QtCore.Qt.PenStyle.DashLine))

        self._li_bar = pg.BarGraphItem(x=[0], height=[0], width=0.8, brush="#2ca02c")
        self._li_plot.addItem(self._li_bar)
        self._li_label = pg.TextItem("LI = 0.000", color="w", anchor=(0.5, 0))
        self._li_label.setPos(0, 1.0)
        self._li_plot.addItem(self._li_label)

        # ── Row 2, col 1: Probability gauge ────────────────────────────
        self._prob_plot = win.addPlot(row=2, col=1, title="Class Probabilities")
        n_cls = len(self._class_names)
        self._prob_plot.setYRange(0, 1.05)
        self._prob_plot.setXRange(-0.5, n_cls - 0.5)
        self._prob_bars = pg.BarGraphItem(
            x=np.arange(n_cls),
            height=np.full(n_cls, 1.0 / n_cls),
            width=0.7,
            brushes=[pg.mkBrush(self.TRACE_COLORS[i % len(self.TRACE_COLORS)])
                     for i in range(n_cls)],
        )
        self._prob_plot.addItem(self._prob_bars)
        self._prob_plot.getAxis("bottom").setTicks(
            [[(i, name) for i, name in enumerate(self._class_names)]]
        )
        self._prob_plot.setLabel("left", "Probability")
        self._prob_label = pg.TextItem("—", color="w", anchor=(0.5, 0))
        self._prob_label.setPos((n_cls - 1) / 2, 0.95)
        self._prob_plot.addItem(self._prob_label)

    # ------------------------------------------------------------------
    # Public update API
    # ------------------------------------------------------------------

    def update(
        self,
        raw_sample: np.ndarray,
        decode_result=None,   # realtime.decoder.DecodeResult | None
    ) -> None:
        """
        Push one raw sample and (optionally) the latest decode result.

        Call this from the acquisition/decode callback.

        Parameters
        ----------
        raw_sample : np.ndarray, shape (n_channels,)
        decode_result : DecodeResult or None
        """
        # Scroll raw buffer
        self._raw_buf = np.roll(self._raw_buf, -1, axis=0)
        self._raw_buf[-1] = raw_sample[:self._n_channels]

        t_axis = np.linspace(-self._history_sec, 0, self._history_samples)
        for ch, curve in enumerate(self._raw_curves):
            offset = ch * 50.0   # vertical separation in µV
            curve.setData(t_axis, self._raw_buf[:, ch] + offset)

        if decode_result is not None:
            self._t_elapsed += self._update_interval
            self._time_axis.append(self._t_elapsed)
            self._alpha_left_hist.append(decode_result.alpha_power_left_db)
            self._alpha_right_hist.append(decode_result.alpha_power_right_db)
            self._li_hist.append(decode_result.lateralization_index)

            # Keep 60 s history for power plots
            max_pts = int(60.0 / self._update_interval)
            self._time_axis       = self._time_axis[-max_pts:]
            self._alpha_left_hist = self._alpha_left_hist[-max_pts:]
            self._alpha_right_hist= self._alpha_right_hist[-max_pts:]
            self._li_hist         = self._li_hist[-max_pts:]

            # Alpha power curves
            t = np.array(self._time_axis)
            self._alpha_left_curve.setData(t, np.array(self._alpha_left_hist))
            self._alpha_right_curve.setData(t, np.array(self._alpha_right_hist))

            # LI bar
            li = decode_result.lateralization_index
            color = "#1f77b4" if li < 0 else "#d62728"
            self._li_bar.setOpts(x=[0], height=[li], brush=color)
            self._li_label.setText(f"LI = {li:+.3f}")

            # Probability gauge
            proba = decode_result.probabilities
            self._prob_bars.setOpts(height=proba)
            cls_name = self._class_names[decode_result.predicted_class]
            conf = float(proba[decode_result.predicted_class]) * 100
            self._prob_label.setText(f"{cls_name} ({conf:.0f}%)")

    # ------------------------------------------------------------------
    # Show / hide
    # ------------------------------------------------------------------

    def show(self) -> None:
        """Display the dashboard window."""
        self._win.show()

    def close(self) -> None:
        """Close the dashboard window."""
        self._win.close()


# ---------------------------------------------------------------------------
# Convenience launcher (blocking)
# ---------------------------------------------------------------------------

@_require_pyqtgraph
def launch_dashboard(cfg: Dict, class_names: Optional[List[str]] = None) -> None:
    """
    Launch the dashboard in a standalone blocking process.

    Intended for offline testing of the dashboard layout.
    Pass synthetic data via the update() method in a QTimer.
    """
    import sys
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    db = EEGDashboard(cfg, class_names=class_names)
    db.show()
    app.exec()
