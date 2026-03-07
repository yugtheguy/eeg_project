"""
run_training_test.py
====================
End-to-end integration test for the EEG Attention Decoder using a fully
synthetic dataset.  No hardware required.

Steps
-----
1. Generate synthetic dataset  (generate_dataset.py)
2. Run ``main.py train`` via subprocess (uses the real pipeline unchanged)
3. Produce validation plots in tests/synthetic_dataset/results/

Run from the eeg_attention_decoder/ directory::

    python tests/synthetic_dataset/run_training_test.py

Expected outcome
----------------
* CV window accuracy: 65 – 85 %
* Permutation p-value: < 0.05  (reject null)
* Model saved to models/saved/
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
HERE     = Path(__file__).resolve().parent          # tests/synthetic_dataset/
PKG_ROOT = HERE.parent.parent                       # eeg_attention_decoder/
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(HERE))                       # for generate_dataset

RESULTS_DIR = HERE / "results"
SUBJECT_ID  = "sub-sim"
SESSION_ID  = "ses-01"
FS          = 250


# ---------------------------------------------------------------------------
# Matplotlib — required for plots; not in main requirements.txt
# ---------------------------------------------------------------------------
def _require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib not found — installing …")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "matplotlib"],
            check=True,
        )


# ---------------------------------------------------------------------------
# Step 1 — Generate dataset
# ---------------------------------------------------------------------------

def step_1_generate():
    sep = "=" * 60
    print(f"\n{sep}\nSTEP 1 — Generating synthetic dataset\n{sep}")
    from generate_dataset import generate_all
    result = generate_all()
    print("Dataset generation complete.\n")
    return result


# ---------------------------------------------------------------------------
# Step 2 — Run training pipeline
# ---------------------------------------------------------------------------

def step_2_train() -> subprocess.CompletedProcess:
    sep = "=" * 60
    print(f"\n{sep}\nSTEP 2 — Running main.py train\n{sep}")

    cmd = [
        sys.executable, "main.py", "train",
        "--subject", SUBJECT_ID,
        "--session", SESSION_ID,
        "--data-dir", "data/raw",
    ]
    print("Command:", " ".join(cmd))
    print(f"CWD:     {PKG_ROOT}\n")

    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=PKG_ROOT)
    elapsed = time.monotonic() - t0

    print(
        f"\nTraining finished in {elapsed:.1f} s  "
        f"(exit code {proc.returncode})"
    )
    if proc.returncode != 0:
        print(
            "WARNING: Training pipeline exited with a non-zero code.\n"
            "Check the output above for error details."
        )
    return proc


# ---------------------------------------------------------------------------
# Step 3 — Validation plots
# ---------------------------------------------------------------------------

def step_3_plots(gen_result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    sep = "=" * 60
    print(f"\n{sep}\nSTEP 3 — Generating validation plots\n{sep}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t7         = gen_result["t7"]
    t8         = gen_result["t8"]
    protocol   = gen_result["protocol"]
    n_samples  = gen_result["n_samples"]
    left_li    = gen_result["left_li"]
    right_li   = gen_result["right_li"]

    # ── Plot 1: Power Spectrum ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, sig, ch_name in zip(axes, [t7, t8], ["T7 (left hemisphere)", "T8 (right hemisphere)"]):
        f, psd = welch(sig, fs=FS, nperseg=4 * FS)
        ax.semilogy(f, psd, lw=1.5, color="steelblue")
        ax.axvspan(8.2, 12.2, alpha=0.20, color="darkorange",
                   label=f"Alpha band\n(IAF ± 2 Hz = 8.2–12.2 Hz)")
        ax.axvline(10.2, color="darkorange", ls="--", lw=1.2, alpha=0.8)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (µV²/Hz)")
        ax.set_title(f"PSD — {ch_name}")
        ax.set_xlim(0, 40)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.35)

    fig.suptitle("Synthetic EEG Power Spectrum (pre-filter)", fontweight="bold", fontsize=13)
    fig.tight_layout()
    out = RESULTS_DIR / "power_spectrum.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.relative_to(HERE)}")

    # ── Plot 2: Alpha Lateralisation Histogram ─────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(
        min(min(left_li, default=-0.3), min(right_li, default=-0.3)) - 0.05,
        max(max(left_li, default=0.3),  max(right_li, default=0.3))  + 0.05,
        35,
    )
    if left_li:
        ax.hist(left_li,  bins=bins, alpha=0.70, color="steelblue",
                label=f"LEFT attention (n={len(left_li)} blocks)")
        ax.axvline(np.mean(left_li),  color="steelblue", ls="--", lw=2,
                   label=f"LEFT mean = {np.mean(left_li):+.3f}")
    if right_li:
        ax.hist(right_li, bins=bins, alpha=0.70, color="tomato",
                label=f"RIGHT attention (n={len(right_li)} blocks)")
        ax.axvline(np.mean(right_li), color="tomato",    ls="--", lw=2,
                   label=f"RIGHT mean = {np.mean(right_li):+.3f}")

    ax.set_xlabel("Lateralisation Index  (T7 − T8) / (T7 + T8)")
    ax.set_ylabel("Number of blocks")
    ax.set_title(
        "Alpha Power Lateralisation by Attention Condition\n"
        "(raw synthetic signal, before bandpass filter)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    out = RESULTS_DIR / "alpha_lateralization_hist.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out.relative_to(HERE)}")

    # ── Plot 3: Training Metrics Summary (confusion_matrix.png) ───────────
    log_dir   = PKG_ROOT / "logs"
    log_files = sorted(log_dir.glob(f"train_{SUBJECT_ID}_{SESSION_ID}_*.json"))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis("off")

    if log_files:
        with open(log_files[-1], encoding="utf-8") as fh:
            log = json.load(fh)

        wm  = log.get("window_metrics", {})
        ba  = log.get("block_accuracy",  "N/A")
        pv  = log.get("perm_p_value",    "N/A")
        rn  = log.get("perm_reject_null","N/A")
        nw  = log.get("n_windows",       "N/A")

        rows = [
            ["Metric",                  "Value",                               "CI / Note"],
            ["Window accuracy",         f"{wm.get('accuracy', 'N/A'):.1%}"    if isinstance(wm.get('accuracy'), float) else "N/A",
                                        f"[{wm.get('accuracy_ci_lo','?'):.1%} – {wm.get('accuracy_ci_hi','?'):.1%}]"
                                        if isinstance(wm.get('accuracy_ci_lo'), float) else ""],
            ["Balanced accuracy",       f"{wm.get('balanced_accuracy','N/A'):.1%}"
                                        if isinstance(wm.get('balanced_accuracy'), float) else "N/A",
                                        f"[{wm.get('bacc_ci_lo','?'):.1%} – {wm.get('bacc_ci_hi','?'):.1%}]"
                                        if isinstance(wm.get('bacc_ci_lo'), float) else ""],
            ["ROC-AUC",                 f"{wm.get('roc_auc', 'N/A'):.3f}"
                                        if isinstance(wm.get('roc_auc'), float) else "N/A",       ""],
            ["Block accuracy",          f"{ba:.1%}" if isinstance(ba, float) else str(ba),        "majority vote"],
            ["Permutation p-value",     f"{pv:.4f}" if isinstance(pv, float) else str(pv),
                                        "reject null" if rn else "fail to reject"],
            ["Null rejected (p<0.05)", str(rn),                                                   ""],
            ["Total CV windows",        str(nw),                                                   ""],
            ["Run log",                 log_files[-1].name,                                        ""],
        ]

        col_w = [0.36, 0.32, 0.32]
        header_color = "#2c4f7c"
        row_colors   = ["#ddeeff", "#ffffff"]

        table = ax.table(
            cellText=rows[1:],
            colLabels=rows[0],
            cellLoc="center",
            loc="center",
            colWidths=col_w,
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)

        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor(header_color)
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor(row_colors[r % 2])
            cell.set_edgecolor("#aaaaaa")

        ax.set_title(
            "EEG Attention Decoder — Training Pipeline Results\n"
            f"Subject: {SUBJECT_ID}  |  Session: {SESSION_ID}  |  Synthetic dataset",
            fontweight="bold", fontsize=11, pad=12,
        )
        print(f"\n  Results from log: {log_files[-1].name}")
        print(f"    Window accuracy : {wm.get('accuracy', '?')}")
        print(f"    Balanced acc    : {wm.get('balanced_accuracy', '?')}")
        print(f"    ROC-AUC         : {wm.get('roc_auc', '?')}")
        print(f"    Block accuracy  : {ba}")
        print(f"    Perm p-value    : {pv}")
        print(f"    Reject null     : {rn}")
    else:
        ax.text(
            0.5, 0.5,
            "No training log found.\nCheck that main.py train completed successfully.",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=12, color="red",
        )
        ax.set_title("Training Pipeline Results — (no log found)", fontweight="bold")

    fig.tight_layout()
    out = RESULTS_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.relative_to(HERE)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _require_matplotlib()

    print("\n" + "#" * 60)
    print("  EEG Attention Decoder — Synthetic Dataset Integration Test")
    print("#" * 60)
    print(f"  Package root : {PKG_ROOT}")
    print(f"  Results dir  : {RESULTS_DIR}")

    gen_result = step_1_generate()
    proc       = step_2_train()
    step_3_plots(gen_result)

    sep = "=" * 60
    print(f"\n{sep}")
    print("Integration test complete.")
    print(f"  Validation plots : {RESULTS_DIR}")
    print(f"  Models           : {PKG_ROOT / 'models' / 'saved'}")
    print(f"  Run logs         : {PKG_ROOT / 'logs'}")
    print(f"  Pipeline exit    : {proc.returncode}")
    print(sep)

    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
