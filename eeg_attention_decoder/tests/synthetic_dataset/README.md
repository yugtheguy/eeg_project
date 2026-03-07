# Synthetic Dataset — EEG Attention Decoder

Integration test that exercises the full training pipeline without any
hardware.  A realistic synthetic EEG dataset is generated, fed into the
unchanged `main.py train` pipeline, and validation plots are produced.

---

## Directory layout

```
tests/synthetic_dataset/
├── generate_dataset.py      # Signal synthesis + file I/O
├── run_training_test.py     # Orchestrator (generate → train → plot)
├── README.md                # This file
└── results/                 # Created on first run
    ├── power_spectrum.png
    ├── alpha_lateralization_hist.png
    └── confusion_matrix.png  # Training metrics table
```

Generated data (inside `eeg_attention_decoder/`):

```
data/raw/sub-sim/ses-01/
├── raw_eeg_simulated.csv   (~150 MB, ~20 min @ 250 Hz)
├── protocol.json           (block design, 160 blocks)
└── iaf.json                (IAF = 10.2 Hz)
```

---

## Prerequisites

```
pip install -r requirements.txt   # core deps (already installed)
pip install matplotlib             # only needed for the plots
```

---

## How to generate the dataset only

Run from the `eeg_attention_decoder/` directory:

```bash
python tests/synthetic_dataset/generate_dataset.py
```

This creates `data/raw/sub-sim/ses-01/` with all three required files.
It does **not** run the training pipeline.

---

## How to run the full integration test

Run from the `eeg_attention_decoder/` directory:

```bash
python tests/synthetic_dataset/run_training_test.py
```

This will:

1. **Generate** the synthetic dataset (skips regeneration if files exist —
   delete them manually to regenerate).
2. **Run** `python main.py train --subject sub-sim --session ses-01 --data-dir data/raw`
   using the real, unmodified pipeline.
3. **Save** three diagnostic plots to `tests/synthetic_dataset/results/`.

Expected wall-clock time: 3–10 minutes (dominated by 1000-permutation test).

---

## Signal model

| Component    | Frequency | Amplitude | Notes                          |
|--------------|-----------|-----------|--------------------------------|
| Delta        | 3 Hz      | 35 µV     | Within 2–25 Hz bandpass        |
| Theta        | 6 Hz      | 22 µV     | Within 2–25 Hz bandpass        |
| **Alpha**    | **10.2 Hz** | **22 µV** | **Modulated by attention**   |
| Beta         | 20 Hz     | 9 µV      | Within 2–25 Hz bandpass        |
| Pink noise   | 1/f       | σ = 28 µV | Realistic spectral tilt        |
| White noise  | broadband | σ = 6 µV  |                                |

### Spatial attention lateralisation

| Condition | T7 (left hemisphere) | T8 (right hemisphere) |
|-----------|----------------------|----------------------|
| LEFT      | alpha −10 %          | alpha +35 %          |
| RIGHT     | alpha +35 %          | alpha −10 %          |
| NEUTRAL   | no change            | no change            |
| CATCH     | no change            | no change            |

This matches the contralateral alpha suppression observed in human EEG
(Jensen & Mazaheri 2010; Worden et al. 2000).

---

## Protocol design

| Parameter              | Value                        |
|------------------------|------------------------------|
| Total blocks           | 160                          |
| Block duration         | 5 – 8 s (uniform random)     |
| LEFT attention blocks  | ~32                          |
| RIGHT attention blocks | ~32                          |
| NEUTRAL blocks         | ~32                          |
| CATCH blocks           | ~32 (20 % of 160)            |
| BILATERAL blocks       | ~32 (20 % of 160)            |
| Total duration         | ~17 min (depends on RNG)     |
| Seed                   | 42 (reproducible)            |

---

## Expected training results

After running `main.py train`, you should see approximately:

| Metric                  | Expected range  |
|-------------------------|-----------------|
| Window accuracy         | 65 % – 82 %     |
| Balanced accuracy       | 63 % – 80 %     |
| Block-level accuracy    | 68 % – 88 %     |
| ROC-AUC                 | 0.70 – 0.90     |
| Permutation p-value     | < 0.05          |
| Null hypothesis rejected | Yes            |

If window accuracy exceeds 90 %, the alpha modulation (`ALPHA_BOOST`) in
`generate_dataset.py` is too strong — reduce it toward 0.20.

---

## Output files

After a successful run:

```
models/saved/lda_sub-sim_ses-01_<timestamp>.joblib   # fitted LDA model
models/saved/lda_sub-sim_ses-01_<timestamp>_meta.json
logs/train_sub-sim_ses-01_<timestamp>.json           # full metrics log
logs/train_sub-sim_ses-01_<timestamp>_null_dist.png  # permutation null dist
tests/synthetic_dataset/results/
    power_spectrum.png
    alpha_lateralization_hist.png
    confusion_matrix.png
```

---

## Constraints respected

* No existing project files are modified.
* No pipeline code is duplicated — `generate_dataset.py` only produces
  input files; all filtering, feature extraction, CV, and model training
  run through the unmodified `main.py` and its submodules.
* Configuration is read from `configs/default.yaml` (unchanged).
* The synthetic subject/session (`sub-sim / ses-01`) is separate from any
  real recorded data.
