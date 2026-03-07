"""
models/lda.py
=============
Linear Discriminant Analysis classifier for EEG spatial attention decoding.

Design constraints
------------------
* Features are standardised with a ``StandardScaler`` fitted on training
  data only (never on test or validation data).
* ``random_state`` is always deterministic (from config / explicit arg).
* Model artifacts are saved via joblib with full training metadata.
* Shrinkage is "auto" (Ledoit–Wolf) by default for well-conditioned
  covariance estimation with small sample sizes.
* Hyperparameters are set before cross-validation; no tuning on test folds.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model bundle
# ---------------------------------------------------------------------------

class AttentionLDA:
    """
    Standardised-LDA pipeline for EEG spatial attention classification.

    Parameters
    ----------
    shrinkage : str or float
        Passed to ``LinearDiscriminantAnalysis``.  ``"auto"`` enables
        Ledoit–Wolf shrinkage (recommended for n_features >> n_samples).
    random_state : int
        Seed for deterministic behaviour.  Must come from config.
    """

    FEATURE_NAMES = [
        "log_alpha_left", "log_alpha_right",
        "log_beta_left",  "log_beta_right",
        "rel_alpha_left", "rel_alpha_right",
        "lateralization_index",
    ]
    # Coherence appended dynamically if enabled.

    def __init__(
        self,
        shrinkage: Any = "auto",
        random_state: int = 42,
    ) -> None:
        self.shrinkage = shrinkage
        self.random_state = random_state
        self._pipeline: Optional[Pipeline] = None
        self._training_metadata: Dict = {}

    # ------------------------------------------------------------------
    # sklearn-style API
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        training_metadata: Optional[Dict] = None,
    ) -> "AttentionLDA":
        """
        Fit scaler + LDA on training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_windows, n_features)
            Feature matrix — must contain only genuine training windows
            (no test windows; caller is responsible).
        y : np.ndarray, shape (n_windows,)
            Integer class labels.
        training_metadata : dict, optional
            Session metadata to embed in the saved artifact
            (e.g. IAF, filter settings, window size).
        """
        lda = LinearDiscriminantAnalysis(
            solver="eigen",
            shrinkage=self.shrinkage,
        )
        scaler = StandardScaler()
        self._pipeline = Pipeline([("scaler", scaler), ("lda", lda)])
        self._pipeline.fit(X, y)

        self._training_metadata = {
            "fit_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_training_windows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "classes": np.unique(y).tolist(),
            "shrinkage": str(self.shrinkage),
            "random_state": self.random_state,
        }
        if training_metadata:
            self._training_metadata.update(training_metadata)

        logger.info(
            "LDA fitted: %d windows, %d features, classes=%s",
            X.shape[0], X.shape[1], np.unique(y).tolist(),
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard-label prediction."""
        self._check_fitted()
        return self._pipeline.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Posterior class probabilities.

        Returns
        -------
        proba : np.ndarray, shape (n_windows, n_classes)
        """
        self._check_fitted()
        return self._pipeline.predict_proba(X)  # type: ignore[union-attr]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Standard accuracy on labelled data."""
        self._check_fitted()
        return float(self._pipeline.score(X, y))  # type: ignore[union-attr]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project to LDA space (for visualisation)."""
        self._check_fitted()
        scaler = self._pipeline.named_steps["scaler"]  # type: ignore[union-attr]
        lda    = self._pipeline.named_steps["lda"]
        return lda.transform(scaler.transform(X))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        directory: Path,
        subject_id: str,
        session_id: str,
    ) -> Tuple[Path, Path]:
        """
        Save pipeline and metadata to ``directory``.

        Returns
        -------
        model_path : Path (joblib)
        meta_path  : Path (JSON)
        """
        self._check_fitted()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%dT%H%M%S")
        stem = f"lda_{subject_id}_{session_id}_{timestamp}"

        model_path = directory / f"{stem}.joblib"
        meta_path  = directory / f"{stem}_meta.json"

        joblib.dump(self._pipeline, model_path, compress=3)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(self._training_metadata, fh, indent=2)

        logger.info("Model saved: %s", model_path)
        logger.info("Metadata saved: %s", meta_path)
        return model_path, meta_path

    @classmethod
    def load(cls, model_path: Path, meta_path: Optional[Path] = None) -> "AttentionLDA":
        """
        Load a previously saved model.

        Parameters
        ----------
        model_path : Path to ``.joblib`` file.
        meta_path  : Optional JSON metadata path.

        Returns
        -------
        AttentionLDA instance (fitted).
        """
        instance = cls()
        instance._pipeline = joblib.load(model_path)
        if meta_path and Path(meta_path).exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                instance._training_metadata = json.load(fh)
        logger.info("Model loaded from %s", model_path)
        return instance

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._pipeline is None:
            raise RuntimeError(
                "Model has not been fitted. Call .fit() before predict/save."
            )
