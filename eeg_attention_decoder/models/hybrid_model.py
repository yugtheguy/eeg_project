"""
models/hybrid_model.py
=======================
Stub for the Phase-2 hybrid EEG + auditory envelope attention decoder.

Phase-1 status
--------------
This module defines the interface contract and provides a functional
EEG-only baseline that delegates to ``AttentionLDA``.  The envelope
branch is intentionally left as a documented extension point — no
``pass`` stubs without explanation.

Envelope integration notes (Phase 2)
--------------------------------------
* The auditory envelope will be extracted from the attended speech
  stream via a half-wave rectifier + low-pass (< 8 Hz) cascade.
* Envelope features will be aligned to EEG windows by the same
  centre-timestamp mechanism used in ``processing/features.py``.
* The fused model will concatenate [eeg_features | envelope_features]
  and retrain from scratch — no incremental learning.
* The interface below is designed so that envelope=None produces
  identical output to the Phase-1 LDA model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from models.lda import AttentionLDA

logger = logging.getLogger(__name__)


class HybridAttentionDecoder:
    """
    Hybrid EEG + auditory envelope attention decoder.

    In Phase 1 (envelope=None) this class is a thin wrapper around
    ``AttentionLDA``.  In Phase 2 the envelope features extend the
    feature vector before classification.

    Parameters
    ----------
    eeg_only_model : AttentionLDA
        A fitted (or unfitted) LDA instance for the EEG branch.
    envelope_feature_dim : int
        Expected dimensionality of the envelope feature vector when
        Phase 2 integration is active.  Set to 0 in Phase 1.
    """

    def __init__(
        self,
        eeg_only_model: Optional[AttentionLDA] = None,
        envelope_feature_dim: int = 0,
    ) -> None:
        self._eeg_model = eeg_only_model or AttentionLDA()
        self._envelope_feature_dim = envelope_feature_dim
        self._phase2_enabled = False

    # ------------------------------------------------------------------
    # Unified fit interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_eeg: np.ndarray,
        y: np.ndarray,
        X_envelope: Optional[np.ndarray] = None,
        training_metadata: Optional[Dict] = None,
    ) -> "HybridAttentionDecoder":
        """
        Fit the decoder.

        Parameters
        ----------
        X_eeg : np.ndarray, shape (n_windows, n_eeg_features)
            EEG feature matrix (from ``processing/features.py``).
        y : np.ndarray, shape (n_windows,)
            Integer class labels.
        X_envelope : np.ndarray, shape (n_windows, n_env_features), optional
            Auditory envelope features aligned to EEG windows.
            Pass ``None`` to use EEG-only (Phase 1).
        training_metadata : dict, optional
            Session metadata propagated to the inner model.

        Returns
        -------
        self
        """
        if X_envelope is not None:
            self._phase2_enabled = True
            if X_envelope.shape[0] != X_eeg.shape[0]:
                raise ValueError(
                    "X_eeg and X_envelope must have the same number of windows; "
                    f"got {X_eeg.shape[0]} vs {X_envelope.shape[0]}."
                )
            X = self._fuse_features(X_eeg, X_envelope)
            logger.info(
                "Phase 2: fusing EEG (%d features) + envelope (%d features).",
                X_eeg.shape[1], X_envelope.shape[1],
            )
        else:
            self._phase2_enabled = False
            X = X_eeg
            logger.info("Phase 1: EEG-only model (%d features).", X_eeg.shape[1])

        meta = training_metadata or {}
        meta["envelope_enabled"] = self._phase2_enabled
        meta["envelope_feature_dim"] = (
            X_envelope.shape[1] if X_envelope is not None else 0
        )
        self._eeg_model.fit(X, y, training_metadata=meta)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        X_eeg: np.ndarray,
        X_envelope: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Hard-label prediction (see ``predict_proba`` for probabilities)."""
        X = self._prepare_input(X_eeg, X_envelope)
        return self._eeg_model.predict(X)

    def predict_proba(
        self,
        X_eeg: np.ndarray,
        X_envelope: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Posterior class probabilities.

        Returns
        -------
        proba : np.ndarray, shape (n_windows, n_classes)
        """
        X = self._prepare_input(X_eeg, X_envelope)
        return self._eeg_model.predict_proba(X)

    # ------------------------------------------------------------------
    # Persistence (delegates to inner LDA)
    # ------------------------------------------------------------------

    def save(
        self,
        directory: Path,
        subject_id: str,
        session_id: str,
    ) -> Tuple[Path, Path]:
        """Save the inner model; see ``AttentionLDA.save`` for details."""
        return self._eeg_model.save(directory, subject_id, session_id)

    @classmethod
    def load(cls, model_path: Path, meta_path=None) -> "HybridAttentionDecoder":
        """Load a saved hybrid decoder."""
        eeg_model = AttentionLDA.load(model_path, meta_path)
        instance = cls(eeg_only_model=eeg_model)
        meta = eeg_model._training_metadata
        instance._phase2_enabled = meta.get("envelope_enabled", False)
        instance._envelope_feature_dim = meta.get("envelope_feature_dim", 0)
        return instance

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _fuse_features(
        self,
        X_eeg: np.ndarray,
        X_envelope: np.ndarray,
    ) -> np.ndarray:
        """
        Concatenate EEG and envelope feature matrices along the feature axis.

        Phase 2 extension point: more sophisticated fusion (e.g. weighted
        combination, cross-modal attention) can replace this method.
        """
        return np.concatenate([X_eeg, X_envelope], axis=1)

    def _prepare_input(
        self,
        X_eeg: np.ndarray,
        X_envelope: Optional[np.ndarray],
    ) -> np.ndarray:
        if self._phase2_enabled and X_envelope is not None:
            return self._fuse_features(X_eeg, X_envelope)
        if self._phase2_enabled and X_envelope is None:
            logger.warning(
                "Model was trained with envelope features but none provided "
                "at inference — using EEG features only."
            )
        return X_eeg
