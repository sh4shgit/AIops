"""Distribution drift detection for feature vectors.

Compares incoming feature vectors against a stored reference distribution
(computed from the training set) and flags significant drift so that
operators know when the model should be retrained.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()

_REFERENCE_FILENAME = "drift_reference.json"

# Default z-score threshold: features with |z| above this are considered drifted.
_DEFAULT_Z_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class DriftResult:
    """Outcome of a drift check against the reference distribution.

    Attributes:
        is_drifted: ``True`` if the overall vector is considered drifted.
        drift_score: Aggregate measure of how far the vector is from the
            reference (mean of absolute z-scores across features).
        features_drifted: Names/indices of individual features that exceed the
            z-score threshold.
        recommendation: Human-readable guidance, e.g. *"retrain recommended"*.
    """

    is_drifted: bool
    drift_score: float
    features_drifted: list[str]
    recommendation: str


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class DriftDetector:
    """Simple z-score–based distribution drift detector.

    Call ``fit`` (or ``load``) to establish the reference distribution, then
    ``check_drift`` on each incoming feature vector.

    Example::

        dd = DriftDetector(feature_names=["tpl_1", "error_ratio", ...])
        dd.fit(X_train)
        result = dd.check_drift(X_new)
        if result.is_drifted:
            print(result.recommendation)
    """

    def __init__(
        self,
        *,
        feature_names: list[str] | None = None,
        z_threshold: float = _DEFAULT_Z_THRESHOLD,
        min_features_drifted_ratio: float = 0.3,
    ) -> None:
        """
        Args:
            feature_names: Optional list of human-readable feature names.
                When provided, ``DriftResult.features_drifted`` uses these
                names instead of numeric indices.
            z_threshold: Absolute z-score above which a single feature is
                considered drifted.
            min_features_drifted_ratio: Fraction of features that must be
                individually drifted for the whole vector to be flagged.
        """
        self._feature_names = feature_names
        self._z_threshold = z_threshold
        self._min_drifted_ratio = min_features_drifted_ratio

        # Reference distribution parameters (set by fit / load).
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, feature_vectors: np.ndarray) -> None:
        """Compute reference distribution statistics from *feature_vectors*.

        Args:
            feature_vectors: 2-D array ``(n_samples, n_features)`` — typically
                the same data used to train the anomaly model.
        """
        if feature_vectors.ndim != 2:
            raise ValueError(
                f"feature_vectors must be 2-D, got shape {feature_vectors.shape}"
            )

        self._mean = np.mean(feature_vectors, axis=0)
        self._std = np.std(feature_vectors, axis=0)

        # Replace zero std with 1.0 to avoid division-by-zero (constant
        # features are never flagged as drifted).
        self._std[self._std == 0.0] = 1.0

        logger.info(
            "drift_detector.fit",
            n_samples=feature_vectors.shape[0],
            n_features=feature_vectors.shape[1],
        )

    # ------------------------------------------------------------------
    # Checking
    # ------------------------------------------------------------------

    def check_drift(self, feature_vector: np.ndarray) -> DriftResult:
        """Compare a single feature vector against the reference distribution.

        Args:
            feature_vector: 1-D array ``(n_features,)``.

        Returns:
            A ``DriftResult`` describing whether drift was detected.

        Raises:
            RuntimeError: If no reference distribution has been set.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError(
                "No reference distribution set. Call fit() or load() first."
            )

        vec = feature_vector.ravel()

        if vec.shape[0] != self._mean.shape[0]:
            raise ValueError(
                f"Feature vector length {vec.shape[0]} does not match "
                f"reference length {self._mean.shape[0]}"
            )

        z_scores = np.abs((vec - self._mean) / self._std)
        drifted_mask = z_scores > self._z_threshold

        drifted_indices: list[int] = list(np.where(drifted_mask)[0])
        feature_labels = self._feature_names or [
            f"feature_{i}" for i in range(len(vec))
        ]
        features_drifted = [feature_labels[i] for i in drifted_indices if i < len(feature_labels)]

        drift_score = float(np.mean(z_scores))
        n_drifted = int(drifted_mask.sum())
        n_total = len(vec)
        is_drifted = (n_drifted / n_total) >= self._min_drifted_ratio if n_total > 0 else False

        recommendation = (
            "retrain recommended"
            if is_drifted
            else "distribution within acceptable range"
        )

        result = DriftResult(
            is_drifted=is_drifted,
            drift_score=drift_score,
            features_drifted=features_drifted,
            recommendation=recommendation,
        )

        logger.debug(
            "drift_detector.check",
            drift_score=drift_score,
            n_drifted=n_drifted,
            is_drifted=is_drifted,
        )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save reference distribution statistics to *path*.

        The stats are stored as a JSON file alongside the model artifact.

        Args:
            path: Directory in which to write ``drift_reference.json``.
        """
        if self._mean is None or self._std is None:
            raise RuntimeError("No reference stats to save — call fit() first.")

        path.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
            "z_threshold": self._z_threshold,
            "min_drifted_ratio": self._min_drifted_ratio,
            "feature_names": self._feature_names,
        }
        (path / _REFERENCE_FILENAME).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.info("drift_detector.saved", path=str(path))

    def load(self, path: Path) -> None:
        """Load reference distribution statistics from *path*.

        Args:
            path: Directory containing ``drift_reference.json``.

        Raises:
            FileNotFoundError: If the reference file is missing.
        """
        ref_file = path / _REFERENCE_FILENAME
        if not ref_file.exists():
            raise FileNotFoundError(f"Drift reference not found: {ref_file}")

        payload = json.loads(ref_file.read_text(encoding="utf-8"))
        self._mean = np.array(payload["mean"], dtype=np.float64)
        self._std = np.array(payload["std"], dtype=np.float64)
        self._z_threshold = payload.get("z_threshold", _DEFAULT_Z_THRESHOLD)
        self._min_drifted_ratio = payload.get("min_drifted_ratio", 0.3)
        self._feature_names = payload.get("feature_names")

        logger.info(
            "drift_detector.loaded",
            path=str(path),
            n_features=len(self._mean),
        )
