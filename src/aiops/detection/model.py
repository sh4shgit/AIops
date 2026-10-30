"""Isolation Forest anomaly detection wrapper.

Provides training, persistence (joblib + JSON metadata), and prediction via
scikit-learn's ``IsolationForest``.  All hyper-parameters are read from
the centralised ``Settings`` object.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import structlog
from sklearn.ensemble import IsolationForest

from aiops.config import get_settings

logger = structlog.get_logger()

_MODEL_FILENAME = "isolation_forest.joblib"
_META_FILENAME = "model_meta.json"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ModelArtifact:
    """Metadata envelope for a persisted model checkpoint."""

    path: Path
    version: str
    trained_at: datetime
    parameters: dict[str, Any]
    metrics: dict[str, Any]
    training_data_hash: str


@dataclasses.dataclass(frozen=True, slots=True)
class AnomalyResult:
    """Single prediction returned by ``AnomalyDetector.predict``."""

    score: float
    is_anomaly: bool
    model_version: str


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class AnomalyDetector:
    """Train, persist, and serve an ``IsolationForest`` model.

    Example::

        detector = AnomalyDetector()
        artifact = detector.train(X_train)
        result   = detector.predict(X_test[0])
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._contamination: float = settings.isolation_forest_contamination
        self._n_estimators: int = settings.isolation_forest_n_estimators
        self._artifact_dir: Path = Path(settings.model_artifact_dir)

        self._model: IsolationForest | None = None
        self._version: str = "unversioned"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        feature_vectors: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> ModelArtifact:
        """Fit a new ``IsolationForest`` on *feature_vectors*.

        Args:
            feature_vectors: 2-D array of shape ``(n_samples, n_features)``.
            metadata: Optional extra information stored alongside the model.

        Returns:
            A ``ModelArtifact`` describing the saved checkpoint.
        """
        if feature_vectors.ndim != 2:
            raise ValueError(
                f"feature_vectors must be 2-D, got shape {feature_vectors.shape}"
            )

        params: dict[str, Any] = {
            "n_estimators": self._n_estimators,
            "contamination": self._contamination,
            "random_state": 42,
            "n_jobs": -1,
        }
        model = IsolationForest(**params)
        model.fit(feature_vectors)

        self._model = model

        # Derive a version tag from timestamp.
        trained_at = datetime.now(tz=timezone.utc)
        self._version = trained_at.strftime("%Y%m%dT%H%M%SZ")

        # Compute training-data hash for reproducibility tracking.
        data_hash = hashlib.sha256(feature_vectors.tobytes()).hexdigest()[:16]

        # Persist to disk.
        save_dir = self._artifact_dir / self._version
        self.save(save_dir, metadata=metadata or {})

        # Basic metrics on training scores.
        train_scores = model.decision_function(feature_vectors)
        metrics: dict[str, Any] = {
            "n_samples": int(feature_vectors.shape[0]),
            "n_features": int(feature_vectors.shape[1]),
            "mean_score": float(np.mean(train_scores)),
            "std_score": float(np.std(train_scores)),
            "min_score": float(np.min(train_scores)),
            "max_score": float(np.max(train_scores)),
        }

        artifact = ModelArtifact(
            path=save_dir,
            version=self._version,
            trained_at=trained_at,
            parameters=params,
            metrics=metrics,
            training_data_hash=data_hash,
        )

        logger.info(
            "anomaly_detector.trained",
            version=self._version,
            n_samples=metrics["n_samples"],
            n_features=metrics["n_features"],
            data_hash=data_hash,
        )
        return artifact

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, feature_vector: np.ndarray) -> AnomalyResult:
        """Score a single feature vector and return an ``AnomalyResult``.

        The Isolation Forest ``decision_function`` returns negative values for
        anomalies and positive values for normal observations.  We expose the
        raw score and a boolean convenience flag.

        Args:
            feature_vector: 1-D array of shape ``(n_features,)``.

        Returns:
            An ``AnomalyResult`` with the raw score and anomaly flag.

        Raises:
            RuntimeError: If no model has been trained or loaded yet.
        """
        if self._model is None:
            raise RuntimeError(
                "No model loaded. Call train() or load() before predict()."
            )

        vec = feature_vector.reshape(1, -1)
        score: float = float(self._model.decision_function(vec)[0])
        label: int = int(self._model.predict(vec)[0])
        is_anomaly = label == -1

        logger.debug(
            "anomaly_detector.predict",
            score=score,
            is_anomaly=is_anomaly,
            version=self._version,
        )
        return AnomalyResult(
            score=score,
            is_anomaly=is_anomaly,
            model_version=self._version,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path, metadata: dict[str, Any] | None = None) -> None:
        """Persist the current model and metadata to *path*.

        Creates the directory if it does not exist.  Two files are written:

        * ``isolation_forest.joblib`` — the sklearn model.
        * ``model_meta.json`` — human-readable metadata.
        """
        if self._model is None:
            raise RuntimeError("No model to save — call train() first.")

        path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self._model, path / _MODEL_FILENAME)

        meta: dict[str, Any] = {
            "version": self._version,
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
            "parameters": {
                "n_estimators": self._n_estimators,
                "contamination": self._contamination,
            },
            **(metadata or {}),
        }
        (path / _META_FILENAME).write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )

        logger.info("anomaly_detector.saved", path=str(path))

    def load(self, path: Path) -> None:
        """Load a previously saved model from *path*.

        Args:
            path: Directory containing ``isolation_forest.joblib`` and
                ``model_meta.json``.

        Raises:
            FileNotFoundError: If the expected files are missing.
        """
        model_file = path / _MODEL_FILENAME
        meta_file = path / _META_FILENAME

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self._model = joblib.load(model_file)

        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            self._version = meta.get("version", "unknown")
        else:
            self._version = "unknown"

        logger.info("anomaly_detector.loaded", path=str(path), version=self._version)
