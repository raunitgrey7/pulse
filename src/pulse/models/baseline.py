"""Classical baseline: hand-crafted features plus histogram gradient boosting.

Gradient boosting on engineered features is the strongest classical approach
for vibration diagnosis and needs no accelerator, so it anchors the
benchmark: a neural model only earns its place by beating or matching it
under tighter resource limits. scikit-learn's histogram implementation is
used because it is already a dependency and trains in seconds.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from pulse.features import FeatureExtractor
from pulse.signals.dataset import Dataset
from pulse.signals.generator import MachineSpec


@dataclass(frozen=True)
class BaselineConfig:
    """Hyperparameters of the boosted trees."""

    max_iter: int = 300
    learning_rate: float = 0.08
    max_leaf_nodes: int = 15
    l2_regularization: float = 0.1
    seed: int = 0


class FeatureBaseline:
    """Feature extraction and classifier bundled so they are saved and loaded together."""

    def __init__(self, spec: MachineSpec, config: BaselineConfig = BaselineConfig()) -> None:
        self.spec = spec
        self.config = config
        self.extractor = FeatureExtractor(spec)
        self.clf = HistGradientBoostingClassifier(
            max_iter=config.max_iter,
            learning_rate=config.learning_rate,
            max_leaf_nodes=config.max_leaf_nodes,
            l2_regularization=config.l2_regularization,
            early_stopping=False,
            random_state=config.seed,
        )

    def fit(self, dataset: Dataset) -> FeatureBaseline:
        features = self.extractor.extract_batch(dataset.x, dataset.rpm)
        self.clf.fit(features, dataset.y)
        return self

    def predict_proba(self, x: np.ndarray, rpm: np.ndarray) -> np.ndarray:
        """Class probabilities for windows of shape (batch, n_samples)."""
        features = self.extractor.extract_batch(x, rpm)
        return self.clf.predict_proba(features).astype(np.float32)

    def predict(self, x: np.ndarray, rpm: np.ndarray) -> np.ndarray:
        return self.predict_proba(x, rpm).argmax(axis=1)

    def save(self, path: Path) -> int:
        """Pickle the whole object and return the file size in bytes."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path.stat().st_size

    @classmethod
    def load(cls, path: Path) -> FeatureBaseline:
        with path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError("file does not contain a FeatureBaseline")
        return obj
