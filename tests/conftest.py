"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from crank.training import train_from_dataset

TRAINING_DATASET = Path(__file__).resolve().parents[1] / "examples" / "training_dataset.jsonl"


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train once per test session and return the model path."""
    model_path = tmp_path_factory.mktemp("models") / "crank.joblib"
    train_from_dataset(TRAINING_DATASET, model_path)
    return model_path
