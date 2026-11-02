
"""Utilities for dataset identification and experiment reproducibility."""

import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn


def fingerprint_file(file_path: str | Path) -> str:
    """Return the SHA-256 fingerprint of a file's exact bytes."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, received: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def environment_metadata() -> dict[str, str]:
    """Capture key software versions for a research experiment."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
    }


def build_experiment_metadata(
    dataset_path: str | Path,
    experiment_name: str,
    random_seed: int,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a reproducibility record without running an experiment."""
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        raise ValueError("Experiment name must be a non-empty string.")

    if not isinstance(random_seed, int) or isinstance(random_seed, bool):
        raise TypeError("Random seed must be an integer.")

    if parameters is not None and not isinstance(parameters, dict):
        raise TypeError("Parameters must be a dictionary or None.")

    path = Path(dataset_path)

    return {
        "experiment_name": experiment_name.strip(),
        "dataset_path": str(path),
        "dataset_sha256": fingerprint_file(path),
        "random_seed": random_seed,
        "parameters": parameters.copy() if parameters is not None else {},
        "environment": environment_metadata(),
    }
