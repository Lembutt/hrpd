"""Loading and validation of the packaged benchmark data.

Each dataset directory (``<data_dir>/<name>_base5_data``) contains, for
five generator seeds ``s = 0..4``:

* ``Xs_s{s}.npy`` -- synthetic source spectra, ``(n_source, n_bins)``;
* ``ys_s{s}.npy`` -- source labels (0 = healthy);

and, shared by all seeds:

* ``Xtt.npy`` -- target-training healthy spectra (real);
* ``Xte.npy`` -- held-out test spectra (real healthy + real faults);
* ``yte.npy`` -- test labels.

Only the synthetic source varies across seeds; the real splits are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DATASETS


@dataclass
class Base5Data:
    """One dataset at one generator seed, ready for training.

    Attributes:
        Xs: Synthetic source spectra, ``(n_source, n_bins)``.
        ys: Source integer labels, ``(n_source,)``.
        Xtt: Target-training healthy spectra, ``(n_tt, n_bins)``.
        Xte: Test spectra, ``(n_te, n_bins)``.
        yte: Test integer labels, ``(n_te,)``.
    """

    Xs: np.ndarray
    ys: np.ndarray
    Xtt: np.ndarray
    Xte: np.ndarray
    yte: np.ndarray


def dataset_dir(data_dir: str | Path, name: str) -> Path:
    """Return the directory of dataset ``name`` under ``data_dir``."""
    return Path(data_dir) / f"{name}_base5_data"


def load_base5(data_dir: str | Path, name: str, seed: int) -> Base5Data:
    """Load one benchmark dataset at one generator seed.

    Args:
        data_dir: Root data directory of the release.
        name: Dataset key (``cwru``, ``mfpt``, ``xjtu``, ``pu``).
        seed: Generator seed in ``0..4``.

    Returns:
        A validated :class:`Base5Data`.

    Raises:
        KeyError: Unknown dataset name.
        FileNotFoundError: A required array file is missing.
        ValueError: Array shapes or labels are inconsistent.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; expected one of {sorted(DATASETS)}")
    spec = DATASETS[name]
    d = dataset_dir(data_dir, name)
    paths = {
        "Xs": d / f"Xs_s{seed}.npy",
        "ys": d / f"ys_s{seed}.npy",
        "Xtt": d / "Xtt.npy",
        "Xte": d / "Xte.npy",
        "yte": d / "yte.npy",
    }
    for key, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"missing {key} array: {p}")
    arrays = {k: np.load(p) for k, p in paths.items()}
    data = Base5Data(**arrays)

    n_bins = data.Xs.shape[1]
    for mat_name in ("Xtt", "Xte"):
        mat = getattr(data, mat_name)
        if mat.ndim != 2 or mat.shape[1] != n_bins:
            raise ValueError(
                f"{mat_name} has shape {mat.shape}; expected (*, {n_bins})"
            )
    if len(data.ys) != len(data.Xs):
        raise ValueError("Xs and ys length mismatch")
    if len(data.yte) != len(data.Xte):
        raise ValueError("Xte and yte length mismatch")
    for labels, label_name in ((data.ys, "ys"), (data.yte, "yte")):
        if labels.min() < 0 or labels.max() >= spec.n_classes:
            raise ValueError(
                f"{label_name} contains labels outside 0..{spec.n_classes - 1}"
            )
    return data
