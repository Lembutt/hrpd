"""Input construction: two-channel spectra and the wide-window physics feature.

All functions operate on envelope-order spectra: each sample is a
1-D vector sampled on a uniform shaft-order grid ``[0, order_max]``.
"""

from __future__ import annotations

import numpy as np


def order_grid(n_bins: int, order_max: float = 30.0) -> np.ndarray:
    """Return the uniform shaft-order grid the spectra are sampled on."""
    return np.linspace(0.0, order_max, n_bins)


def two_channel(X: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Build the two-channel input ``u(x) = [x - baseline, log(1 + x)]``.

    The first channel is the healthy-baseline residual; the second keeps
    the (log-compressed) original spectral structure.

    Args:
        X: Spectra, shape ``(n_samples, n_bins)``.
        baseline: Domain mean healthy spectrum, shape ``(n_bins,)``.

    Returns:
        Array of shape ``(n_samples, 2, n_bins)``, dtype ``float32``.
    """
    residual = X - baseline
    log_mag = np.log1p(np.clip(X, 0, None))
    return np.stack([residual, log_mag], 1).astype("float32")


def phys_features(
    X: np.ndarray,
    orders: tuple[float, float, float],
    width: float = 0.10,
    n_harmonics: int = 10,
    order_max: float = 30.0,
) -> np.ndarray:
    """Extract the wide-window harmonic feature vector from spectra.

    For each characteristic-order family ``f`` in ``orders`` (BPFO, BPFI,
    BSF) and each harmonic ``i = 1..n_harmonics``, the feature is the
    maximum of the spectrum over the relative window
    ``[(1 - width) * i * f, (1 + width) * i * f]`` intersected with the
    analysed order range. A harmonic whose window lies entirely above
    ``order_max`` (i.e. ``(1 - width) * i * f > order_max``) contributes
    zero, so the feature is defined for every bearing geometry. The
    resulting vector is RMS-normalised per sample.

    Args:
        X: Spectra, shape ``(n_samples, n_bins)``.
        orders: The (BPFO, BPFI, BSF) characteristic orders.
        width: Window half-width ``w`` (paper default ``0.10``).
        n_harmonics: Harmonics per family (paper default ``10``).
        order_max: Upper edge of the order grid.

    Returns:
        Array of shape ``(n_samples, 3 * n_harmonics)``, dtype ``float32``.
    """
    X = X.astype("float32")
    grid = order_grid(X.shape[1], order_max)
    feats = []
    for fc in orders:
        for i in range(1, n_harmonics + 1):
            o = i * fc
            mask = (grid >= (1 - width) * o) & (grid <= (1 + width) * o)
            if mask.any():
                feats.append(X[:, mask].max(1))
            else:
                feats.append(np.zeros(len(X), "float32"))
    H = np.stack(feats, 1).astype("float32")
    return H / (np.sqrt((H**2).sum(1, keepdims=True)) + 1e-8)
