"""Training objectives: supervised contrastive loss and Mixup."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def supcon_loss(z: torch.Tensor, y: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al., 2020), ``L_out`` variant.

    Embeddings are L2-normalised; for each anchor, all same-class samples
    in the batch are positives. An anchor with no positives contributes
    zero (via the ``max(|P(i)|, 1)`` guard and the empty inner sum), and
    the mean is taken over all anchors regardless.

    Args:
        z: Embeddings, shape ``(batch, dim)``.
        y: Integer class labels, shape ``(batch,)``.
        tau: Temperature.

    Returns:
        Scalar loss tensor.
    """
    z = F.normalize(z, dim=1)
    sim = z @ z.T / tau
    labels = y.unsqueeze(1)
    pos = (labels == labels.T).float()
    pos.fill_diagonal_(0)
    sim = sim.masked_fill(torch.eye(len(z), device=z.device).bool(), -1e9)
    log_prob = sim - torch.logsumexp(sim, 1, keepdim=True)
    n_pos = pos.sum(1).clamp(min=1)
    return (-(pos * log_prob).sum(1) / n_pos).mean()


def mixup(
    x: torch.Tensor,
    p: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    rng: np.random.RandomState | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup (Zhang et al., 2018) of the input and its physics feature.

    A single coefficient ``lambda ~ Beta(alpha, alpha)`` is shared by the
    two-channel input and the physics feature so the pair stays consistent.
    Mixing acts on the constructed channels (after residual subtraction and
    the logarithm), not on the raw spectrum.

    Args:
        x: Two-channel inputs, shape ``(batch, 2, n_bins)``.
        p: Physics features, shape ``(batch, phys_dim)``.
        y: Integer labels, shape ``(batch,)``.
        alpha: Beta-distribution parameter.
        rng: Optional NumPy RNG (for reproducibility in tests).

    Returns:
        ``(x_mix, p_mix, y_a, y_b, lam)`` where the classification loss is
        ``lam * CE(logits, y_a) + (1 - lam) * CE(logits, y_b)``.
    """
    draw = (rng or np.random).beta(alpha, alpha)
    lam = float(draw)
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], lam * p + (1 - lam) * p[perm], y, y[perm], lam
