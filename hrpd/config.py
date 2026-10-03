"""Configuration objects and the per-dataset registry for HRPD.

Every hyperparameter of the method lives in :class:`HRPDConfig`, with the
defaults set to the values used for the results reported in the paper
(the "complete configuration": two-stage 120+160-epoch schedule,
half-width ``w = 0.10``, SupCon weight ``0.01``, adversarial weight ``0.1``).

The :data:`DATASETS` registry maps a dataset key to its class layout and
geometry-derived characteristic orders (BPFO, BPFI, BSF), expressed as
multiples of the shaft rotation frequency.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of one benchmark dataset.

    Attributes:
        n_classes: Number of classes including the healthy class.
        class_names: Human-readable class names, index-aligned with labels
            (label ``0`` is always the healthy class).
        bpfo: Ball-pass frequency, outer race (shaft orders).
        bpfi: Ball-pass frequency, inner race (shaft orders).
        bsf: Ball-spin frequency (shaft orders). A rolling-element defect
            strikes both races, so its diagnostic line appears at ``2 * bsf``.
    """

    n_classes: int
    class_names: tuple[str, ...]
    bpfo: float
    bpfi: float
    bsf: float


#: Registry of the four benchmarks used in the paper. The characteristic
#: orders are derived from the published bearing geometry.
DATASETS: dict[str, DatasetSpec] = {
    "cwru": DatasetSpec(4, ("Normal", "IR", "Ball", "OR"), 3.585, 5.415, 2.357),
    "mfpt": DatasetSpec(3, ("Normal", "IR", "OR"), 3.2450, 4.7550, 2.5546),
    "xjtu": DatasetSpec(3, ("Normal", "IR", "OR"), 3.083, 4.917, 2.067),
    "pu": DatasetSpec(3, ("Normal", "IR", "OR"), 3.053, 4.947, 1.994),
}


@dataclass
class HRPDConfig:
    """All hyperparameters of the HRPD training and inference pipeline.

    The defaults reproduce the paper's complete configuration; the CLI
    exposes each field as a command-line flag.

    Attributes:
        epochs_pretrain: Source pre-training epochs (stage 1).
        epochs_adapt: Adaptation epochs with the adversarial term (stage 2).
        batch_size: Minibatch size per domain.
        lr: Learning rate of every AdamW optimiser.
        embed_dim: Dimension of the shared embedding space.
        width: Half-width ``w`` of the relative physics window
            ``[(1 - w) * i * f, (1 + w) * i * f]``.
        n_harmonics: Harmonics read per characteristic-order family.
        order_max: Upper edge of the shaft-order grid of the spectra.
        supcon_weight: Weight of the supervised-contrastive loss.
        supcon_tau: SupCon temperature.
        mixup_alpha: ``Beta(alpha, alpha)`` parameter of the Mixup draw.
        lambda_adv: Weight of the adversarial loss in the total objective.
        grl_coeff: Fixed gradient-reversal coefficient (not scheduled).
        temperature_init: Initial value of the learnable cosine temperature.
        device: Torch device string; ``"cuda"`` falls back to CPU if
            unavailable.
    """

    epochs_pretrain: int = 120
    epochs_adapt: int = 160
    batch_size: int = 32
    lr: float = 1e-4
    embed_dim: int = 768
    width: float = 0.10
    n_harmonics: int = 10
    order_max: float = 30.0
    supcon_weight: float = 0.01
    supcon_tau: float = 0.07
    mixup_alpha: float = 0.2
    lambda_adv: float = 0.1
    grl_coeff: float = 0.1
    temperature_init: float = 16.0
    device: str = "cuda"

    def phys_dim(self) -> int:
        """Dimension of the physics feature vector (3 families x harmonics)."""
        return 3 * self.n_harmonics
