"""Neural components of HRPD.

The model is a fusion of two branches in a shared embedding space:

* :class:`Encoder` -- a dilated 1-D convolutional network over the
  two-channel residual input;
* :class:`PhysMLP` -- a small MLP over the wide-window harmonic feature.

The two branch embeddings are added; a :class:`PrototypeClassifier`
produces scaled-cosine logits against one learnable prototype per class,
and a :class:`Discriminator` behind a gradient-reversal layer drives the
healthy-class-restricted adversarial alignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Dilated 1-D convolutional encoder over the two-channel input.

    Four blocks; block ``i`` applies two kernel-3 convolutions with
    dilation ``2**(i-1)`` (length-preserving padding, no bias), each
    followed by BatchNorm and ReLU, then halves the length by max-pooling.
    Channel width doubles per block (128 -> 256 -> 512 -> 1024). The final
    map is globally average-pooled over the order axis and linearly
    projected to the embedding dimension.

    Args:
        in_channels: Input channels (2: residual + log-magnitude).
        embed_dim: Output embedding dimension.
        base_channels: Channels of the first block.
        n_blocks: Number of blocks.
    """

    def __init__(
        self,
        in_channels: int = 2,
        embed_dim: int = 768,
        base_channels: int = 128,
        n_blocks: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ic = in_channels
        for i in range(n_blocks):
            oc = base_channels * 2**i
            d = 2**i
            layers += [
                nn.Conv1d(ic, oc, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm1d(oc),
                nn.ReLU(),
                nn.Conv1d(oc, oc, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm1d(oc),
                nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            ic = oc
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(ic, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, 2, n_bins)`` input to ``(batch, embed_dim)``."""
        return self.head(self.pool(self.tcn(x)).flatten(1))


class PhysMLP(nn.Module):
    """Small MLP mapping the physics feature to the embedding dimension.

    Args:
        in_dim: Physics feature dimension (30 for 3 families x 10 harmonics).
        embed_dim: Output embedding dimension.
    """

    def __init__(self, in_dim: int = 30, embed_dim: int = 768) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, in_dim)`` features to ``(batch, embed_dim)``."""
        return self.proj(self.net(p))


class _GradReverse(torch.autograd.Function):
    """Identity forward; negated, scaled gradient backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, coeff: float) -> torch.Tensor:  # noqa: D102
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):  # noqa: D102
        return -ctx.coeff * grad, None


def grad_reverse(x: torch.Tensor, coeff: float) -> torch.Tensor:
    """Gradient-reversal layer with a fixed (unscheduled) coefficient."""
    return _GradReverse.apply(x, coeff)


class Discriminator(nn.Module):
    """Domain discriminator: an MLP ``embed_dim -> 128 -> 64 -> 1``.

    Trained with binary cross-entropy to separate source-healthy
    embeddings (label 0) from target embeddings (label 1).
    """

    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(batch,)``."""
        return self.net(z).reshape(-1)


class PrototypeClassifier(nn.Module):
    """Cosine-prototype classifier with a learnable scalar temperature.

    Holds one learnable prototype per class; the logit for class ``c`` is
    the cosine similarity between the (L2-normalised) embedding and
    prototype, multiplied by ``|temperature|``.

    Args:
        n_classes: Number of classes (healthy included).
        embed_dim: Embedding dimension.
        temperature_init: Initial temperature value.
    """

    def __init__(
        self, n_classes: int, embed_dim: int = 768, temperature_init: float = 16.0
    ) -> None:
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(n_classes, embed_dim))
        self.temperature = nn.Parameter(torch.tensor(temperature_init))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return scaled-cosine logits of shape ``(batch, n_classes)``."""
        cos = F.normalize(z, dim=1) @ F.normalize(self.prototypes, dim=1).T
        return cos * self.temperature.abs()

    @torch.no_grad()
    def shift(self, offset: torch.Tensor) -> None:
        """Translate every prototype by ``offset`` (the prototype shift)."""
        self.prototypes += offset


class HRPDModel(nn.Module):
    """The full HRPD embedding: additive fusion of the two branches.

    ``z = encoder(u(x)) + fuse(phys_mlp(phi(x)))`` where ``fuse`` is a
    learned linear map. The single fused embedding enters every loss,
    the discriminator, the prototype shift, and inference.

    Args:
        n_classes: Number of classes.
        phys_dim: Physics feature dimension.
        embed_dim: Embedding dimension.
        temperature_init: Initial cosine temperature.
    """

    def __init__(
        self,
        n_classes: int,
        phys_dim: int = 30,
        embed_dim: int = 768,
        temperature_init: float = 16.0,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(2, embed_dim)
        self.phys = PhysMLP(phys_dim, embed_dim)
        self.fuse = nn.Linear(embed_dim, embed_dim)
        self.classifier = PrototypeClassifier(n_classes, embed_dim, temperature_init)
        self.discriminator = Discriminator(embed_dim)

    def embed(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Fused embedding of a two-channel input and its physics feature."""
        return self.encoder(x) + self.fuse(self.phys(p))

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Class logits for a batch."""
        return self.classifier(self.embed(x, p))
