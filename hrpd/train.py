"""HRPD training and inference.

1. **Input construction.** Domain healthy baselines are estimated once
   (source: mean of the labelled synthetic-healthy spectra; target: mean
   of the target-training healthy spectra) and frozen. Every spectrum is
   turned into the two-channel input ``[x - baseline, log(1 + x)]`` and
   the wide-window physics feature.
2. **Stage 1 -- source pre-training.** Mixup cross-entropy on the cosine
   logits plus a weighted SupCon term, on the labelled synthetic source.
3. **Stage 2 -- adaptation.** The same objective plus the conditional
   adversarial term: a domain discriminator behind a gradient-reversal
   layer sees only source-*healthy* versus target embeddings, so
   alignment acts on the healthy class alone.
4. **Post-hoc calibration**, applied once, in this order: the prototype
   shift (measured with the encoder in evaluation mode, i.e. under source
   BatchNorm statistics), then AdaBN (one gradient-free pass over the
   target-training healthy data with training-mode BatchNorm).
5. **Inference.** Every quantity is frozen; each held-out sample is scored
   independently by the scaled-cosine logits against the shifted
   prototypes.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from .config import DatasetSpec, HRPDConfig
from .data import Base5Data
from .features import phys_features, two_channel
from .losses import mixup, supcon_loss
from .models import HRPDModel, grad_reverse


def _resolve_device(requested: str) -> torch.device:
    """Return the requested device, falling back to CPU without CUDA."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def train_hrpd(
    data: Base5Data,
    spec: DatasetSpec,
    cfg: HRPDConfig,
    seed: int = 0,
) -> tuple[np.ndarray, HRPDModel]:
    """Train HRPD on one dataset/seed and score the held-out test set.

    Args:
        data: Loaded dataset arrays (see :class:`hrpd.data.Base5Data`).
        spec: Dataset spec (class count and characteristic orders).
        cfg: Hyperparameters.
        seed: Random seed for model initialisation and batching.

    Returns:
        ``(probs, model)`` where ``probs`` has shape
        ``(n_test, n_classes)`` (softmax of the cosine logits) and
        ``model`` is the trained, frozen :class:`HRPDModel`.
    """
    device = _resolve_device(cfg.device)
    orders = (spec.bpfo, spec.bpfi, spec.bsf)

    # --- frozen healthy baselines and constructed inputs -----------------
    source_baseline = data.Xs[data.ys == 0].mean(0)
    target_baseline = data.Xtt.mean(0)
    Xs_c = two_channel(data.Xs, source_baseline)
    Xtt_c = two_channel(data.Xtt, target_baseline)
    Xte_c = two_channel(data.Xte, target_baseline)
    PHs = phys_features(data.Xs, orders, cfg.width, cfg.n_harmonics, cfg.order_max)
    PHtt = phys_features(data.Xtt, orders, cfg.width, cfg.n_harmonics, cfg.order_max)
    PHte = phys_features(data.Xte, orders, cfg.width, cfg.n_harmonics, cfg.order_max)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = HRPDModel(
        spec.n_classes, cfg.phys_dim(), cfg.embed_dim, cfg.temperature_init
    ).to(device)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    encoder_params = (
        list(model.encoder.parameters())
        + list(model.phys.parameters())
        + list(model.fuse.parameters())
    )
    opt_encoder = torch.optim.AdamW(encoder_params, cfg.lr)
    opt_protos = torch.optim.AdamW(model.classifier.parameters(), cfg.lr)
    opt_disc = torch.optim.AdamW(model.discriminator.parameters(), cfg.lr)

    Xs_t = torch.tensor(Xs_c)
    ys_t = torch.tensor(data.ys.astype("int64"))
    PHs_t = torch.tensor(PHs)
    Xtt_t = torch.tensor(Xtt_c)
    PHtt_t = torch.tensor(PHtt)
    n_source, n_target = len(Xs_t), len(Xtt_t)
    B = cfg.batch_size

    def zero_grad() -> None:
        opt_encoder.zero_grad()
        opt_protos.zero_grad()
        opt_disc.zero_grad()

    # --- stage 1: source pre-training ------------------------------------
    for _ in range(cfg.epochs_pretrain):
        model.train()
        perm = np.random.permutation(n_source)
        for i in range(0, n_source - B + 1, B):
            idx = perm[i : i + B]
            x = Xs_t[idx].to(device)
            ph = PHs_t[idx].to(device)
            y = ys_t[idx].to(device)
            xm, pm, ya, yb, lam = mixup(x, ph, y, cfg.mixup_alpha)
            logits = model(xm, pm)
            loss = (
                lam * ce(logits, ya)
                + (1 - lam) * ce(logits, yb)
                + cfg.supcon_weight * supcon_loss(model.embed(x, ph), y, cfg.supcon_tau)
            )
            loss.backward()
            opt_encoder.step()
            opt_protos.step()
            zero_grad()

    # --- stage 2: adaptation ---------------------------------------------
    for _ in range(cfg.epochs_adapt):
        model.train()
        perm = np.random.permutation(n_source)
        perm_t = np.random.permutation(n_target)
        ti = 0
        for i in range(0, n_source - B + 1, B):
            idx = perm[i : i + B]
            x = Xs_t[idx].to(device)
            ph = PHs_t[idx].to(device)
            y = ys_t[idx].to(device)
            idx_t = perm_t[np.arange(ti, ti + B) % n_target]
            ti += B
            xt = Xtt_t[idx_t].to(device)
            pt = PHtt_t[idx_t].to(device)

            xm, pm, ya, yb, lam = mixup(x, ph, y, cfg.mixup_alpha)
            zs_mix = model.embed(xm, pm)
            zt = model.embed(xt, pt)
            logits = model.classifier(zs_mix)
            loss_cls = lam * ce(logits, ya) + (1 - lam) * ce(logits, yb)

            healthy = y == 0
            if healthy.any():
                d_src = model.discriminator(
                    grad_reverse(model.embed(x[healthy], ph[healthy]), cfg.grl_coeff)
                )
                d_tgt = model.discriminator(grad_reverse(zt, cfg.grl_coeff))
                loss_adv = bce(d_src, torch.zeros_like(d_src)) + bce(
                    d_tgt, torch.ones_like(d_tgt)
                )
            else:
                loss_adv = torch.tensor(0.0, device=device)

            loss = (
                loss_cls
                + cfg.lambda_adv * loss_adv
                + cfg.supcon_weight * supcon_loss(model.embed(x, ph), y, cfg.supcon_tau)
            )
            loss.backward()
            opt_encoder.step()
            opt_protos.step()
            opt_disc.step()
            zero_grad()

    # --- post-hoc step (i): prototype shift (source BatchNorm stats) -----
    model.eval()

    @torch.no_grad()
    def mean_embedding(Xc: np.ndarray, PH: np.ndarray) -> torch.Tensor:
        chunks = []
        for i in range(0, len(Xc), B):
            chunks.append(
                model.embed(
                    torch.tensor(Xc[i : i + B]).to(device),
                    torch.tensor(PH[i : i + B]).to(device),
                ).cpu()
            )
        return torch.cat(chunks).mean(0)

    target_mean = mean_embedding(Xtt_c, PHtt)
    source_healthy_mean = mean_embedding(
        Xs_c[data.ys == 0], PHs[data.ys == 0]
    )
    model.classifier.shift((target_mean - source_healthy_mean).to(device))

    # --- post-hoc step (ii): AdaBN over target-training healthy data -----
    model.train()
    with torch.no_grad():
        for i in range(0, len(Xtt_c), B):
            model.embed(
                torch.tensor(Xtt_c[i : i + B]).to(device),
                torch.tensor(PHtt[i : i + B]).to(device),
            )

    # --- inference: everything frozen ------------------------------------
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(Xte_c), 512):
            logits = model(
                torch.tensor(Xte_c[i : i + 512]).to(device),
                torch.tensor(PHte[i : i + 512]).to(device),
            )
            probs.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(probs), model
