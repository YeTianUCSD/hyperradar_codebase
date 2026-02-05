# pcdet/models/hd/hd_core.py
# --------------------------------------------------------
# Hyperdimensional Computing (HDC) core module for anchor-level online adaptation.
#
# This file merges:
#   1) Encoder: feature -> hypervector (HV)
#   2) Memory: class prototypes updated with index_add_
#   3) Sampler: hard + random sampling for anchors
#   4) Updater: train-update & retrain-update (perceptron-style correction)
#
# Notes:
# - Designed to be called from AnchorHeadSingle forward() to compute HD logits.
# - Online updates should be done outside forward (e.g., in tools/online_hd_test.py),
#   using GT labels at first, to avoid side effects in model inference.
# --------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utility helpers
# -----------------------------

def _safe_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Fetch attribute or dict key from config-like objects."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2 normalization."""
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


def _hard_quantize(x: torch.Tensor) -> torch.Tensor:
    """Hard quantize to {-1, +1}. Zero becomes +1."""
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


# -----------------------------
# Config dataclass
# -----------------------------

@dataclass
class HDConfig:
    enabled: bool = True
    mode: str = "fused"              # "baseline" | "hd_only" | "fused"
    lam: float = 0.5                 # fusion lambda
    num_classes: int = 3
    feat_dim: int = 128              # mid-feature dim
    hd_dim: int = 10000
    encoder: str = "rp"              # "rp" | "idlevel" | "nonlinear"
    quantize: bool = True
    temperature: float = 1.0         # optional scaling on logits
    seed: int = 0

    # Sampler
    sample_percentage: float = 0.05  # fraction of anchors selected for update
    hard_ratio: float = 0.5          # hard portion in selected samples
    min_pos: int = 64                # minimum positive samples per batch (if available)

    # Update policy
    update_steps: int = 1            # how many times to apply update per batch
    retrain_steps: int = 1           # how many times to apply retrain correction
    normalize_every: int = 1         # normalize prototypes every K updates
    ignore_bg: bool = True           # ignore background label by default

    # Label semantics
    # In OpenPCDet anchor assigner, background could be 0 or -1 depending on head.
    # We'll treat labels <= bg_threshold as background if ignore_bg is True.
    bg_threshold: int = 0


# -----------------------------
# Encoder: feature -> HV
# -----------------------------

class HDEmbedder(nn.Module):
    """
    Encode dense features into hypervectors.

    Supported encoders:
    - rp: random projection (fixed matrix)
    - idlevel: quantize feature into levels and bind with position codes (simple torch version)
    - nonlinear: sinusoidal projection (simple torch version)
    """

    def __init__(
        self,
        feat_dim: int,
        hd_dim: int,
        encoder: str = "rp",
        quantize: bool = True,
        seed: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_levels: int = 100,
        randomness: float = 0.0,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.hd_dim = int(hd_dim)
        self.encoder = str(encoder).lower()
        self.quantize = bool(quantize)
        self.num_levels = int(num_levels)
        self.randomness = float(randomness)

        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed))

        if self.encoder == "rp":
            # Fixed random projection matrix: [feat_dim, hd_dim]
            # Use normal distribution for approximate orthogonality.
            W = torch.randn(self.feat_dim, self.hd_dim, generator=g)
            self.register_buffer("rp_weight", W, persistent=True)

        elif self.encoder == "idlevel":
            # Position codes: [feat_dim, hd_dim] (fixed random)
            pos = _hard_quantize(torch.randn(self.feat_dim, self.hd_dim, generator=g))
            self.register_buffer("pos_code", pos, persistent=True)

            # Level codes: [num_levels, hd_dim] (fixed random)
            lvl = _hard_quantize(torch.randn(self.num_levels, self.hd_dim, generator=g))
            self.register_buffer("lvl_code", lvl, persistent=True)

        elif self.encoder == "nonlinear":
            # Sinusoidal projection parameters: [feat_dim, hd_dim]
            # This is a simple alternative to torchhd Sinusoid embedding.
            freq = torch.randn(self.feat_dim, self.hd_dim, generator=g)
            phase = 2.0 * torch.pi * torch.rand(self.feat_dim, self.hd_dim, generator=g)
            self.register_buffer("nl_freq", freq, persistent=True)
            self.register_buffer("nl_phase", phase, persistent=True)

        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder}")

        # Optionally cast buffers to desired dtype/device at init
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    @torch.no_grad()
    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [N, C] float features

        Returns:
            hv: [N, D] hypervectors
        """
        assert feat.dim() == 2, f"Expected [N, C], got {tuple(feat.shape)}"
        N, C = feat.shape
        assert C == self.feat_dim, f"feat_dim mismatch: expected {self.feat_dim}, got {C}"

        if self.encoder == "rp":
            W = self.rp_weight
            if W.dtype != feat.dtype:
                W = W.to(dtype=feat.dtype)
            if W.device != feat.device:
                W = W.to(device=feat.device)
            hv = feat @ W  # [N, D]

        elif self.encoder == "idlevel":
            # Quantize each feature dimension into [0, num_levels-1]
            # Normalize feat roughly to [0, 1] by per-batch min/max to avoid config dependence.
            f_min = feat.min(dim=0, keepdim=True).values
            f_max = feat.max(dim=0, keepdim=True).values
            denom = (f_max - f_min).clamp_min(1e-6)
            f01 = (feat - f_min) / denom
            idx = torch.clamp((f01 * (self.num_levels - 1)).long(), 0, self.num_levels - 1)  # [N, C]

            pos = self.pos_code
            lvl = self.lvl_code
            if pos.device != feat.device:
                pos = pos.to(feat.device)
            if lvl.device != feat.device:
                lvl = lvl.to(feat.device)
            if pos.dtype != feat.dtype:
                pos = pos.to(feat.dtype)
            if lvl.dtype != feat.dtype:
                lvl = lvl.to(feat.dtype)

            # Bind: position_code[c] * level_code[idx[n,c]]
            # Then sum over feature dims -> [N, D]
            # This is a simple "multiset" by summation.
            hv = torch.zeros((N, self.hd_dim), device=feat.device, dtype=feat.dtype)
            # Vectorized gather: lvl[idx] -> [N, C, D]
            lvl_g = lvl.index_select(0, idx.reshape(-1)).view(N, C, self.hd_dim)
            pos_e = pos.unsqueeze(0).expand(N, C, self.hd_dim)
            bound = pos_e * lvl_g
            hv = bound.sum(dim=1)

        elif self.encoder == "nonlinear":
            freq = self.nl_freq
            phase = self.nl_phase
            if freq.device != feat.device:
                freq = freq.to(feat.device)
                phase = phase.to(feat.device)
            if freq.dtype != feat.dtype:
                freq = freq.to(feat.dtype)
                phase = phase.to(feat.dtype)

            # hv[d] = sum_c sin(freq[c,d] * feat[c] + phase[c,d])
            # This is an efficient nonlinear projection that preserves locality.
            # feat: [N, C] -> [N, C, 1]
            hv = torch.sin(feat.unsqueeze(-1) * freq.unsqueeze(0) + phase.unsqueeze(0)).sum(dim=1)

        else:
            raise RuntimeError("Unreachable")

        if self.quantize:
            hv = _hard_quantize(hv)

        return hv


# -----------------------------
# Memory: prototypes and updates
# -----------------------------

class HDMemory(nn.Module):
    """
    Class prototype memory using additive updates.

    - classify_weights: accumulator (sum of hypervectors)
    - prototypes: normalized version used for cosine-sim logits
    """

    def __init__(self, num_classes: int, hd_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_classes = int(num_classes)
        self.hd_dim = int(hd_dim)

        w = torch.zeros(self.num_classes, self.hd_dim)
        self.register_buffer("classify_weights", w, persistent=True)
        p = torch.zeros_like(w)
        self.register_buffer("prototypes", p, persistent=True)

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    @torch.no_grad()
    def reset(self):
        self.classify_weights.zero_()
        self.prototypes.zero_()

    @torch.no_grad()
    def normalize_(self):
        self.prototypes.copy_(_normalize_rows(self.classify_weights))

    @torch.no_grad()
    def add_(self, labels: torch.Tensor, hv: torch.Tensor, alpha: float = 1.0):
        """
        Add hypervectors to class accumulators using index_add_.

        Args:
            labels: [N] int64 in [0, K-1]
            hv:     [N, D]
            alpha:  update step size
        """
        assert labels.dim() == 1, "labels should be [N]"
        assert hv.dim() == 2, "hv should be [N, D]"
        assert hv.shape[0] == labels.shape[0], "N mismatch"
        assert hv.shape[1] == self.hd_dim, "D mismatch"

        if hv.dtype != self.classify_weights.dtype:
            hv = hv.to(self.classify_weights.dtype)
        if hv.device != self.classify_weights.device:
            hv = hv.to(self.classify_weights.device)
        if labels.device != self.classify_weights.device:
            labels = labels.to(self.classify_weights.device)

        if alpha != 1.0:
            hv = hv * float(alpha)

        self.classify_weights.index_add_(0, labels, hv)

    @torch.no_grad()
    def retrain_correct_(self, y_true: torch.Tensor, y_pred: torch.Tensor, hv: torch.Tensor, alpha: float = 1.0):
        """
        Perceptron-style correction:
          W[y_true] += hv
          W[y_pred] -= hv

        Args:
            y_true: [N]
            y_pred: [N]
            hv:     [N, D]
        """
        assert y_true.shape == y_pred.shape
        if alpha != 1.0:
            hv = hv * float(alpha)
        self.add_(y_true, hv, alpha=1.0)
        self.add_(y_pred, -hv, alpha=1.0)

    @torch.no_grad()
    def logits(self, hv: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Cosine-similarity logits between hv and prototypes.

        Args:
            hv: [N, D]
        """
        if hv.dtype != self.prototypes.dtype:
            hv = hv.to(self.prototypes.dtype)
        if hv.device != self.prototypes.device:
            hv = hv.to(self.prototypes.device)

        hv_n = _normalize_rows(hv)
        # [N, D] @ [D, K] -> [N, K]
        logits = hv_n @ self.prototypes.t()
        if temperature != 1.0:
            logits = logits / float(temperature)
        return logits


# -----------------------------
# Sampler: hard + random
# -----------------------------

class HDSampler:
    """
    Anchor sampler for online updates.

    We usually update memory only using positive anchors (GT assigned).
    This sampler chooses a subset:
      - hard part: smallest margin / low confidence / or mismatched predictions
      - random part: random positives for coverage

    You can pass either:
      - margin: [N] (smaller = harder)
      - or scores: [N, K] + labels: [N] to compute margin internally
    """

    @staticmethod
    @torch.no_grad()
    def sample_positive_indices(
        labels: torch.Tensor,
        sample_percentage: float,
        hard_ratio: float,
        min_pos: int = 64,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Basic positive-only random sampler (fallback).

        Args:
            labels: [N] anchor labels (pos: >=1 or >=0 depends; caller should pre-mask)
        Returns:
            selected_indices: [M]
        """
        N = labels.numel()
        M = max(int(N * float(sample_percentage)), int(min_pos))
        M = min(M, N)
        if M <= 0:
            return labels.new_empty((0,), dtype=torch.long)

        if device is None:
            device = labels.device

        perm = torch.randperm(N, device=device)
        return perm[:M]

    @staticmethod
    @torch.no_grad()
    def sample_hard_random(
        margin: torch.Tensor,
        sample_percentage: float,
        hard_ratio: float,
        min_keep: int = 64,
    ) -> torch.Tensor:
        """
        Hard + random sampling based on margin (smaller margin = harder).

        Args:
            margin: [N_pos]
        Returns:
            selected_indices (within the positive subset): [M]
        """
        assert margin.dim() == 1
        N = margin.numel()
        if N == 0:
            return margin.new_empty((0,), dtype=torch.long)

        M = max(int(N * float(sample_percentage)), int(min_keep))
        M = min(M, N)

        hard_n = int(M * float(hard_ratio))
        hard_n = min(hard_n, M)
        rand_n = M - hard_n

        # Hard: pick smallest margins
        sorted_margin, sorted_idx = torch.sort(margin, descending=False)
        hard_idx = sorted_idx[:hard_n]

        if rand_n <= 0:
            return hard_idx

        # Random from the remaining
        mask = torch.ones(N, device=margin.device, dtype=torch.bool)
        mask[hard_idx] = False
        remain_idx = torch.nonzero(mask, as_tuple=False).view(-1)

        if remain_idx.numel() == 0:
            return hard_idx

        if remain_idx.numel() <= rand_n:
            rand_idx = remain_idx
        else:
            perm = torch.randperm(remain_idx.numel(), device=margin.device)
            rand_idx = remain_idx[perm[:rand_n]]

        return torch.cat([hard_idx, rand_idx], dim=0)


# -----------------------------
# Core: glue encoder + memory + sampling + fusion
# -----------------------------

class HDCore(nn.Module):
    """
    HDC core that:
    - encodes mid features into HV
    - computes HD logits
    - provides sampling utilities for online updates
    - updates prototypes using GT (train-update) and optional correction (retrain-update)
    """

    def __init__(self, cfg: HDConfig, device=None, dtype=None):
        super().__init__()
        self.cfg = cfg

        self.embedder = HDEmbedder(
            feat_dim=cfg.feat_dim,
            hd_dim=cfg.hd_dim,
            encoder=cfg.encoder,
            quantize=cfg.quantize,
            seed=cfg.seed,
            device=device,
            dtype=dtype,
        )
        self.memory = HDMemory(
            num_classes=cfg.num_classes,
            hd_dim=cfg.hd_dim,
            device=device,
            dtype=dtype,
        )

        self._update_counter = 0

    @staticmethod
    def from_cfg(hd_cfg: Any, feat_dim: int, num_classes: int) -> "HDCore":
        """
        Build HDCore from a nested config (EasyDict / dict / object).
        You can call this from anchor_head_single.py.
        """
        enabled = bool(_safe_get(hd_cfg, "ENABLED", True))
        mode = str(_safe_get(hd_cfg, "MODE", "fused")).lower()
        lam = float(_safe_get(hd_cfg, "LAMBDA", 0.5))
        hd_dim = int(_safe_get(hd_cfg, "HD_DIM", 10000))
        encoder = str(_safe_get(hd_cfg, "ENCODER", "rp")).lower()
        quantize = bool(_safe_get(hd_cfg, "QUANTIZE", True))
        temperature = float(_safe_get(hd_cfg, "TEMPERATURE", 1.0))
        seed = int(_safe_get(hd_cfg, "SEED", 0))

        sampler_cfg = _safe_get(hd_cfg, "SAMPLER", {})
        sample_percentage = float(_safe_get(sampler_cfg, "PERCENTAGE", 0.05))
        hard_ratio = float(_safe_get(sampler_cfg, "HARD_RATIO", 0.5))
        min_pos = int(_safe_get(sampler_cfg, "MIN_POS", 64))

        update_cfg = _safe_get(hd_cfg, "UPDATE", {})
        update_steps = int(_safe_get(update_cfg, "STEPS", 1))
        retrain_steps = int(_safe_get(update_cfg, "RETRAIN_STEPS", 1))
        normalize_every = int(_safe_get(update_cfg, "NORMALIZE_EVERY", 1))
        ignore_bg = bool(_safe_get(update_cfg, "IGNORE_BG", True))
        bg_threshold = int(_safe_get(update_cfg, "BG_THRESHOLD", 0))

        cfg = HDConfig(
            enabled=enabled,
            mode=mode,
            lam=lam,
            num_classes=num_classes,
            feat_dim=feat_dim,
            hd_dim=hd_dim,
            encoder=encoder,
            quantize=quantize,
            temperature=temperature,
            seed=seed,
            sample_percentage=sample_percentage,
            hard_ratio=hard_ratio,
            min_pos=min_pos,
            update_steps=update_steps,
            retrain_steps=retrain_steps,
            normalize_every=normalize_every,
            ignore_bg=ignore_bg,
            bg_threshold=bg_threshold,
        )
        return HDCore(cfg)

    @torch.no_grad()
    def compute_hd_logits(self, feat_mid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute HD logits for ALL anchors (no sampling).

        Args:
            feat_mid: [N, C] mid features for anchors

        Returns:
            logits_hd: [N, K]
            hv:        [N, D] (quantized if enabled)
        """
        hv = self.embedder(feat_mid)
        logits_hd = self.memory.logits(hv, temperature=self.cfg.temperature)
        return logits_hd, hv

    @torch.no_grad()
    def fuse_logits(
        self,
        logits_origin: torch.Tensor,
        logits_hd: torch.Tensor,
        mode: Optional[str] = None,
        lam: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Fuse original logits with HD logits.

        mode:
          - "baseline": return origin
          - "hd_only":  return hd
          - "fused":    lam * origin + (1-lam) * hd
        """
        if mode is None:
            mode = self.cfg.mode
        mode = str(mode).lower()

        if lam is None:
            lam = self.cfg.lam
        lam = float(lam)

        if mode == "baseline":
            return logits_origin
        if mode == "hd_only":
            return logits_hd
        if mode == "fused":
            return lam * logits_origin + (1.0 - lam) * logits_hd
        raise ValueError(f"Unknown fusion mode: {mode}")

    @torch.no_grad()
    def _filter_positive(
        self,
        labels: torch.Tensor,
        feat_mid: torch.Tensor,
        logits_origin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Filter positive anchors (GT assigned) for updates.

        Returns:
            pos_feat: [N_pos, C]
            pos_labels: [N_pos]
            pos_logits_origin: [N_pos, K] or None
            pos_indices: [N_pos] indices in original anchor list
        """
        assert labels.dim() == 1, "labels must be [N]"
        N = labels.numel()
        assert feat_mid.shape[0] == N, "feat_mid and labels must align"

        if self.cfg.ignore_bg:
            # Treat labels <= bg_threshold as background (ignore)
            pos_mask = labels > int(self.cfg.bg_threshold)
        else:
            pos_mask = torch.ones_like(labels, dtype=torch.bool)

        pos_indices = torch.nonzero(pos_mask, as_tuple=False).view(-1)
        if pos_indices.numel() == 0:
            empty_feat = feat_mid.new_empty((0, feat_mid.shape[1]))
            empty_lab = labels.new_empty((0,), dtype=torch.long)
            empty_log = None if logits_origin is None else logits_origin.new_empty((0, logits_origin.shape[1]))
            return empty_feat, empty_lab, empty_log, pos_indices

        pos_feat = feat_mid[pos_indices]
        pos_labels = labels[pos_indices].long()
        pos_logits = None if logits_origin is None else logits_origin[pos_indices]
        return pos_feat, pos_labels, pos_logits, pos_indices

    @torch.no_grad()
    def _compute_margin(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute margin = (best_other - true_score).
        Smaller margin => easier, larger => harder (if wrong, margin > 0).

        We will use absolute hardness criterion:
          hardness = best_other - true_score  (higher => harder)
        For sampling by "small margin", invert sign if you want.
        Here we return hardness, and sampler picks largest hardness by sorting desc.
        """
        # logits: [N, K], labels: [N]
        N, K = logits.shape
        idx = labels.view(-1, 1)
        true_score = torch.gather(logits, 1, idx).squeeze(1)  # [N]

        # mask true class and take max over others
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        other = logits.masked_fill(mask, float("-inf"))
        best_other = other.max(dim=1).values  # [N]

        hardness = best_other - true_score
        return hardness

    @torch.no_grad()
    def sample_for_update(
        self,
        labels: torch.Tensor,
        feat_mid: torch.Tensor,
        logits_origin: Optional[torch.Tensor] = None,
        prefer_hard: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Select a subset of anchors for online memory update.

        Returns a dict containing:
            selected_anchor_indices: indices in the full anchor list [M]
            selected_pos_indices: indices within positives [M] (optional)
            selected_labels: [M]
        """
        pos_feat, pos_labels, pos_logits, pos_indices = self._filter_positive(labels, feat_mid, logits_origin)

        if pos_indices.numel() == 0:
            return {
                "selected_anchor_indices": pos_indices,
                "selected_labels": pos_labels,
            }

        # Determine how many to keep
        Npos = pos_indices.numel()
        M = max(int(Npos * float(self.cfg.sample_percentage)), int(self.cfg.min_pos))
        M = min(M, Npos)
        if M <= 0:
            return {
                "selected_anchor_indices": pos_indices.new_empty((0,), dtype=torch.long),
                "selected_labels": pos_labels.new_empty((0,), dtype=torch.long),
            }

        if (not prefer_hard) or (pos_logits is None):
            # Fallback: pure random over positives
            perm = torch.randperm(Npos, device=pos_indices.device)
            keep_pos = perm[:M]
        else:
            # Hard+random sampling based on hardness (larger = harder)
            hardness = self._compute_margin(pos_logits, pos_labels)  # [Npos]
            # We'll choose hard as top hardness (descending), plus random remaining.
            hard_n = int(M * float(self.cfg.hard_ratio))
            hard_n = min(hard_n, M)
            rand_n = M - hard_n

            # Top-hard
            _, sorted_idx = torch.sort(hardness, descending=True)
            hard_idx = sorted_idx[:hard_n]

            if rand_n <= 0:
                keep_pos = hard_idx
            else:
                mask = torch.ones(Npos, device=pos_indices.device, dtype=torch.bool)
                mask[hard_idx] = False
                remain = torch.nonzero(mask, as_tuple=False).view(-1)
                if remain.numel() == 0:
                    keep_pos = hard_idx
                elif remain.numel() <= rand_n:
                    keep_pos = torch.cat([hard_idx, remain], dim=0)
                else:
                    perm = torch.randperm(remain.numel(), device=pos_indices.device)
                    keep_pos = torch.cat([hard_idx, remain[perm[:rand_n]]], dim=0)

        selected_anchor_indices = pos_indices[keep_pos]
        selected_labels = pos_labels[keep_pos]

        return {
            "selected_anchor_indices": selected_anchor_indices,
            "selected_labels": selected_labels,
        }

    @torch.no_grad()
    def update_train(
        self,
        feat_mid: torch.Tensor,
        labels: torch.Tensor,
        selected_anchor_indices: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
    ):
        """
        Train-style update: prototype[y] += hv

        Args:
            feat_mid: [N, C] (all anchors)
            labels: [N] (anchor labels, pos are class ids)
            selected_anchor_indices: [M] indices in [0..N-1] to update; if None, update all positives.
        """
        if selected_anchor_indices is None:
            pos_feat, pos_labels, _, _ = self._filter_positive(labels, feat_mid, None)
        else:
            pos_feat = feat_mid[selected_anchor_indices]
            pos_labels = labels[selected_anchor_indices].long()

        if pos_feat.numel() == 0:
            return

        hv = self.embedder(pos_feat)
        self.memory.add_(pos_labels, hv, alpha=alpha)

        self._update_counter += 1
        if (self.cfg.normalize_every > 0) and (self._update_counter % int(self.cfg.normalize_every) == 0):
            self.memory.normalize_()

    @torch.no_grad()
    def update_retrain(
        self,
        feat_mid: torch.Tensor,
        labels: torch.Tensor,
        logits_origin: torch.Tensor,
        selected_anchor_indices: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
    ):
        """
        Retrain-style update: for misclassified samples:
            W[y_true] += hv
            W[y_pred] -= hv

        Args:
            logits_origin: [N, K] (use fused or origin logits depending on your design)
        """
        if selected_anchor_indices is None:
            pos_feat, pos_labels, pos_logits, _ = self._filter_positive(labels, feat_mid, logits_origin)
        else:
            pos_feat = feat_mid[selected_anchor_indices]
            pos_labels = labels[selected_anchor_indices].long()
            pos_logits = logits_origin[selected_anchor_indices]

        if pos_feat.numel() == 0:
            return

        hv = self.embedder(pos_feat)
        pred = pos_logits.argmax(dim=1).long()

        wrong = pred != pos_labels
        if wrong.sum().item() == 0:
            return

        y_true = pos_labels[wrong]
        y_pred = pred[wrong]
        hv_wrong = hv[wrong]

        # Apply correction multiple times if needed
        steps = max(int(self.cfg.retrain_steps), 1)
        for _ in range(steps):
            self.memory.retrain_correct_(y_true, y_pred, hv_wrong, alpha=alpha)

        self._update_counter += 1
        if (self.cfg.normalize_every > 0) and (self._update_counter % int(self.cfg.normalize_every) == 0):
            self.memory.normalize_()

    @torch.no_grad()
    def save_memory(self, path: str):
        """
        Save only the HDC state (encoder buffers + memory weights).
        """
        payload = {
            "cfg": self.cfg.__dict__,
            "embedder": self.embedder.state_dict(),
            "memory": self.memory.state_dict(),
            "_update_counter": int(self._update_counter),
        }
        torch.save(payload, path)

    @torch.no_grad()
    def load_memory(self, path: str, strict: bool = True, map_location: str = "cpu"):
        """
        Load HDC state.
        """
        payload = torch.load(path, map_location=map_location)
        if "embedder" in payload:
            self.embedder.load_state_dict(payload["embedder"], strict=strict)
        if "memory" in payload:
            self.memory.load_state_dict(payload["memory"], strict=strict)
        self._update_counter = int(payload.get("_update_counter", 0))

        # Ensure prototypes are normalized after load
        self.memory.normalize_()
