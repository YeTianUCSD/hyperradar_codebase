
# Hyperdimensional Computing (HDC) core module for anchor-level online adaptation.
#
# TorchHD version:
# - Encoder uses torchhd.embeddings (Projection / Level / Random / Sinusoid)
# - Binding and multiset use torchhd.functional (bind / multiset / hard_quantize)
# - Supports chunked encoding / logits to prevent OOM when N is large.
# --------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from torchhd import functional
from torchhd import embeddings


# -----------------------------
# Utility helpers
# -----------------------------

def _hard_quantize_inplace(x: torch.Tensor) -> torch.Tensor:
    """
    In-place quantize to {-1, +1}. Zeros become +1.
    Much lower peak memory than torchhd.functional.hard_quantize (torch.where allocates).
    """
    # sign_ gives {-1,0,+1}
    x.sign_()
    # 0 -> +1
    x.masked_fill_(x == 0, 1)
    return x


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


# -----------------------------
# Config dataclass
# -----------------------------

@dataclass
class HDConfig:
    enabled: bool = True
    mode: str = "fused"              # "baseline" | "hd_only" | "fused"
    lam: float = 0.5                 # fusion lambda
    num_classes: int = 3

    feat_dim: int = 128              # mid-feature dim (input feature dim)
    hd_dim: int = 10000              # hypervector dim

    encoder: str = "rp"              # "rp" | "idlevel" | "nonlinear"
    quantize: bool = True
    temperature: float = 1.0
    seed: int = 0

    # Sampler
    sample_percentage: float = 0.05
    hard_ratio: float = 0.5
    min_pos: int = 64

    # Update policy
    update_steps: int = 1
    retrain_steps: int = 1
    normalize_every: int = 1
    ignore_bg: bool = True
    bg_threshold: int = 0

    # Encoder-specific (TorchHD)
    num_levels: int = 100
    randomness: float = 0.0

    # Memory build / inference safety
    encode_chunk: int = 8192         # chunk size for encoding anchors
    logits_chunk: int = 8192         # chunk size for computing logits from hv
    anchor_id_scale: float = 0.0     # >0 enables anchor-aware feature context injection
    bg_enabled: bool = True          # enable background prototype suppression
    bg_margin_scale: float = 1.0     # fg_logit <- fg_logit - scale * bg_logit
    bg_sample_ratio: float = 0.25    # used by memory build to sample bg anchors
    bg_min_per_anchor: int = 32      # minimum sampled bg anchors per anchor type per batch


# -----------------------------
# Encoder: feature -> HV (TorchHD)
# -----------------------------

class HDEmbedder(nn.Module):
    """
    Encode dense features into hypervectors using TorchHD.

    Supported encoders:
    - rp:        embeddings.Projection(feat_dim -> hd_dim)
    - idlevel:   Level codes + Position codes + bind + multiset
    - nonlinear: embeddings.Sinusoid(feat_dim -> hd_dim)

    IMPORTANT:
    - Do NOT manually touch Projection.weight; its internal layout differs by torchhd version.
      For determinism, seed torch before constructing embeddings.
    - Do NOT move modules with .to() inside forward; model.cuda() should move everything.
    """

    def __init__(
        self,
        feat_dim: int,
        hd_dim: int,
        encoder: str = "rp",
        quantize: bool = True,
        seed: int = 0,
        num_levels: int = 100,
        randomness: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.hd_dim = int(hd_dim)
        self.encoder = str(encoder).lower()
        self.quantize = bool(quantize)

        self.seed = int(seed)
        self.num_levels = int(num_levels)
        self.randomness = float(randomness)

        # Make init deterministic (best-effort)
        torch.manual_seed(self.seed)

        if self.encoder == "rp":
            # TorchHD Projection: internal layout differs across versions -> do NOT override weight.
            self.projection = embeddings.Projection(self.feat_dim, self.hd_dim)
            for p in self.projection.parameters():
                p.requires_grad_(False)

        elif self.encoder == "idlevel":
            # Levels: [num_levels, D], Position: [feat_dim, D]
            self.value = embeddings.Level(self.num_levels, self.hd_dim, randomness=self.randomness)
            self.position = embeddings.Random(self.feat_dim, self.hd_dim)
            for p in self.value.parameters():
                p.requires_grad_(False)
            for p in self.position.parameters():
                p.requires_grad_(False)

        elif self.encoder == "nonlinear":
            self.nonlinear = embeddings.Sinusoid(self.feat_dim, self.hd_dim)
            for p in self.nonlinear.parameters():
                p.requires_grad_(False)

        else:
            raise ValueError(f"Unsupported encoder type: {self.encoder}")

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def _encode_rp(self, feat: torch.Tensor) -> torch.Tensor:
        # Projection expects float tensors; keep stability with fp32 when needed.
        if feat.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            feat = feat.float()
        # Assume embedder moved with model.cuda()
        return self.projection(feat)

    def _encode_idlevel(self, feat: torch.Tensor) -> torch.Tensor:
        """
        idlevel encoding:
        - Normalize each feature dim to [0,1] per batch
        - Quantize into level indices [0..num_levels-1]
        - Look up value hypervectors: [N, C, D]
        - Bind with position hypervectors: [C, D]
        - Multiset over C => [N, D]
        """
        assert feat.dim() == 2
        N, C = feat.shape
        assert C == self.feat_dim, f"feat_dim mismatch: expected {self.feat_dim}, got {C}"

        if feat.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            feat = feat.float()

        # Normalize to [0, 1] per dimension (per batch)
        f_min = feat.min(dim=0, keepdim=True).values
        f_max = feat.max(dim=0, keepdim=True).values
        denom = (f_max - f_min).clamp_min(1e-6)
        f01 = (feat - f_min) / denom

        # Discretize to level indices
        idx = torch.clamp((f01 * (self.num_levels - 1)).long(), 0, self.num_levels - 1)  # [N, C]

        # Use weights directly; assume module already on correct device
        val_w = self.value.weight
        pos_w = self.position.weight

        # If device mismatch, something is wrong with model.cuda(); move feat to module device once
        if val_w.device != feat.device:
            feat = feat.to(val_w.device)
            idx = idx.to(val_w.device)
        if pos_w.device != val_w.device:
            # Extremely unlikely; keep consistent
            pos_w = pos_w.to(val_w.device)

        # Gather: val_w[idx] -> [N, C, D]
        v = val_w.index_select(0, idx.reshape(-1)).view(N, C, self.hd_dim)
        # Expand pos: [C, D] -> [N, C, D]
        p = pos_w.unsqueeze(0).expand(N, C, self.hd_dim)

        # Bind + multiset
        bound = functional.bind(p, v)          # [N, C, D]
        hv = functional.multiset(bound)        # [N, D]
        return hv

    def _encode_nonlinear(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            feat = feat.float()
        return self.nonlinear(feat)  # [N, D]

    def forward(self, feat: torch.Tensor, quantize: Optional[bool] = None) -> torch.Tensor:
        """
        Args:
            feat: [N, C] float features
        Returns:
            hv:  [N, D] hypervectors (optionally hard-quantized)
        """
        assert feat.dim() == 2, f"Expected [N, C], got {tuple(feat.shape)}"
        N, C = feat.shape
        assert C == self.feat_dim, f"feat_dim mismatch: expected {self.feat_dim}, got {C}"

        if self.encoder == "rp":
            hv = self._encode_rp(feat)
        elif self.encoder == "idlevel":
            hv = self._encode_idlevel(feat)
        elif self.encoder == "nonlinear":
            hv = self._encode_nonlinear(feat)
        else:
            raise RuntimeError("Unreachable")

        do_q = self.quantize if quantize is None else bool(quantize)
        if do_q:
            _hard_quantize_inplace(hv)
        return hv

    def forward_chunked(self, feat: torch.Tensor, chunk: int = 8192, quantize: Optional[bool] = None) -> torch.Tensor:
        """
        Chunked encoding to prevent OOM.
        """
        if chunk is None or chunk <= 0 or feat.shape[0] <= chunk:
            return self.forward(feat, quantize=quantize)

        outs = []
        N = feat.shape[0]
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            outs.append(self.forward(feat[s:e], quantize=quantize))
        return torch.cat(outs, dim=0)


# -----------------------------
# Memory: prototypes and updates
# -----------------------------

class HDMemory(nn.Module):
    """
    Class prototype memory using additive updates.
    - classify_weights/prototypes: foreground class memory
    - bg_weight/bg_prototype: background memory
    """

    def __init__(self, num_classes: int, hd_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_classes = int(num_classes)
        self.hd_dim = int(hd_dim)

        w = torch.zeros(self.num_classes, self.hd_dim, dtype=torch.float32)
        self.register_buffer("classify_weights", w, persistent=True)
        p = torch.zeros_like(w)
        self.register_buffer("prototypes", p, persistent=True)
        self.register_buffer("bg_weight", torch.zeros(self.hd_dim, dtype=torch.float32), persistent=True)
        self.register_buffer("bg_prototype", torch.zeros(self.hd_dim, dtype=torch.float32), persistent=True)

        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    @torch.no_grad()
    def reset(self):
        self.classify_weights.zero_()
        self.prototypes.zero_()
        self.bg_weight.zero_()
        self.bg_prototype.zero_()

    @torch.no_grad()
    def normalize_(self):
        self.prototypes.copy_(_normalize_rows(self.classify_weights))
        bg_norm = self.bg_weight.norm().clamp_min(1e-12)
        self.bg_prototype.copy_(self.bg_weight / bg_norm)

    @torch.no_grad()
    def add_(self, labels: torch.Tensor, hv: torch.Tensor, alpha: float = 1.0):
        """
        Add hypervectors to class accumulators using index_add_.
        labels: [N] in [0..K-1]
        hv:     [N, D]
        """
        assert labels.dim() == 1
        assert hv.dim() == 2
        assert hv.shape[0] == labels.shape[0]
        assert hv.shape[1] == self.hd_dim

        dev = self.classify_weights.device
        if labels.device != dev:
            labels = labels.to(dev)
        if hv.device != dev:
            hv = hv.to(dev)
        if hv.dtype != self.classify_weights.dtype:
            hv = hv.to(self.classify_weights.dtype)

        if alpha != 1.0:
            hv = hv * float(alpha)

        self.classify_weights.index_add_(0, labels, hv)

    @torch.no_grad()
    def add_bg_(self, hv: torch.Tensor, alpha: float = 1.0):
        """
        Add hypervectors into background accumulator.
        hv: [N, D]
        """
        assert hv.dim() == 2 and hv.shape[1] == self.hd_dim
        dev = self.bg_weight.device
        if hv.device != dev:
            hv = hv.to(dev)
        if hv.dtype != self.bg_weight.dtype:
            hv = hv.to(self.bg_weight.dtype)
        if alpha != 1.0:
            hv = hv * float(alpha)
        self.bg_weight.add_(hv.sum(dim=0))

    @torch.no_grad()
    def retrain_correct_(self, y_true: torch.Tensor, y_pred: torch.Tensor, hv: torch.Tensor, alpha: float = 1.0):
        """
        Perceptron-style correction:
          W[y_true] += hv
          W[y_pred] -= hv
        """
        assert y_true.shape == y_pred.shape
        if alpha != 1.0:
            hv = hv * float(alpha)
        self.add_(y_true, hv, alpha=1.0)
        self.add_(y_pred, -hv, alpha=1.0)

    def logits(self, hv: torch.Tensor, temperature: float = 1.0, chunk: int = 0) -> torch.Tensor:
        """
        Cosine-similarity logits between hv and prototypes.
        hv: [N, D]
        return: [N, K]
        """
        dev = self.prototypes.device
        if hv.device != dev:
            hv = hv.to(dev)
        if hv.dtype != self.prototypes.dtype:
            hv = hv.to(self.prototypes.dtype)

        hv_n = _normalize_rows(hv)
        Pt = self.prototypes.t()

        if chunk is None or chunk <= 0 or hv_n.shape[0] <= chunk:
            out = hv_n @ Pt
        else:
            outs = []
            N = hv_n.shape[0]
            for s in range(0, N, chunk):
                e = min(s + chunk, N)
                outs.append(hv_n[s:e] @ Pt)
            out = torch.cat(outs, dim=0)

        if temperature != 1.0:
            out = out / float(temperature)
        return out

    def bg_logits(self, hv: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Cosine similarity to background prototype.
        hv: [N, D], return: [N]
        """
        dev = self.bg_prototype.device
        if hv.device != dev:
            hv = hv.to(dev)
        if hv.dtype != self.bg_prototype.dtype:
            hv = hv.to(self.bg_prototype.dtype)
        hv_n = _normalize_rows(hv)
        bg = self.bg_prototype
        out = hv_n @ bg
        if temperature != 1.0:
            out = out / float(temperature)
        return out


# -----------------------------
# Sampler: hard + random
# -----------------------------

class HDSampler:
    @staticmethod
    @torch.no_grad()
    def sample_hard_random(
        hardness: torch.Tensor,
        sample_percentage: float,
        hard_ratio: float,
        min_keep: int = 64,
    ) -> torch.Tensor:
        """
        Hard + random sampling based on hardness (larger => harder).
        Returns indices within the positive subset.
        """
        assert hardness.dim() == 1
        N = hardness.numel()
        if N == 0:
            return hardness.new_empty((0,), dtype=torch.long)

        M = max(int(N * float(sample_percentage)), int(min_keep))
        M = min(M, N)

        hard_n = int(M * float(hard_ratio))
        hard_n = min(hard_n, M)
        rand_n = M - hard_n

        _, sorted_idx = torch.sort(hardness, descending=True)
        hard_idx = sorted_idx[:hard_n]

        if rand_n <= 0:
            return hard_idx

        mask = torch.ones(N, device=hardness.device, dtype=torch.bool)
        mask[hard_idx] = False
        remain = torch.nonzero(mask, as_tuple=False).view(-1)

        if remain.numel() == 0:
            return hard_idx

        if remain.numel() <= rand_n:
            rand_idx = remain
        else:
            perm = torch.randperm(remain.numel(), device=hardness.device)
            rand_idx = remain[perm[:rand_n]]

        return torch.cat([hard_idx, rand_idx], dim=0)


# -----------------------------
# Core: glue encoder + memory + sampling + fusion
# -----------------------------

class HDCore(nn.Module):
    """
    HDC core:
    - encode mid features into hypervectors (HV)
    - compute HD logits
    - sample anchors for updates
    - update prototypes with GT and optional perceptron correction
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
            num_levels=cfg.num_levels,
            randomness=cfg.randomness,
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

        # Optional nested encoder cfg for chunk sizes / levels
        enc_cfg = _safe_get(hd_cfg, "ENCODER_CFG", {})
        num_levels = int(_safe_get(enc_cfg, "NUM_LEVELS", 100))
        randomness = float(_safe_get(enc_cfg, "RANDOMNESS", 0.0))
        encode_chunk = int(_safe_get(enc_cfg, "ENCODE_CHUNK", 8192))
        logits_chunk = int(_safe_get(enc_cfg, "LOGITS_CHUNK", 8192))
        anchor_id_scale = float(_safe_get(hd_cfg, "ANCHOR_ID_SCALE", 0.0))
        bg_enabled = bool(_safe_get(hd_cfg, "BG_ENABLED", True))
        bg_margin_scale = float(_safe_get(hd_cfg, "BG_MARGIN_SCALE", 1.0))
        bg_sample_ratio = float(_safe_get(hd_cfg, "BG_SAMPLE_RATIO", 0.25))
        bg_min_per_anchor = int(_safe_get(hd_cfg, "BG_MIN_PER_ANCHOR", 32))

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
            num_levels=num_levels,
            randomness=randomness,
            encode_chunk=encode_chunk,
            logits_chunk=logits_chunk,
            anchor_id_scale=anchor_id_scale,
            bg_enabled=bg_enabled,
            bg_margin_scale=bg_margin_scale,
            bg_sample_ratio=bg_sample_ratio,
            bg_min_per_anchor=bg_min_per_anchor,
        )
        return HDCore(cfg)

    def inject_anchor_context(
        self,
        feat_mid: torch.Tensor,
        anchor_ids: torch.Tensor,
        num_anchors: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Add a lightweight, deterministic anchor-id context into features so anchors at the same
        cell are no longer forced to share exactly the same HD logits.
        """
        scale = float(getattr(self.cfg, "anchor_id_scale", 0.0))
        if scale <= 0.0:
            return feat_mid

        assert feat_mid.dim() == 2, f"Expected feat_mid [N, C], got {tuple(feat_mid.shape)}"
        assert anchor_ids.dim() == 1, f"Expected anchor_ids [N], got {tuple(anchor_ids.shape)}"
        assert feat_mid.shape[0] == anchor_ids.shape[0], \
            f"N mismatch: feat_mid={feat_mid.shape[0]}, anchor_ids={anchor_ids.shape[0]}"

        N, C = feat_mid.shape
        out = feat_mid.clone()
        a = anchor_ids.long()
        if num_anchors is not None and int(num_anchors) > 0:
            a = torch.remainder(a, int(num_anchors))

        # Map each anchor id to a stable feature channel index.
        cols = torch.remainder(a, C)
        rows = torch.arange(N, device=out.device)
        out[rows, cols] = out[rows, cols] + scale
        return out

    def compute_hd_logits(self, feat_mid: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Streaming compute:
        feat chunk -> hv -> (inplace quantize) -> logits chunk
        Avoids materializing full hv (N x D) on GPU -> prevents OOM.

        Returns:
        logits_hd: [N, K]
        hv: None  (we intentionally do NOT return full hv to save memory)
        """
        N = feat_mid.shape[0]
        K = self.cfg.num_classes
        device = feat_mid.device
        dtype = self.memory.prototypes.dtype  # logits will use prototype dtype

        encode_chunk = int(self.cfg.encode_chunk) if int(self.cfg.encode_chunk) > 0 else N

        out_logits = []
        for s in range(0, N, encode_chunk):
            e = min(s + encode_chunk, N)
            feat_chunk = feat_mid[s:e]

            # Encode to hv (on GPU). Quantization behavior is controlled only by cfg.quantize
            # so train/build/infer remain consistent.
            use_quantize = bool(self.cfg.quantize)
            hv = self.embedder.forward(feat_chunk, quantize=use_quantize)


            # NOTE: embedder.forward already quantizes; if you moved quantize out, call _hard_quantize_inplace here
            # _hard_quantize_inplace(hv)

            # Compute logits for this chunk, immediately
            logits = self.memory.logits(
                hv,
                temperature=float(self.cfg.temperature),
                chunk=int(self.cfg.logits_chunk) if int(self.cfg.logits_chunk) > 0 else 0
            )
            if bool(getattr(self.cfg, "bg_enabled", True)):
                bg_logit = self.memory.bg_logits(hv, temperature=float(self.cfg.temperature)).unsqueeze(1)
                logits = logits - float(getattr(self.cfg, "bg_margin_scale", 1.0)) * bg_logit

            out_logits.append(logits)

            # Explicitly free
            del hv, logits

        logits_hd = torch.cat(out_logits, dim=0)
        return logits_hd, None


    def fuse_logits(
        self,
        logits_origin: torch.Tensor,
        logits_hd: torch.Tensor,
        mode: Optional[str] = None,
        lam: Optional[float] = None,
    ) -> torch.Tensor:
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

    # -------- sampling / update helpers --------

    @torch.no_grad()
    def _filter_positive(
        self,
        labels: torch.Tensor,
        feat_mid: torch.Tensor,
        logits_origin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Filter positive anchors for updates.
        labels: [N] with bg <= bg_threshold if ignore_bg
        """
        assert labels.dim() == 1
        assert feat_mid.shape[0] == labels.shape[0]

        if self.cfg.ignore_bg:
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
    def _compute_hardness(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        hardness = best_other - true_score (larger => harder)
        """
        # labels from anchor targets are usually 1..K for positives; convert to 0..K-1.
        if labels.numel() > 0 and int(labels.min().item()) >= 1:
            labels = labels - 1
        idx = labels.view(-1, 1)
        true_score = torch.gather(logits, 1, idx).squeeze(1)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, idx, True)
        other = logits.masked_fill(mask, float("-inf"))
        best_other = other.max(dim=1).values
        return best_other - true_score

    @torch.no_grad()
    def sample_for_update(
        self,
        labels: torch.Tensor,
        feat_mid: torch.Tensor,
        logits_origin: Optional[torch.Tensor] = None,
        prefer_hard: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Select a subset of positive anchors for update.
        Returns:
            selected_anchor_indices: indices in [0..N-1]
            selected_labels: labels for these indices (same space as input labels)
        """
        pos_feat, pos_labels, pos_logits, pos_indices = self._filter_positive(labels, feat_mid, logits_origin)
        if pos_indices.numel() == 0:
            return {"selected_anchor_indices": pos_indices, "selected_labels": pos_labels}

        Npos = pos_indices.numel()
        M = max(int(Npos * float(self.cfg.sample_percentage)), int(self.cfg.min_pos))
        M = min(M, Npos)
        if M <= 0:
            return {
                "selected_anchor_indices": pos_indices.new_empty((0,), dtype=torch.long),
                "selected_labels": pos_labels.new_empty((0,), dtype=torch.long),
            }

        if (not prefer_hard) or (pos_logits is None):
            perm = torch.randperm(Npos, device=pos_indices.device)
            keep_pos = perm[:M]
        else:
            hardness = self._compute_hardness(pos_logits, pos_labels)
            keep_pos = HDSampler.sample_hard_random(
                hardness=hardness,
                sample_percentage=float(self.cfg.sample_percentage),
                hard_ratio=float(self.cfg.hard_ratio),
                min_keep=int(self.cfg.min_pos),
            )
            if keep_pos.numel() > M:
                keep_pos = keep_pos[:M]

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
        Train-style update: W[y] += hv
        """
        if selected_anchor_indices is None:
            pos_feat, pos_labels, _, _ = self._filter_positive(labels, feat_mid, None)
        else:
            pos_feat = feat_mid[selected_anchor_indices]
            pos_labels = labels[selected_anchor_indices].long()

        if pos_feat.numel() == 0:
            return
        # labels are usually 1..K for positives; memory indices are 0..K-1
        if pos_labels.numel() > 0 and int(pos_labels.min().item()) >= 1:
            pos_labels = pos_labels - 1

        steps = max(int(self.cfg.update_steps), 1)
        for _ in range(steps):
            hv = self.embedder.forward_chunked(pos_feat, chunk=int(self.cfg.encode_chunk))
            # Real-valued centroid update: accumulate unit-norm sample hypervectors.
            hv = _normalize_rows(hv)
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
        Retrain-style perceptron correction on wrong predictions.
        """
        if selected_anchor_indices is None:
            pos_feat, pos_labels, pos_logits, _ = self._filter_positive(labels, feat_mid, logits_origin)
        else:
            pos_feat = feat_mid[selected_anchor_indices]
            pos_labels = labels[selected_anchor_indices].long()
            pos_logits = logits_origin[selected_anchor_indices]

        if pos_feat.numel() == 0:
            return
        # labels are usually 1..K for positives; convert to 0..K-1 to match logits/memory indexing
        if pos_labels.numel() > 0 and int(pos_labels.min().item()) >= 1:
            pos_labels = pos_labels - 1

        hv = self.embedder.forward_chunked(pos_feat, chunk=int(self.cfg.encode_chunk), quantize=self.cfg.quantize)
        hv = _normalize_rows(hv)
        pred = pos_logits.argmax(dim=1).long()
        wrong = pred != pos_labels
        if wrong.sum().item() == 0:
            return

        y_true = pos_labels[wrong]
        y_pred = pred[wrong]
        hv_wrong = hv[wrong]

        steps = max(int(self.cfg.retrain_steps), 1)
        for _ in range(steps):
            self.memory.retrain_correct_(y_true, y_pred, hv_wrong, alpha=alpha)

        self._update_counter += 1
        if (self.cfg.normalize_every > 0) and (self._update_counter % int(self.cfg.normalize_every) == 0):
            self.memory.normalize_()

    # -------- persistence --------

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
        self.memory.normalize_()
