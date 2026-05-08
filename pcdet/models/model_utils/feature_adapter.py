import math

import torch
import torch.nn as nn


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class ResidualAdapter2d(nn.Module):
    """
    Lightweight residual adapter for BEV/feature-map tensors.

    The adapter follows the common down-proj -> nonlinearity -> up-proj design:
        y = x + scale * up(ReLU(down(x)))

    By default, the up projection is zero-initialized so this module is an exact
    identity at insertion time. That keeps old checkpoints and baseline behavior
    stable before adapter training is enabled.
    """

    def __init__(
        self,
        channels,
        reduction=32,
        hidden_channels=None,
        kernel_size=1,
        activation="relu",
        dropout=0.0,
        scale=1.0,
        learnable_scale=False,
        zero_init=True,
        use_norm=False,
    ):
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")

        reduction = max(int(reduction), 1)
        if hidden_channels is None:
            hidden_channels = max(channels // reduction, 1)
        hidden_channels = int(hidden_channels)
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")

        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        padding = kernel_size // 2

        self.channels = channels
        self.hidden_channels = hidden_channels
        self.reduction = reduction
        self.zero_init = bool(zero_init)

        self.down_proj = nn.Conv2d(
            channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=True
        )
        self.norm = nn.BatchNorm2d(hidden_channels) if use_norm else None
        self.act = self._build_activation(activation)
        self.dropout = nn.Dropout2d(p=float(dropout)) if float(dropout) > 0 else None
        self.up_proj = nn.Conv2d(hidden_channels, channels, kernel_size=1, padding=0, bias=True)

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float32))
        else:
            self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32), persistent=False)

        self.reset_parameters()

    @staticmethod
    def _build_activation(name):
        name = str(name).lower()
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name == "gelu":
            return nn.GELU()
        if name in ("silu", "swish"):
            return nn.SiLU(inplace=True)
        if name in ("identity", "none"):
            return nn.Identity()
        raise ValueError(f"Unsupported adapter activation: {name}")

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.bias)

        if self.norm is not None:
            nn.init.ones_(self.norm.weight)
            nn.init.zeros_(self.norm.bias)

        if self.zero_init:
            nn.init.zeros_(self.up_proj.weight)
            nn.init.zeros_(self.up_proj.bias)
        else:
            nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.up_proj(x)
        return residual + self.scale.to(dtype=x.dtype, device=x.device) * x


def build_feature_adapter_from_cfg(adapter_cfg, channels):
    enabled = bool(_cfg_get(adapter_cfg, "ENABLED", False))
    if not enabled:
        return None

    adapter_type = str(_cfg_get(adapter_cfg, "TYPE", "residual_2d")).lower()
    if adapter_type not in ("residual_2d", "bottleneck_2d", "adapter2d"):
        raise ValueError(f"Unsupported feature adapter type: {adapter_type}")

    return ResidualAdapter2d(
        channels=channels,
        reduction=_cfg_get(adapter_cfg, "RATIO", _cfg_get(adapter_cfg, "REDUCTION", 32)),
        hidden_channels=_cfg_get(adapter_cfg, "HIDDEN_CHANNELS", None),
        kernel_size=_cfg_get(adapter_cfg, "KERNEL_SIZE", 1),
        activation=_cfg_get(adapter_cfg, "ACTIVATION", "relu"),
        dropout=_cfg_get(adapter_cfg, "DROPOUT", 0.0),
        scale=_cfg_get(adapter_cfg, "SCALE", 1.0),
        learnable_scale=_cfg_get(adapter_cfg, "LEARNABLE_SCALE", False),
        zero_init=_cfg_get(adapter_cfg, "ZERO_INIT", True),
        use_norm=_cfg_get(adapter_cfg, "USE_NORM", False),
    )
