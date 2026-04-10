from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _to_cpu_clone_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu().clone()


def _state_dict_to_cpu(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = copy.deepcopy(v)
    return out


def _clone_mem_hv_cpu(mem_hv: Optional[Dict[int, Optional[torch.Tensor]]]) -> Dict[int, Optional[torch.Tensor]]:
    out: Dict[int, Optional[torch.Tensor]] = {}
    if mem_hv is None:
        return out
    for k, v in mem_hv.items():
        if v is None:
            out[int(k)] = None
        else:
            if not torch.is_tensor(v):
                v = torch.tensor(v)
            out[int(k)] = v.detach().float().cpu().clone()
    return out


def _restore_mem_hv_to_cpu(payload_mem_hv: Optional[Dict[Any, Any]], num_classes: int) -> Dict[int, Optional[torch.Tensor]]:
    restored: Dict[int, Optional[torch.Tensor]] = {}
    if not isinstance(payload_mem_hv, dict):
        for c in range(num_classes):
            restored[c] = None
        return restored

    for c in range(num_classes):
        t = payload_mem_hv.get(c, payload_mem_hv.get(str(c), None))
        if t is None:
            restored[c] = None
        else:
            if not torch.is_tensor(t):
                t = torch.tensor(t)
            restored[c] = t.detach().float().cpu().clone()
    return restored


class OnlineStateIO:
    """
    Save/load online adaptation state with HDCore memory tensors.

    Compatible memory payloads:
    - full memory state_dict: payload['memory_state']
    - simplified tensors: classify_weights/prototypes/bg_weight/bg_prototype
    """

    @staticmethod
    @torch.no_grad()
    def build_payload(
        *,
        hd_core,
        teacher_prototypes: Optional[torch.Tensor] = None,
        teacher_bg_prototype: Optional[torch.Tensor] = None,
        source_prototypes: Optional[torch.Tensor] = None,
        source_bg_prototype: Optional[torch.Tensor] = None,
        tau_prob: Optional[float] = None,
        tau_margin: Optional[float] = None,
        alpha: Optional[float] = None,
        step_idx: int = 0,
        update_idx: int = 0,
        best_metric: float = float("-inf"),
        last_metric: float = float("-inf"),
        best_state: Optional[Dict[str, Any]] = None,
        mem_hv: Optional[Dict[int, Optional[torch.Tensor]]] = None,
        online_cfg: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        mem = hd_core.memory

        payload: Dict[str, Any] = {
            # canonical memory snapshot
            "memory_state": _state_dict_to_cpu(mem.state_dict()),
            # simplified compatibility fields
            "classify_weights": _to_cpu_clone_tensor(mem.classify_weights),
            "prototypes": _to_cpu_clone_tensor(mem.prototypes),
            "step_idx": int(step_idx),
            "update_idx": int(update_idx),
            "best_metric": float(best_metric),
            "last_metric": float(last_metric),
            "tau_prob": None if tau_prob is None else float(tau_prob),
            "tau_margin": None if tau_margin is None else float(tau_margin),
            "alpha": None if alpha is None else float(alpha),
            "mem_hv": _clone_mem_hv_cpu(mem_hv),
            "online_cfg": {} if online_cfg is None else copy.deepcopy(online_cfg),
        }

        if hasattr(mem, "bg_weight"):
            payload["bg_weight"] = _to_cpu_clone_tensor(mem.bg_weight)
        if hasattr(mem, "bg_prototype"):
            payload["bg_prototype"] = _to_cpu_clone_tensor(mem.bg_prototype)

        if teacher_prototypes is not None:
            payload["teacher_prototypes"] = _to_cpu_clone_tensor(teacher_prototypes)
        if teacher_bg_prototype is not None:
            payload["teacher_bg_prototype"] = _to_cpu_clone_tensor(teacher_bg_prototype)
        if source_prototypes is not None:
            payload["source_prototypes"] = _to_cpu_clone_tensor(source_prototypes)
        if source_bg_prototype is not None:
            payload["source_bg_prototype"] = _to_cpu_clone_tensor(source_bg_prototype)

        if best_state is not None:
            payload["best_state"] = copy.deepcopy(best_state)

        if extra:
            payload.update(copy.deepcopy(extra))

        return payload

    @staticmethod
    @torch.no_grad()
    def apply_payload(
        *,
        hd_core,
        payload: Dict[str, Any],
        device: torch.device,
        num_classes: int,
        strict_memory: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            raise RuntimeError(f"Online state payload must be dict, got {type(payload)}")

        mem = hd_core.memory
        loaded_any = False

        # 1) canonical state dict path
        if "memory_state" in payload and isinstance(payload["memory_state"], dict):
            mem.load_state_dict(payload["memory_state"], strict=strict_memory)
            loaded_any = True

        # 2) simplified tensors path
        if "classify_weights" in payload and hasattr(mem, "classify_weights"):
            cw = payload["classify_weights"]
            if not torch.is_tensor(cw):
                cw = torch.tensor(cw)
            mem.classify_weights.copy_(cw.to(mem.classify_weights.device).to(mem.classify_weights.dtype))
            loaded_any = True

        if "prototypes" in payload and payload["prototypes"] is not None and hasattr(mem, "prototypes"):
            p = payload["prototypes"]
            if not torch.is_tensor(p):
                p = torch.tensor(p)
            mem.prototypes.copy_(p.to(mem.prototypes.device).to(mem.prototypes.dtype))
            loaded_any = True

        if "bg_weight" in payload and hasattr(mem, "bg_weight"):
            bgw = payload["bg_weight"]
            if not torch.is_tensor(bgw):
                bgw = torch.tensor(bgw)
            mem.bg_weight.copy_(bgw.to(mem.bg_weight.device).to(mem.bg_weight.dtype))
            loaded_any = True

        if "bg_prototype" in payload and payload["bg_prototype"] is not None and hasattr(mem, "bg_prototype"):
            bgp = payload["bg_prototype"]
            if not torch.is_tensor(bgp):
                bgp = torch.tensor(bgp)
            mem.bg_prototype.copy_(bgp.to(mem.bg_prototype.device).to(mem.bg_prototype.dtype))
            loaded_any = True

        if not loaded_any:
            raise RuntimeError("No valid HD memory fields found in payload")

        mem.normalize_()

        # restore non-memory runtime states
        restored: Dict[str, Any] = {
            "step_idx": int(payload.get("step_idx", 0)),
            "update_idx": int(payload.get("update_idx", 0)),
            "best_metric": float(payload.get("best_metric", float("-inf"))),
            "last_metric": float(payload.get("last_metric", float("-inf"))),
            "tau_prob": payload.get("tau_prob", None),
            "tau_margin": payload.get("tau_margin", None),
            "alpha": payload.get("alpha", None),
            "online_cfg": copy.deepcopy(payload.get("online_cfg", {})),
            "best_state": copy.deepcopy(payload.get("best_state", None)),
            "mem_hv": _restore_mem_hv_to_cpu(payload.get("mem_hv", None), num_classes),
        }

        for k in ("teacher_prototypes", "teacher_bg_prototype", "source_prototypes", "source_bg_prototype"):
            if k in payload and payload[k] is not None:
                t = payload[k]
                if not torch.is_tensor(t):
                    t = torch.tensor(t)
                restored[k] = t.detach().float().to(device).clone()

        return restored

    @staticmethod
    def save(path: str, payload: Dict[str, Any]):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, p)

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Online state file not found: {path}")
        obj = torch.load(p, map_location="cpu")
        if not isinstance(obj, dict):
            raise RuntimeError(f"Online state file must contain a dict payload, got {type(obj)}")
        return obj
