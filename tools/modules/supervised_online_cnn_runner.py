from __future__ import annotations

import copy
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from pcdet.models import load_data_to_gpu


def _state_dict_to_cpu(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        elif isinstance(v, dict):
            out[k] = _state_dict_to_cpu(v)
        elif isinstance(v, list):
            out[k] = [(_state_dict_to_cpu(x) if isinstance(x, dict) else x.detach().cpu().clone() if torch.is_tensor(x) else copy.deepcopy(x)) for x in v]
        elif isinstance(v, tuple):
            out[k] = tuple((_state_dict_to_cpu(x) if isinstance(x, dict) else x.detach().cpu().clone() if torch.is_tensor(x) else copy.deepcopy(x)) for x in v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class SupervisedOnlineCNNConfig:
    stream_split: str = "val_stream"
    eval_split: str = "val_eval"
    stream_ratio: float = 1.0
    max_stream_samples: int = 0
    use_stream_prefix: bool = True
    stream_seed: int = 0

    update_every_n_batches: int = 1
    grad_clip: float = 0.0

    max_pos_per_class_per_batch: int = 128
    min_pos_per_class_per_batch: int = 0
    max_total_pos_per_batch: int = 0

    eval_every_updates: int = 50
    metric_key: str = "recall/rcnn_0.3"
    fast_recall_only: bool = True
    final_full_eval: bool = True

    save_every_updates: int = 50
    log_every_n_batches: int = 10
    save_to_file: bool = False
    save_best_model: bool = True
    save_last_model: bool = True
    experiment_note: str = "supervised_stream_cnn_head_update"


class SupervisedOnlineCNNRunner:
    """
    Supervised streaming CNN head updater.

    Design:
    - Force the detector to use baseline CNN logits (dense_head.hd_mode='baseline').
    - Keep the detector mostly frozen and update only selected CNN classification-head params.
    - Reuse the same stream/eval protocol as supervised HD online adaptation.
    - Evaluate periodically on a fixed holdout loader and save best/last checkpoints.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        stream_loader,
        logger,
        cfg: Optional[Dict[str, Any]] = None,
        *,
        eval_loader=None,
        eval_fn: Optional[Callable[[torch.nn.Module, Any], float]] = None,
        output_dir: Optional[str] = None,
        state_save_prefix: str = "supervised_online_cnn",
    ):
        self.model = model
        self.optimizer = optimizer
        self.stream_loader = stream_loader
        self.eval_loader = eval_loader
        self.eval_fn = eval_fn
        self.logger = logger
        self.cfg = self._build_cfg(cfg or {})

        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.state_save_prefix = state_save_prefix
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.metrics_csv_path = self.output_dir / "online_metrics.csv"
        else:
            self.metrics_csv_path = None
        self._metrics_header_written = False

        self.device = next(self.model.parameters()).device

        self.dense_head = getattr(self.model, "dense_head", None)
        if self.dense_head is None:
            raise RuntimeError("SupervisedOnlineCNNRunner requires model.dense_head.")

        self.num_classes = int(getattr(self.dense_head, "num_class", 0))
        if self.num_classes <= 0:
            raise RuntimeError("Invalid dense_head.num_class.")
        self.num_anchors_per_loc = int(getattr(self.dense_head, "num_anchors_per_location", 0))
        if self.num_anchors_per_loc <= 0:
            raise RuntimeError("Invalid dense_head.num_anchors_per_location.")

        self._force_baseline_mode()
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(self.cfg.stream_seed))

        self.step_idx = 0
        self.update_idx = 0
        self.best_metric = float("-inf")
        self.last_metric = float("-inf")
        self.best_state: Optional[Dict[str, Any]] = None

    def run(self, *, max_steps: int = -1):
        self.model.eval()
        self._force_baseline_mode()
        self.optimizer.zero_grad(set_to_none=True)
        self.best_state = self._snapshot_state()
        self._log("[CNN] start")
        self._log(
            f"[CNN] cfg: positive-anchor-only cls update, "
            f"update_every_n_batches={self.cfg.update_every_n_batches}, grad_clip={self.cfg.grad_clip:.4f}"
        )

        if self.eval_fn is not None and self.eval_loader is not None:
            baseline_metric = self._evaluate_metric()
            self.last_metric = baseline_metric
            self.best_metric = baseline_metric
            self.best_state = self._snapshot_state()
            self._append_metric_row(
                {
                    "event": "baseline_eval",
                    "step_idx": int(self.step_idx),
                    "update_idx": int(self.update_idx),
                    "metric": float(baseline_metric),
                    "best_metric": float(self.best_metric),
                    "loss": "",
                    "grad_norm": "",
                }
            )
            self._save_state(tag="baseline")
            self._log(f"[BASELINE] metric={baseline_metric:.6f} | saved baseline state")

        t0 = time.time()
        accum = self._new_accum_buffer()

        for batch in self.stream_loader:
            if max_steps > 0 and self.step_idx >= max_steps:
                break

            self.step_idx += 1
            step = self._process_one_step(batch)
            self._accumulate_step(accum, step)

            if self.step_idx % self.cfg.update_every_n_batches == 0:
                self._finalize_update(accum, reason=f"step_interval@{self.step_idx}")
                accum = self._new_accum_buffer()

            if self.step_idx % self.cfg.log_every_n_batches == 0:
                self._log(
                    f"[STEP {self.step_idx}] pending_batches={accum['pending_batches']} "
                    f"pending_loss={accum['loss_sum'] / max(1, accum['pending_batches']):.6f}"
                )

        self._finalize_update(accum, reason="final_flush", force=True)

        if self.cfg.save_last_model:
            self._save_state(tag="last")

        self._export_best_model_payload()
        self._log(f"[CNN] finished | steps={self.step_idx} updates={self.update_idx} elapsed={time.time()-t0:.1f}s")

    def _process_one_step(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        self._force_baseline_mode()
        self.model.eval()

        load_data_to_gpu(batch_dict)
        _pred, _ret = self.model(batch_dict)
        if "box_cls_labels" not in self.dense_head.forward_ret_dict:
            raise RuntimeError(
                "dense_head.forward_ret_dict['box_cls_labels'] not found during CNN online update. "
                "This runner expects target assignment to be enabled in eval-time forward, "
                "for example via HD.ASSIGN_TARGETS_IN_EVAL=True with gt_boxes present."
            )
        logits_bnk = self._extract_anchor_logits(self.dense_head.forward_ret_dict)
        pos_mask_bn, pos_label_bn = self._extract_supervised_positive_targets(self.dense_head.forward_ret_dict)

        selected_logits: List[torch.Tensor] = []
        selected_labels: List[torch.Tensor] = []
        selected_total = 0
        class_count = torch.zeros((self.num_classes,), device=self.device, dtype=torch.long)

        B = logits_bnk.shape[0]
        for b in range(B):
            pos_idx = torch.nonzero(pos_mask_bn[b], as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                continue

            pos_labels = pos_label_bn[b][pos_idx]
            pos_idx, pos_labels = self._sample_positive_anchors(pos_idx, pos_labels)
            if pos_idx.numel() == 0:
                continue

            selected_logits.append(logits_bnk[b][pos_idx])
            selected_labels.append(pos_labels)
            selected_total += int(pos_idx.numel())
            class_count += self._count_by_class(pos_labels - 1, self.num_classes)

        if selected_logits:
            logits_sel = torch.cat(selected_logits, dim=0)
            labels_sel = torch.cat(selected_labels, dim=0) - 1
            loss = F.cross_entropy(logits_sel.float(), labels_sel.long(), reduction="mean")
        else:
            loss = logits_bnk.sum() * 0.0

        loss.backward()

        out = {
            "loss": float(loss.detach().item()),
            "selected": int(selected_total),
            "updated_classes": int((class_count > 0).sum().item()),
            "class_count": class_count.detach(),
        }
        return out

    def _finalize_update(self, accum: Dict[str, Any], reason: str, force: bool = False):
        if accum["pending_batches"] <= 0 and not force:
            return
        if accum["pending_batches"] <= 0 and force:
            return
        if accum["selected_total"] <= 0 and not force:
            self.optimizer.zero_grad(set_to_none=True)
            return
        if accum["selected_total"] <= 0 and force:
            self.optimizer.zero_grad(set_to_none=True)
            return

        self.update_idx += 1
        avg_loss = accum["loss_sum"] / max(1, accum["pending_batches"])
        grad_norm = self._compute_grad_norm()
        if self.cfg.grad_clip > 0:
            clip_grad_norm_(self._trainable_parameters(), max_norm=float(self.cfg.grad_clip))
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._force_baseline_mode()

        self._log(
            f"[UPDATE {self.update_idx}] reason={reason} batches={accum['pending_batches']} "
            f"avg_loss={avg_loss:.6f} grad_norm={grad_norm:.6f} "
            f"selected={accum['selected_total']} updated_classes={int((accum['class_count'] > 0).sum().item())}"
        )

        metric = float("nan")
        eval_enabled = (
            self.cfg.eval_every_updates > 0
            and self.eval_fn is not None
            and self.eval_loader is not None
        )
        should_eval_now = eval_enabled and (self.update_idx % self.cfg.eval_every_updates == 0)

        if should_eval_now:
            metric = self._evaluate_metric()
            self.last_metric = metric
            if metric > self.best_metric:
                self.best_metric = metric
                self.best_state = self._snapshot_state()
                if self.cfg.save_best_model:
                    self._save_state(tag="best")
            self._log(f"[UPDATE {self.update_idx}] post-eval metric={metric:.6f}, best={self.best_metric:.6f}")
        else:
            self.last_metric = float("nan")
            if not eval_enabled:
                self.best_state = self._snapshot_state()

        self._append_metric_row(
            {
                "event": "update",
                "step_idx": int(self.step_idx),
                "update_idx": int(self.update_idx),
                "metric": float(metric),
                "best_metric": float(self.best_metric),
                "loss": float(avg_loss),
                "grad_norm": float(grad_norm),
                "selected": int(accum["selected_total"]),
                "updated_classes": int((accum["class_count"] > 0).sum().item()),
            }
        )

        if self.cfg.save_every_updates > 0 and (self.update_idx % self.cfg.save_every_updates == 0):
            self._save_state(tag=f"upd{self.update_idx:04d}")

    def _evaluate_metric(self) -> float:
        self._force_baseline_mode()
        if self.eval_fn is None or self.eval_loader is None:
            return float("nan")
        out = self.eval_fn(self.model, self.eval_loader)
        if isinstance(out, dict):
            key = str(self.cfg.metric_key)
            if key in out:
                return float(out[key])
            raise RuntimeError(f"eval_fn returned dict without metric key '{key}'. keys={list(out.keys())[:20]}")
        return float(out)

    def _snapshot_state(self) -> Dict[str, Any]:
        return {
            "model_state": _state_dict_to_cpu(self.model.state_dict()),
            "optimizer_state": _state_dict_to_cpu(self.optimizer.state_dict()),
            "step_idx": int(self.step_idx),
            "update_idx": int(self.update_idx),
            "best_metric": float(self.best_metric),
            "last_metric": float(self.last_metric),
        }

    def _save_state(self, tag: str):
        if self.output_dir is None:
            return
        payload = self._snapshot_state()
        payload["online_cfg"] = copy.deepcopy(self.cfg.__dict__)
        payload["runner_type"] = "supervised_online_cnn_runner"
        payload["experiment_note"] = self.cfg.experiment_note
        payload["metric_key"] = self.cfg.metric_key
        payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        path = self.output_dir / f"{self.state_save_prefix}_{tag}.pth"
        torch.save(payload, path)
        self._log(f"[STATE] saved: {str(path)}")

    def _export_best_model_payload(self):
        if self.output_dir is None or self.best_state is None:
            return
        path = self.output_dir / "best_supervised_cnn_head.pth"
        torch.save(self.best_state, path)
        self._log(f"[BEST] exported best CNN head checkpoint: {str(path)}")

    def _trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _compute_grad_norm(self) -> float:
        sq_sum = 0.0
        found = False
        for p in self._trainable_parameters():
            if p.grad is None:
                continue
            found = True
            g = p.grad.detach()
            sq_sum += float(torch.sum(g * g).item())
        return sq_sum ** 0.5 if found else 0.0

    def _force_baseline_mode(self):
        if hasattr(self.dense_head, "hd_mode"):
            self.dense_head.hd_mode = "baseline"

    @staticmethod
    def _new_accum_buffer() -> Dict[str, Any]:
        return {
            "pending_batches": 0,
            "loss_sum": 0.0,
            "selected_total": 0,
            "class_count": None,
        }

    @staticmethod
    def _accumulate_step(accum: Dict[str, Any], step: Dict[str, Any]):
        accum["pending_batches"] += 1
        accum["loss_sum"] += float(step.get("loss", 0.0))
        accum["selected_total"] += int(step.get("selected", 0))
        if "class_count" not in step:
            return
        if accum["class_count"] is None:
            accum["class_count"] = step["class_count"].detach().clone()
        else:
            accum["class_count"] += step["class_count"]

    @staticmethod
    def _build_cfg(cfg_in: Dict[str, Any]) -> SupervisedOnlineCNNConfig:
        cfg = SupervisedOnlineCNNConfig()
        for k, v in cfg_in.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        cfg.stream_split = str(cfg.stream_split)
        cfg.eval_split = str(cfg.eval_split)
        cfg.stream_ratio = float(max(0.0, min(1.0, cfg.stream_ratio)))
        cfg.max_stream_samples = int(max(0, cfg.max_stream_samples))
        cfg.use_stream_prefix = bool(cfg.use_stream_prefix)
        cfg.stream_seed = int(cfg.stream_seed)

        cfg.update_every_n_batches = int(max(1, cfg.update_every_n_batches))
        cfg.grad_clip = float(max(0.0, cfg.grad_clip))
        cfg.max_pos_per_class_per_batch = int(max(0, cfg.max_pos_per_class_per_batch))
        cfg.min_pos_per_class_per_batch = int(max(0, cfg.min_pos_per_class_per_batch))
        cfg.max_total_pos_per_batch = int(max(0, cfg.max_total_pos_per_batch))

        cfg.eval_every_updates = int(max(0, cfg.eval_every_updates))
        cfg.metric_key = str(cfg.metric_key)
        cfg.fast_recall_only = bool(cfg.fast_recall_only)
        cfg.final_full_eval = bool(cfg.final_full_eval)

        cfg.save_every_updates = int(max(0, cfg.save_every_updates))
        cfg.log_every_n_batches = int(max(1, cfg.log_every_n_batches))
        cfg.save_to_file = bool(cfg.save_to_file)
        cfg.save_best_model = bool(cfg.save_best_model)
        cfg.save_last_model = bool(cfg.save_last_model)
        cfg.experiment_note = str(cfg.experiment_note)
        return cfg

    def _append_metric_row(self, row: Dict[str, Any]):
        if self.metrics_csv_path is None:
            return
        fieldnames = [
            "event", "step_idx", "update_idx", "metric", "best_metric", "loss", "grad_norm", "selected", "updated_classes"
        ]
        self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (not self._metrics_header_written) and (not self.metrics_csv_path.exists())
        with open(self.metrics_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                self._metrics_header_written = True
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    def _log(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def _extract_anchor_logits(self, forward_ret_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        cls_hwk = forward_ret_dict.get("cls_preds_origin", None)
        if cls_hwk is None:
            cls_hwk = forward_ret_dict.get("cls_preds", None)
        if cls_hwk is None:
            raise RuntimeError("Expected cls_preds_origin or cls_preds in dense_head.forward_ret_dict.")
        if cls_hwk.dim() != 4:
            raise RuntimeError(f"Unexpected cls prediction shape: {tuple(cls_hwk.shape)}")

        B, H, W, AK = cls_hwk.shape
        K = self.num_classes
        A = self.num_anchors_per_loc
        if AK != A * K:
            raise RuntimeError(f"Unexpected cls prediction last dim: got {AK}, expected {A*K}")
        return cls_hwk.reshape(B, H * W * A, K).float()

    def _extract_supervised_positive_targets(
        self, forward_ret_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        box_cls_labels = forward_ret_dict.get("box_cls_labels", None)
        if box_cls_labels is None:
            raise RuntimeError("dense_head.forward_ret_dict['box_cls_labels'] not found.")
        if box_cls_labels.dim() != 2:
            raise RuntimeError(f"Unexpected box_cls_labels shape: {tuple(box_cls_labels.shape)}")

        pos_mask = box_cls_labels > 0
        pos_labels = box_cls_labels.long()
        if pos_mask.any():
            pos_only = pos_labels[pos_mask]
            if int(pos_only.min().item()) < 1 or int(pos_only.max().item()) > self.num_classes:
                raise RuntimeError(
                    f"Positive box_cls_labels out of range: min={int(pos_only.min().item())}, "
                    f"max={int(pos_only.max().item())}, num_classes={self.num_classes}"
                )
        return pos_mask, pos_labels

    def _sample_positive_anchors(
        self, pos_idx: torch.Tensor, pos_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        selected_idx_parts: List[torch.Tensor] = []
        selected_label_parts: List[torch.Tensor] = []

        max_per_class = int(self.cfg.max_pos_per_class_per_batch)
        min_per_class = int(self.cfg.min_pos_per_class_per_batch)

        for cls_id in range(1, self.num_classes + 1):
            cls_mask = pos_labels == cls_id
            idx_cls = pos_idx[cls_mask]
            if idx_cls.numel() == 0:
                continue
            if min_per_class > 0 and idx_cls.numel() < min_per_class:
                continue

            if max_per_class > 0 and idx_cls.numel() > max_per_class:
                perm = torch.randperm(idx_cls.numel(), generator=self._rng)
                idx_cls = idx_cls[perm.to(idx_cls.device)[:max_per_class]]

            selected_idx_parts.append(idx_cls)
            selected_label_parts.append(torch.full_like(idx_cls, cls_id, dtype=torch.long))

        if not selected_idx_parts:
            empty = pos_idx.new_empty((0,), dtype=torch.long)
            return empty, empty

        selected_idx = torch.cat(selected_idx_parts, dim=0)
        selected_labels = torch.cat(selected_label_parts, dim=0)

        max_total = int(self.cfg.max_total_pos_per_batch)
        if max_total > 0 and selected_idx.numel() > max_total:
            perm = torch.randperm(selected_idx.numel(), generator=self._rng)
            keep = perm.to(selected_idx.device)[:max_total]
            selected_idx = selected_idx[keep]
            selected_labels = selected_labels[keep]

        return selected_idx, selected_labels

    @staticmethod
    def _count_by_class(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        out = torch.zeros((num_classes,), device=labels.device, dtype=torch.long)
        if labels.numel() == 0:
            return out
        ones = torch.ones_like(labels, dtype=torch.long)
        out.index_add_(0, labels.long(), ones)
        return out
