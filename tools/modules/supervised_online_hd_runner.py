from __future__ import annotations

import copy
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from modules.online_state_io import OnlineStateIO
from pcdet.models import load_data_to_gpu


@dataclass
class SupervisedOnlineHDConfig:
    stream_split: str = "val_stream"
    eval_split: str = "val_eval"
    stream_ratio: float = 1.0
    max_stream_samples: int = 0
    use_stream_prefix: bool = True
    stream_seed: int = 0

    feature_source: str = "cls"  # cls | bev
    update_mode: str = "train"   # train | retrain | both
    alpha: float = 0.02
    update_every_n_batches: int = 1
    normalize_every_updates: int = 1

    max_pos_per_class_per_batch: int = 128
    min_pos_per_class_per_batch: int = 0
    max_total_pos_per_batch: int = 0
    gt_only: bool = True

    eval_every_updates: int = 50
    metric_key: str = "recall/rcnn_0.3"
    fast_recall_only: bool = True
    final_full_eval: bool = True

    save_every_updates: int = 50
    log_every_n_batches: int = 10
    save_to_file: bool = False
    save_best_memory: bool = True
    save_last_memory: bool = True
    experiment_note: str = "supervised_stream_hd_update"


class SupervisedOnlineHDRunner:
    """
    Supervised streaming HD memory updater.

    Design:
    - Keep detector in eval mode; do not optimize normal network parameters.
    - Use GT positive anchors from dense head target assignment.
    - Update only hd_core.memory from positive anchor features and labels.
    - Evaluate periodically on a fixed holdout loader and save best/last memory.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        stream_loader,
        logger,
        cfg: Optional[Dict[str, Any]] = None,
        *,
        eval_loader=None,
        eval_fn: Optional[Callable[[torch.nn.Module, Any], float]] = None,
        output_dir: Optional[str] = None,
        state_save_prefix: str = "supervised_online_hd",
    ):
        self.model = model
        self.stream_loader = stream_loader
        self.eval_loader = eval_loader
        self.eval_fn = eval_fn
        self.logger = logger
        self.cfg = self._build_cfg(cfg or {})
        if not self.cfg.gt_only:
            raise RuntimeError("SupervisedOnlineHDRunner currently supports only gt_only=True.")
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(self.cfg.stream_seed))

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
            raise RuntimeError("SupervisedOnlineHDRunner requires model.dense_head.")

        self.hd_core = getattr(self.dense_head, "hd_core", None)
        if self.hd_core is None:
            raise RuntimeError("SupervisedOnlineHDRunner requires dense_head.hd_core.")

        self.num_anchors_per_loc = int(getattr(self.dense_head, "num_anchors_per_location", 0))
        if self.num_anchors_per_loc <= 0:
            raise RuntimeError("Invalid dense_head.num_anchors_per_location.")

        self.num_classes = int(getattr(self.dense_head, "num_class", 0))
        if self.num_classes <= 0:
            raise RuntimeError("Invalid dense_head.num_class.")

        self.step_idx = 0
        self.update_idx = 0
        self.best_metric = float("-inf")
        self.last_metric = float("-inf")
        self.best_state: Optional[Dict[str, Any]] = None

        self.pending = self._new_pending_buffer()

    def run(self, *, max_steps: int = -1):
        self.model.eval()
        self.best_state = self._snapshot_state()
        self._log("[SUP] start")
        self._log(
            f"[SUP] cfg: feature_source={self.cfg.feature_source}, update_mode={self.cfg.update_mode}, "
            f"alpha={self.cfg.alpha:.4f}, update_every_n_batches={self.cfg.update_every_n_batches}"
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
                    "selected": 0,
                    "updated_classes": 0,
                }
            )
            self._save_state(tag="baseline")
            self._log(f"[BASELINE] metric={baseline_metric:.6f} | saved baseline state")

        t0 = time.time()

        for batch in self.stream_loader:
            if max_steps > 0 and self.step_idx >= max_steps:
                break

            self.step_idx += 1
            step = self._process_one_step(batch)
            self._accumulate_pending(step)

            if self.step_idx % self.cfg.update_every_n_batches == 0:
                self._maybe_apply_pending_update(reason=f"step_interval@{self.step_idx}")

            if self.step_idx % self.cfg.log_every_n_batches == 0:
                self._log(
                    f"[STEP {self.step_idx}] pending_batches={self.pending['pending_batches']} "
                    f"pending_selected={self.pending['total_selected']} "
                    f"class_count=[{self._format_count_tensor(self.pending.get('class_count', None))}]"
                )

        self._maybe_apply_pending_update(reason="final_flush", force=True)

        if self.cfg.save_last_memory:
            self._save_state(tag="last")

        self._export_best_memory_payload()
        self._log(f"[SUP] finished | steps={self.step_idx} updates={self.update_idx} elapsed={time.time()-t0:.1f}s")

    def _process_one_step(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        load_data_to_gpu(batch_dict)

        with torch.no_grad():
            _pred, _ret = self.model(batch_dict)

        feat_bnc = self._extract_anchor_features(batch_dict)
        mode = str(self.cfg.update_mode).lower()
        logits_bnk = None
        if mode in ("retrain", "both"):
            logits_bnk = self._extract_anchor_logits(self.dense_head.forward_ret_dict)
        pos_mask_bn, pos_label_bn = self._extract_supervised_positive_targets(self.dense_head.forward_ret_dict)

        B, N, C = feat_bnc.shape
        step = {
            "total_valid": int(B * N),
            "total_selected": 0,
            "class_count": torch.zeros((self.num_classes,), device=self.device, dtype=torch.long),
            "batch_selected": [],
        }

        for b in range(B):
            pos_idx = torch.nonzero(pos_mask_bn[b], as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                continue

            pos_labels = pos_label_bn[b][pos_idx]
            pos_idx, pos_labels = self._sample_positive_anchors(pos_idx, pos_labels)
            if pos_idx.numel() == 0:
                continue

            feat_sel = feat_bnc[b][pos_idx]
            logits_sel = None if logits_bnk is None else logits_bnk[b][pos_idx]
            step["batch_selected"].append((feat_sel, pos_labels, logits_sel))
            step["total_selected"] += int(pos_idx.numel())
            step["class_count"] += self._count_by_class(pos_labels - 1, self.num_classes)

        return step

    def _extract_anchor_features(self, batch_dict: Dict[str, Any]) -> torch.Tensor:
        src = str(self.cfg.feature_source).lower()
        if src == "cls":
            feat_map = batch_dict.get("hd_cls_feat", None)
            if feat_map is None:
                raise RuntimeError("Expected batch_dict['hd_cls_feat'] for feature_source='cls'.")
        elif src == "bev":
            feat_map = batch_dict.get("spatial_features_2d", None)
            if feat_map is None:
                raise RuntimeError("Expected batch_dict['spatial_features_2d'] for feature_source='bev'.")
        else:
            raise RuntimeError(f"Unknown feature_source={self.cfg.feature_source}")

        B, C, H, W = feat_map.shape
        A = self.num_anchors_per_loc
        feat_cell = feat_map.permute(0, 2, 3, 1).contiguous()
        feat_anchor = feat_cell.unsqueeze(3).expand(B, H, W, A, C).reshape(B, H * W * A, C).contiguous()
        return feat_anchor

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
            raise RuntimeError(
                "dense_head.forward_ret_dict['box_cls_labels'] not found. "
                "Make sure target assignment is enabled during eval/forward."
            )
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

    def _maybe_apply_pending_update(self, reason: str, force: bool = False):
        if self.pending["total_selected"] <= 0 and not force:
            return
        if self.pending["total_selected"] <= 0 and force:
            self._reset_pending()
            return

        updated_classes = self._apply_supervised_update_from_pending()
        self.update_idx += 1

        self._log(
            f"[UPDATE {self.update_idx}] reason={reason} selected={self.pending['total_selected']} "
            f"updated_classes={updated_classes} alpha={self.cfg.alpha:.4f}"
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
                if self.cfg.save_best_memory:
                    self._save_state(tag="best")
            self._log(f"[UPDATE {self.update_idx}] post-eval metric={metric:.6f}, best={self.best_metric:.6f}")
        else:
            self.last_metric = float("nan")
            if not eval_enabled:
                # Without periodic eval there is no separate "best" concept, so keep
                # best_state in sync with the latest state.
                self.best_state = self._snapshot_state()

        self._append_metric_row(
            {
                "event": "update",
                "step_idx": int(self.step_idx),
                "update_idx": int(self.update_idx),
                "metric": float(metric),
                "best_metric": float(self.best_metric),
                "selected": int(self.pending["total_selected"]),
                "updated_classes": int(updated_classes),
            }
        )

        if self.cfg.save_every_updates > 0 and (self.update_idx % self.cfg.save_every_updates == 0):
            self._save_state(tag=f"upd{self.update_idx:04d}")

        self._reset_pending()

    def _apply_supervised_update_from_pending(self) -> int:
        class_touch = torch.zeros((self.num_classes,), device=self.device, dtype=torch.long)

        for feat_sel, labels_sel, logits_sel in self.pending["batch_selected"]:
            if labels_sel.numel() == 0:
                continue

            class_touch += self._count_by_class(labels_sel - 1, self.num_classes)
            mode = str(self.cfg.update_mode).lower()

            if mode in ("train", "both"):
                self.hd_core.update_train(
                    feat_mid=feat_sel,
                    labels=labels_sel,
                    selected_anchor_indices=None,
                    alpha=float(self.cfg.alpha),
                )

            if mode in ("retrain", "both"):
                if logits_sel is None:
                    raise RuntimeError("Retrain-style supervised update requires anchor logits, but logits were not extracted.")
                self.hd_core.update_retrain(
                    feat_mid=feat_sel,
                    labels=labels_sel,
                    logits_origin=logits_sel,
                    selected_anchor_indices=None,
                    alpha=float(self.cfg.alpha),
                )

        if self.cfg.normalize_every_updates > 0 and (self.update_idx + 1) % int(self.cfg.normalize_every_updates) == 0:
            with torch.no_grad():
                self.hd_core.memory.normalize_()

        updated_classes = int((class_touch > 0).sum().item())
        return updated_classes

    def _evaluate_metric(self) -> float:
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
        mem_state = {}
        for k, v in self.hd_core.memory.state_dict().items():
            if torch.is_tensor(v):
                mem_state[k] = v.detach().cpu().clone()
            else:
                mem_state[k] = copy.deepcopy(v)
        return {
            "memory_state": mem_state,
            "step_idx": int(self.step_idx),
            "update_idx": int(self.update_idx),
            "best_metric": float(self.best_metric),
            "last_metric": float(self.last_metric),
            "alpha": float(self.cfg.alpha),
        }

    def _save_state(self, tag: str):
        if self.output_dir is None:
            return
        payload = OnlineStateIO.build_payload(
            hd_core=self.hd_core,
            alpha=self.cfg.alpha,
            step_idx=self.step_idx,
            update_idx=self.update_idx,
            best_metric=self.best_metric,
            last_metric=self.last_metric,
            best_state=self.best_state,
            online_cfg=self.cfg.__dict__.copy(),
            extra={
                "runner_type": "supervised_online_hd_runner",
                "experiment_note": self.cfg.experiment_note,
                "metric_key": self.cfg.metric_key,
            },
        )
        path = self.output_dir / f"{self.state_save_prefix}_{tag}.pth"
        OnlineStateIO.save(str(path), payload)
        self._log(f"[STATE] saved: {str(path)}")

    def _export_best_memory_payload(self):
        if self.output_dir is None or self.best_state is None:
            return
        mem_state = self.best_state.get("memory_state", None)
        if not isinstance(mem_state, dict):
            return

        payload: Dict[str, Any] = {
            "memory": copy.deepcopy(mem_state),
            "meta": {
                "source": "supervised_online_hd_runner.best_state",
                "best_metric": float(self.best_metric),
                "step_idx": int(self.best_state.get("step_idx", self.step_idx)),
                "update_idx": int(self.best_state.get("update_idx", self.update_idx)),
                "metric_key": str(self.cfg.metric_key),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        embedder = getattr(self.hd_core, "embedder", None)
        if embedder is not None:
            emb_state = {}
            for k, v in embedder.state_dict().items():
                if torch.is_tensor(v):
                    emb_state[k] = v.detach().cpu().clone()
                else:
                    emb_state[k] = copy.deepcopy(v)
            payload["embedder"] = emb_state

        for k in ("classify_weights", "prototypes", "bg_weight", "bg_prototype"):
            if k in mem_state:
                v = mem_state[k]
                payload[k] = v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)

        path = self.output_dir / "best_supervised_hd_memory.pth"
        OnlineStateIO.save(str(path), payload)
        self._log(f"[BEST] exported best memory payload: {str(path)}")

    def _new_pending_buffer(self) -> Dict[str, Any]:
        return {
            "pending_batches": 0,
            "total_valid": 0,
            "total_selected": 0,
            "class_count": None,
            "batch_selected": [],
        }

    def _accumulate_pending(self, step: Dict[str, Any]):
        self.pending["pending_batches"] += 1
        self.pending["total_valid"] += int(step["total_valid"])
        self.pending["total_selected"] += int(step["total_selected"])

        if self.pending["class_count"] is None:
            self.pending["class_count"] = step["class_count"].detach().clone()
        else:
            self.pending["class_count"] += step["class_count"]

        self.pending["batch_selected"].extend(step["batch_selected"])

    def _reset_pending(self):
        self.pending = self._new_pending_buffer()

    @staticmethod
    def _count_by_class(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        out = torch.zeros((num_classes,), device=labels.device, dtype=torch.long)
        if labels.numel() == 0:
            return out
        ones = torch.ones_like(labels, dtype=torch.long)
        out.index_add_(0, labels.long(), ones)
        return out

    @staticmethod
    def _format_count_tensor(x: Optional[torch.Tensor]) -> str:
        if x is None:
            return "None"
        return ",".join([str(int(v)) for v in x.detach().cpu().tolist()])

    @staticmethod
    def _build_cfg(cfg_in: Dict[str, Any]) -> SupervisedOnlineHDConfig:
        cfg = SupervisedOnlineHDConfig()
        for k, v in cfg_in.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        cfg.stream_ratio = float(max(0.0, min(1.0, cfg.stream_ratio)))
        cfg.max_stream_samples = int(max(0, cfg.max_stream_samples))
        cfg.use_stream_prefix = bool(cfg.use_stream_prefix)
        cfg.stream_seed = int(cfg.stream_seed)

        cfg.feature_source = str(cfg.feature_source).lower()
        if cfg.feature_source not in ("cls", "bev"):
            cfg.feature_source = "cls"

        cfg.update_mode = str(cfg.update_mode).lower()
        if cfg.update_mode not in ("train", "retrain", "both"):
            cfg.update_mode = "train"

        cfg.alpha = float(max(0.0, cfg.alpha))
        cfg.update_every_n_batches = int(max(1, cfg.update_every_n_batches))
        cfg.normalize_every_updates = int(max(0, cfg.normalize_every_updates))

        cfg.max_pos_per_class_per_batch = int(max(0, cfg.max_pos_per_class_per_batch))
        cfg.min_pos_per_class_per_batch = int(max(0, cfg.min_pos_per_class_per_batch))
        cfg.max_total_pos_per_batch = int(max(0, cfg.max_total_pos_per_batch))
        cfg.gt_only = bool(cfg.gt_only)

        cfg.eval_every_updates = int(max(0, cfg.eval_every_updates))
        cfg.metric_key = str(cfg.metric_key)
        cfg.fast_recall_only = bool(cfg.fast_recall_only)
        cfg.final_full_eval = bool(cfg.final_full_eval)

        cfg.save_every_updates = int(max(0, cfg.save_every_updates))
        cfg.log_every_n_batches = int(max(1, cfg.log_every_n_batches))
        cfg.save_to_file = bool(cfg.save_to_file)
        cfg.save_best_memory = bool(cfg.save_best_memory)
        cfg.save_last_memory = bool(cfg.save_last_memory)
        cfg.experiment_note = str(cfg.experiment_note)
        return cfg

    def _append_metric_row(self, row: Dict[str, Any]):
        if self.metrics_csv_path is None:
            return
        fieldnames = [
            "event", "step_idx", "update_idx", "metric", "best_metric", "selected", "updated_classes"
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
