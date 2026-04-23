from __future__ import annotations

import copy
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
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
class SupervisedOnlineAllConfig:
    stream_split: str = "val_stream_by_scene"
    eval_split: str = "val_eval_by_scene"
    stream_ratio: float = 1.0
    max_stream_samples: int = 0
    use_stream_prefix: bool = True
    stream_seed: int = 0

    train_conv_cls_pre: bool = True
    train_conv_cls_out: bool = True
    train_hd_embedder: bool = True
    update_hd_memory: bool = True
    freeze_vfe: bool = True
    freeze_map_to_bev: bool = True
    freeze_backbone_2d: bool = True
    freeze_box_head: bool = True
    freeze_dir_head: bool = True

    loss_hd_weight: float = 1.0
    loss_cnn_weight: float = 0.5
    update_every_n_batches: int = 1
    grad_clip: float = 10.0

    feature_source: str = "cls"
    update_mode: str = "train"
    alpha: float = 1.0
    normalize_every_updates: int = 1

    max_pos_per_class_per_batch: int = 4096
    min_pos_per_class_per_batch: int = 0
    max_total_pos_per_batch: int = 0

    eval_every_updates: int = 10
    metric_key: str = "recall/rcnn_0.3"
    fast_recall_only: bool = True
    final_full_eval: bool = True
    guard_max_drop: float = 0.01
    guard_use_best: bool = True

    save_every_updates: int = 50
    log_every_n_batches: int = 10
    save_to_file: bool = False
    save_best_model: bool = True
    save_last_model: bool = True
    experiment_note: str = "supervised_stream_all_cls_hd_update"


class SupervisedOnlineAllRunner:
    """
    Supervised streaming updater for ALL-cls+HD.

    Trainable parameters are configured by the entry script. This runner:
    - uses GT-positive anchors from target assignment;
    - optimizes HD-path CE loss and CNN-origin CE loss;
    - updates HD memory/prototypes with the same GT-positive anchors;
    - evaluates periodically and rolls back full model + optimizer state on guard failure.
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
        state_save_prefix: str = "supervised_online_all",
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
            raise RuntimeError("SupervisedOnlineAllRunner requires model.dense_head.")
        self.hd_core = getattr(self.dense_head, "hd_core", None)
        if self.hd_core is None:
            raise RuntimeError("SupervisedOnlineAllRunner requires dense_head.hd_core.")
        self._orig_predict_boxes_when_training = bool(
            getattr(self.dense_head, "predict_boxes_when_training", False)
        )

        self.num_classes = int(getattr(self.dense_head, "num_class", 0))
        if self.num_classes <= 0:
            raise RuntimeError("Invalid dense_head.num_class.")
        self.num_anchors_per_loc = int(getattr(self.dense_head, "num_anchors_per_location", 0))
        if self.num_anchors_per_loc <= 0:
            raise RuntimeError("Invalid dense_head.num_anchors_per_location.")

        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(int(self.cfg.stream_seed))

        self.step_idx = 0
        self.update_idx = 0
        self.best_metric = float("-inf")
        self.last_metric = float("-inf")
        self.best_state: Optional[Dict[str, Any]] = None

    def run(self, *, max_steps: int = -1):
        self._set_online_train_mode()
        self.optimizer.zero_grad(set_to_none=True)
        self.best_state = self._snapshot_state()
        self._log("[ALL-SUP] start")
        self._log(
            f"[ALL-SUP] cfg: loss_hd={self.cfg.loss_hd_weight:.4f}, "
            f"loss_cnn={self.cfg.loss_cnn_weight:.4f}, update_hd_memory={self.cfg.update_hd_memory}, "
            f"update_mode={self.cfg.update_mode}, alpha={self.cfg.alpha:.4f}, "
            f"update_every_n_batches={self.cfg.update_every_n_batches}"
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
                    "rolled_back": 0,
                    "loss": "",
                    "loss_hd": "",
                    "loss_cnn": "",
                    "grad_norm": "",
                    "selected": 0,
                    "updated_classes": 0,
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
                avg_loss = accum["loss_sum"] / max(1, accum["pending_batches"])
                self._log(
                    f"[STEP {self.step_idx}] pending_batches={accum['pending_batches']} "
                    f"pending_selected={accum['selected_total']} pending_loss={avg_loss:.6f}"
                )

        self._finalize_update(accum, reason="final_flush", force=True)

        if self.cfg.save_last_model:
            self._save_state(tag="last")

        self._export_best_model_payload()
        self._log(f"[ALL-SUP] finished | steps={self.step_idx} updates={self.update_idx} elapsed={time.time()-t0:.1f}s")

    def _process_one_step(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        self._set_online_train_mode()
        load_data_to_gpu(batch_dict)

        self._forward_modules_only(batch_dict)
        if "box_cls_labels" not in self.dense_head.forward_ret_dict:
            raise RuntimeError(
                "dense_head.forward_ret_dict['box_cls_labels'] not found during ALL supervised update. "
                "This runner expects eval-time target assignment with gt_boxes, e.g. HD.ASSIGN_TARGETS_IN_EVAL=True."
            )

        origin_logits_bnk = self._extract_anchor_logits(self.dense_head.forward_ret_dict, source="origin")
        hd_logits_bnk = self._extract_anchor_logits(self.dense_head.forward_ret_dict, source="hd_or_final")
        feat_bnc = self._extract_anchor_features(batch_dict)
        pos_mask_bn, pos_label_bn = self._extract_supervised_positive_targets(self.dense_head.forward_ret_dict)

        selected_origin: List[torch.Tensor] = []
        selected_hd: List[torch.Tensor] = []
        selected_labels: List[torch.Tensor] = []
        batch_selected_for_memory: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        selected_total = 0
        class_count = torch.zeros((self.num_classes,), device=self.device, dtype=torch.long)

        B = origin_logits_bnk.shape[0]
        for b in range(B):
            pos_idx = torch.nonzero(pos_mask_bn[b], as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                continue

            pos_labels = pos_label_bn[b][pos_idx]
            pos_idx, pos_labels = self._sample_positive_anchors(pos_idx, pos_labels)
            if pos_idx.numel() == 0:
                continue

            selected_origin.append(origin_logits_bnk[b][pos_idx])
            selected_hd.append(hd_logits_bnk[b][pos_idx])
            selected_labels.append(pos_labels)

            feat_sel = feat_bnc[b][pos_idx].detach()
            logits_sel = hd_logits_bnk[b][pos_idx].detach()
            batch_selected_for_memory.append((feat_sel, pos_labels.detach(), logits_sel))

            selected_total += int(pos_idx.numel())
            class_count += self._count_by_class(pos_labels - 1, self.num_classes)

        if selected_total > 0:
            labels_sel = torch.cat(selected_labels, dim=0).long() - 1
            logits_origin_sel = torch.cat(selected_origin, dim=0)
            logits_hd_sel = torch.cat(selected_hd, dim=0)
            loss_cnn = F.cross_entropy(logits_origin_sel.float(), labels_sel, reduction="mean")
            loss_hd = F.cross_entropy(logits_hd_sel.float(), labels_sel, reduction="mean")
            loss = float(self.cfg.loss_cnn_weight) * loss_cnn + float(self.cfg.loss_hd_weight) * loss_hd
        else:
            zero = origin_logits_bnk.sum() * 0.0
            loss_cnn = zero
            loss_hd = zero
            loss = zero

        loss.backward()

        return {
            "loss": float(loss.detach().item()),
            "loss_hd": float(loss_hd.detach().item()),
            "loss_cnn": float(loss_cnn.detach().item()),
            "selected": int(selected_total),
            "updated_classes": int((class_count > 0).sum().item()),
            "class_count": class_count.detach(),
            "batch_selected": batch_selected_for_memory,
        }

    def _finalize_update(self, accum: Dict[str, Any], reason: str, force: bool = False):
        if accum["pending_batches"] <= 0:
            return
        if accum["selected_total"] <= 0:
            self.optimizer.zero_grad(set_to_none=True)
            return

        self.update_idx += 1
        snapshot = self._snapshot_state()

        avg_loss = accum["loss_sum"] / max(1, accum["pending_batches"])
        avg_loss_hd = accum["loss_hd_sum"] / max(1, accum["pending_batches"])
        avg_loss_cnn = accum["loss_cnn_sum"] / max(1, accum["pending_batches"])
        grad_norm = self._compute_grad_norm()

        if self.cfg.grad_clip > 0:
            clip_grad_norm_(self._trainable_parameters(), max_norm=float(self.cfg.grad_clip))
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        updated_classes = 0
        if self.cfg.update_hd_memory:
            updated_classes = self._apply_hd_memory_update(accum)
        else:
            updated_classes = int((accum["class_count"] > 0).sum().item()) if accum["class_count"] is not None else 0

        self._set_online_train_mode()

        self._log(
            f"[UPDATE {self.update_idx}] reason={reason} batches={accum['pending_batches']} "
            f"selected={accum['selected_total']} updated_classes={updated_classes} "
            f"loss={avg_loss:.6f} loss_hd={avg_loss_hd:.6f} loss_cnn={avg_loss_cnn:.6f} "
            f"grad_norm={grad_norm:.6f}"
        )

        rolled_back = False
        metric = float("nan")
        eval_enabled = (
            self.cfg.eval_every_updates > 0
            and self.eval_fn is not None
            and self.eval_loader is not None
        )
        should_eval_now = eval_enabled and (self.update_idx % self.cfg.eval_every_updates == 0)

        if should_eval_now:
            metric = self._evaluate_metric()
            ref = self.best_metric if self.cfg.guard_use_best else self.last_metric

            if ref != float("-inf") and (ref - metric) > self.cfg.guard_max_drop:
                self._rollback_state(snapshot)
                rolled_back = True
                self._log(
                    f"[GUARD] rollback update={self.update_idx}, metric={metric:.6f}, ref={ref:.6f}, "
                    f"drop={ref-metric:.6f} > {self.cfg.guard_max_drop:.6f}"
                )
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
                "rolled_back": int(rolled_back),
                "loss": float(avg_loss),
                "loss_hd": float(avg_loss_hd),
                "loss_cnn": float(avg_loss_cnn),
                "grad_norm": float(grad_norm),
                "selected": int(accum["selected_total"]),
                "updated_classes": int(updated_classes),
            }
        )

        if self.cfg.save_every_updates > 0 and (self.update_idx % self.cfg.save_every_updates == 0):
            self._save_state(tag=f"upd{self.update_idx:04d}")

    def _apply_hd_memory_update(self, accum: Dict[str, Any]) -> int:
        class_touch = torch.zeros((self.num_classes,), device=self.device, dtype=torch.long)
        mode = str(self.cfg.update_mode).lower()

        for feat_sel, labels_sel, logits_sel in accum["batch_selected"]:
            if labels_sel.numel() == 0:
                continue

            class_touch += self._count_by_class(labels_sel - 1, self.num_classes)

            if mode in ("train", "both"):
                self.hd_core.update_train(
                    feat_mid=feat_sel,
                    labels=labels_sel,
                    selected_anchor_indices=None,
                    alpha=float(self.cfg.alpha),
                )

            if mode in ("retrain", "both"):
                self.hd_core.update_retrain(
                    feat_mid=feat_sel,
                    labels=labels_sel,
                    logits_origin=logits_sel,
                    selected_anchor_indices=None,
                    alpha=float(self.cfg.alpha),
                )

        if self.cfg.normalize_every_updates > 0 and (self.update_idx % int(self.cfg.normalize_every_updates) == 0):
            with torch.no_grad():
                self.hd_core.memory.normalize_()

        return int((class_touch > 0).sum().item())

    def _evaluate_metric(self) -> float:
        self._set_eval_mode()
        if self.eval_fn is None or self.eval_loader is None:
            self._set_online_train_mode()
            return float("nan")
        out = self.eval_fn(self.model, self.eval_loader)
        self._set_online_train_mode()
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

    def _rollback_state(self, state: Dict[str, Any]):
        model_state = state.get("model_state", None)
        optim_state = state.get("optimizer_state", None)
        if isinstance(model_state, dict):
            self.model.load_state_dict(model_state, strict=False)
        if isinstance(optim_state, dict):
            self.optimizer.load_state_dict(optim_state)
        self.optimizer.zero_grad(set_to_none=True)
        self._set_online_train_mode()

    def _save_state(self, tag: str):
        if self.output_dir is None:
            return
        payload = self._snapshot_state()
        payload["online_cfg"] = copy.deepcopy(self.cfg.__dict__)
        payload["runner_type"] = "supervised_online_all_runner"
        payload["experiment_note"] = self.cfg.experiment_note
        payload["metric_key"] = self.cfg.metric_key
        payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        path = self.output_dir / f"{self.state_save_prefix}_{tag}.pth"
        torch.save(payload, path)
        self._log(f"[STATE] saved: {str(path)}")

    def _export_best_model_payload(self):
        if self.output_dir is None or self.best_state is None:
            return
        path = self.output_dir / "best_supervised_all_model.pth"
        torch.save(self.best_state, path)
        self._log(f"[BEST] exported best ALL-cls+HD checkpoint: {str(path)}")

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

    def _set_eval_mode(self):
        self.model.eval()
        if hasattr(self.dense_head, "predict_boxes_when_training"):
            self.dense_head.predict_boxes_when_training = self._orig_predict_boxes_when_training

    def _set_online_train_mode(self):
        self.model.eval()
        self.dense_head.train()
        if hasattr(self.dense_head, "predict_boxes_when_training"):
            self.dense_head.predict_boxes_when_training = False
        # Dense-head BN running statistics are kept fixed for streaming stability,
        # while dense_head.training=True keeps the HD branch differentiable.
        for m in self.dense_head.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def _forward_modules_only(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        module_list = getattr(self.model, "module_list", None)
        if module_list is None and hasattr(self.model, "module"):
            module_list = getattr(self.model.module, "module_list", None)
        if module_list is None:
            raise RuntimeError("Model has no module_list; cannot run online forward without post-processing.")

        for cur_module in module_list:
            batch_dict = cur_module(batch_dict)
        return batch_dict

    @staticmethod
    def _new_accum_buffer() -> Dict[str, Any]:
        return {
            "pending_batches": 0,
            "loss_sum": 0.0,
            "loss_hd_sum": 0.0,
            "loss_cnn_sum": 0.0,
            "selected_total": 0,
            "class_count": None,
            "batch_selected": [],
        }

    @staticmethod
    def _accumulate_step(accum: Dict[str, Any], step: Dict[str, Any]):
        accum["pending_batches"] += 1
        accum["loss_sum"] += float(step.get("loss", 0.0))
        accum["loss_hd_sum"] += float(step.get("loss_hd", 0.0))
        accum["loss_cnn_sum"] += float(step.get("loss_cnn", 0.0))
        accum["selected_total"] += int(step.get("selected", 0))
        accum["batch_selected"].extend(step.get("batch_selected", []))
        if "class_count" not in step:
            return
        if accum["class_count"] is None:
            accum["class_count"] = step["class_count"].detach().clone()
        else:
            accum["class_count"] += step["class_count"]

    @staticmethod
    def _build_cfg(cfg_in: Dict[str, Any]) -> SupervisedOnlineAllConfig:
        cfg = SupervisedOnlineAllConfig()
        for k, v in cfg_in.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        cfg.stream_split = str(cfg.stream_split)
        cfg.eval_split = str(cfg.eval_split)
        cfg.stream_ratio = float(max(0.0, min(1.0, cfg.stream_ratio)))
        cfg.max_stream_samples = int(max(0, cfg.max_stream_samples))
        cfg.use_stream_prefix = bool(cfg.use_stream_prefix)
        cfg.stream_seed = int(cfg.stream_seed)

        cfg.train_conv_cls_pre = bool(cfg.train_conv_cls_pre)
        cfg.train_conv_cls_out = bool(cfg.train_conv_cls_out)
        cfg.train_hd_embedder = bool(cfg.train_hd_embedder)
        cfg.update_hd_memory = bool(cfg.update_hd_memory)
        cfg.freeze_vfe = bool(cfg.freeze_vfe)
        cfg.freeze_map_to_bev = bool(cfg.freeze_map_to_bev)
        cfg.freeze_backbone_2d = bool(cfg.freeze_backbone_2d)
        cfg.freeze_box_head = bool(cfg.freeze_box_head)
        cfg.freeze_dir_head = bool(cfg.freeze_dir_head)

        cfg.loss_hd_weight = float(max(0.0, cfg.loss_hd_weight))
        cfg.loss_cnn_weight = float(max(0.0, cfg.loss_cnn_weight))
        cfg.update_every_n_batches = int(max(1, cfg.update_every_n_batches))
        cfg.grad_clip = float(max(0.0, cfg.grad_clip))

        cfg.feature_source = str(cfg.feature_source).lower()
        if cfg.feature_source not in ("cls", "bev"):
            cfg.feature_source = "cls"
        cfg.update_mode = str(cfg.update_mode).lower()
        if cfg.update_mode not in ("train", "retrain", "both"):
            cfg.update_mode = "train"
        cfg.alpha = float(max(0.0, cfg.alpha))
        cfg.normalize_every_updates = int(max(0, cfg.normalize_every_updates))

        cfg.max_pos_per_class_per_batch = int(max(0, cfg.max_pos_per_class_per_batch))
        cfg.min_pos_per_class_per_batch = int(max(0, cfg.min_pos_per_class_per_batch))
        cfg.max_total_pos_per_batch = int(max(0, cfg.max_total_pos_per_batch))

        cfg.eval_every_updates = int(max(0, cfg.eval_every_updates))
        cfg.metric_key = str(cfg.metric_key)
        cfg.fast_recall_only = bool(cfg.fast_recall_only)
        cfg.final_full_eval = bool(cfg.final_full_eval)
        cfg.guard_max_drop = float(max(0.0, cfg.guard_max_drop))
        cfg.guard_use_best = bool(cfg.guard_use_best)

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
            "event", "step_idx", "update_idx", "metric", "best_metric", "rolled_back",
            "loss", "loss_hd", "loss_cnn", "grad_norm", "selected", "updated_classes",
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
        return feat_cell.unsqueeze(3).expand(B, H, W, A, C).reshape(B, H * W * A, C).contiguous()

    def _extract_anchor_logits(self, forward_ret_dict: Dict[str, torch.Tensor], *, source: str) -> torch.Tensor:
        if source == "origin":
            cls_hwk = forward_ret_dict.get("cls_preds_origin", None)
            if cls_hwk is None:
                raise RuntimeError(
                    "Expected cls_preds_origin in dense_head.forward_ret_dict for CNN loss. "
                    "Use detector cfg with HD.DEBUG_SAVE_LOGITS=True."
                )
        elif source == "hd_or_final":
            cls_hwk = forward_ret_dict.get("cls_preds_hd", None)
            if cls_hwk is None:
                cls_hwk = forward_ret_dict.get("cls_preds", None)
            if cls_hwk is None:
                raise RuntimeError("Expected cls_preds_hd or cls_preds in dense_head.forward_ret_dict for HD loss.")
        else:
            raise RuntimeError(f"Unknown logits source: {source}")

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
