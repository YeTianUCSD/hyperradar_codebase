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
            out[k] = [
                _state_dict_to_cpu(x) if isinstance(x, dict)
                else x.detach().cpu().clone() if torch.is_tensor(x)
                else copy.deepcopy(x)
                for x in v
            ]
        elif isinstance(v, tuple):
            out[k] = tuple(
                _state_dict_to_cpu(x) if isinstance(x, dict)
                else x.detach().cpu().clone() if torch.is_tensor(x)
                else copy.deepcopy(x)
                for x in v
            )
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class OnlineAllConfig:
    stream_split: str = "test"
    eval_split: str = "val"
    stream_ratio: float = 1.0
    max_stream_samples: int = 0
    use_stream_prefix: bool = True
    stream_seed: int = 0

    train_conv_cls_pre: bool = True
    train_conv_cls_out: bool = True
    train_hd_embedder: bool = False
    update_hd_memory: bool = True
    freeze_vfe: bool = True
    freeze_map_to_bev: bool = True
    freeze_backbone_2d: bool = True
    freeze_box_head: bool = True
    freeze_dir_head: bool = True

    pseudo_logits_source: str = "origin"
    tau_prob: float = 0.40
    tau_margin: float = 0.08
    min_tau_prob: float = 0.15
    min_tau_margin: float = 0.01
    select_top_ratio: float = 0.005
    select_min_k: int = 64
    select_max_k: int = 256
    use_teacher: bool = True
    teacher_momentum: float = 0.995
    consistency_bonus: float = 0.10
    inconsistency_penalty: float = 0.05
    use_hd_consistency: bool = True

    loss_cnn_weight: float = 1.0
    loss_hd_weight: float = 0.5
    grad_clip: float = 10.0

    update_every_n_steps: int = 32
    min_selected_anchors: int = 1024
    min_confpass_anchors: int = 256

    feature_source: str = "cls"
    update_mode: str = "train"
    alpha: float = 0.005
    normalize_every_updates: int = 1
    source_pullback_lambda: float = 0.0

    max_per_class_per_step: int = 0
    min_per_class_for_update: int = 64
    class_balance_enable: bool = True
    class_balance_min_classes_to_update: int = 2
    class_balance_max_pending_steps: int = 0

    eval_every_updates: int = 1
    metric_key: str = "recall/rcnn_0.3"
    guard_max_drop: float = 0.01
    guard_use_best: bool = True

    save_every_updates: int = 5
    log_interval_steps: int = 10
    save_best_model: bool = True
    save_last_model: bool = True
    save_to_file: bool = False
    zero_update_tau_prob_step: float = 0.02
    zero_update_tau_margin_step: float = 0.005
    experiment_note: str = "unsupervised_stream_all_cls_hd_update"


class OnlineAllRunner:
    """
    Unsupervised streaming updater for ALL-cls+HD.

    This runner does not call dense_head.forward in training mode, because the
    unlabeled stream may not contain gt_boxes and training-mode dense head would
    run target assignment. Instead it forwards the frozen feature extractor,
    computes CNN logits with conv_cls_pre/conv_cls_out, and computes HD logits
    manually through hd_core so gradients can still flow into the online
    classification path.
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
        eval_fn: Optional[Callable[[torch.nn.Module, Any], Any]] = None,
        output_dir: Optional[str] = None,
        state_save_prefix: str = "online_all",
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
            raise RuntimeError("OnlineAllRunner requires model.dense_head.")
        self.hd_core = getattr(self.dense_head, "hd_core", None)
        if self.hd_core is None:
            raise RuntimeError("OnlineAllRunner requires dense_head.hd_core.")
        if not hasattr(self.dense_head, "conv_cls_pre") or not hasattr(self.dense_head, "conv_cls_out"):
            raise RuntimeError("OnlineAllRunner requires dense_head.conv_cls_pre and conv_cls_out.")

        self.num_classes = int(getattr(self.dense_head, "num_class", 0))
        self.num_anchors_per_loc = int(getattr(self.dense_head, "num_anchors_per_location", 0))
        if self.num_classes <= 0 or self.num_anchors_per_loc <= 0:
            raise RuntimeError("Invalid dense head class/anchor settings.")

        self.teacher_cls_pre = copy.deepcopy(self.dense_head.conv_cls_pre).to(self.device)
        self.teacher_cls_out = copy.deepcopy(self.dense_head.conv_cls_out).to(self.device)
        self.teacher_cls_pre.eval()
        self.teacher_cls_out.eval()
        for p in list(self.teacher_cls_pre.parameters()) + list(self.teacher_cls_out.parameters()):
            p.requires_grad_(False)

        self.source_memory_state = self._snapshot_memory_state()

        self.step_idx = 0
        self.update_idx = 0
        self.pending = self._new_pending_buffer()
        self.best_metric = float("-inf")
        self.last_metric = float("-inf")
        self.best_state: Optional[Dict[str, Any]] = None

    def run(self, *, max_steps: int = -1):
        self._set_online_mode()
        self.optimizer.zero_grad(set_to_none=True)
        self.best_state = self._snapshot_state()
        self._log("[ALL-UNSUP] start")
        self._log(
            f"[ALL-UNSUP] cfg: loss_cnn={self.cfg.loss_cnn_weight:.4f}, "
            f"loss_hd={self.cfg.loss_hd_weight:.4f}, use_teacher={self.cfg.use_teacher}, "
            f"use_hd_consistency={self.cfg.use_hd_consistency}, update_hd_memory={self.cfg.update_hd_memory}, "
            f"alpha={self.cfg.alpha:.6f}, update_every_n_steps={self.cfg.update_every_n_steps}"
        )

        if self.eval_fn is not None and self.eval_loader is not None:
            baseline_metric = self._evaluate_metric()
            self.last_metric = baseline_metric
            self.best_metric = baseline_metric
            self.best_state = self._snapshot_state()
            self._append_metric_row(
                {
                    "event": "baseline_eval",
                    "step_idx": self.step_idx,
                    "update_idx": self.update_idx,
                    "metric": baseline_metric,
                    "best_metric": self.best_metric,
                    "rolled_back": 0,
                    "loss": "",
                    "loss_cnn": "",
                    "loss_hd": "",
                    "grad_norm": "",
                    "selected": 0,
                    "conf_pass": 0,
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

            if self.step_idx % self.cfg.update_every_n_steps == 0:
                self._maybe_update_pending(reason=f"step_interval@{self.step_idx}")

            if self.step_idx % self.cfg.log_interval_steps == 0:
                self._log_pending(prefix=f"[STEP {self.step_idx}]")

        self._maybe_update_pending(reason="final_flush", force=True)

        if self.cfg.save_last_model:
            self._save_state(tag="last")
        self._export_best_model_payload()
        self._log(f"[ALL-UNSUP] finished | steps={self.step_idx} updates={self.update_idx} elapsed={time.time()-t0:.1f}s")

    def _process_one_step(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        self._set_online_mode()
        load_data_to_gpu(batch_dict)

        batch_dict = self._forward_feature_modules_only(batch_dict)
        spatial = batch_dict.get("spatial_features_2d", None)
        if spatial is None:
            raise RuntimeError("Expected batch_dict['spatial_features_2d'] after feature modules.")

        cls_feat = self.dense_head.conv_cls_pre(spatial)
        batch_dict["hd_cls_feat"] = cls_feat
        origin_logits_bnk = self._compute_origin_logits(cls_feat)
        with torch.no_grad():
            hd_logits_bnk = self._compute_hd_logits(batch_dict)
        teacher_logits_bnk = self._compute_teacher_logits(spatial) if self.cfg.use_teacher else None
        feature_map_for_update = self._get_feature_map_for_update(batch_dict)
        origin_logits_for_select = origin_logits_bnk.detach()

        B, N, K = origin_logits_bnk.shape
        step = {
            "total_valid": int(B * N),
            "total_selected": 0,
            "total_conf_pass": 0,
            "total_consistent": 0,
            "class_count": torch.zeros((K,), device=self.device, dtype=torch.long),
            "top1_count": torch.zeros((K,), device=self.device, dtype=torch.long),
            "cand_count": torch.zeros((K,), device=self.device, dtype=torch.long),
            "agree_count": torch.zeros((K,), device=self.device, dtype=torch.long),
            "selected_count": torch.zeros((K,), device=self.device, dtype=torch.long),
            "loss": 0.0,
            "loss_cnn": 0.0,
            "loss_hd": 0.0,
            "batch_selected": [],
        }

        loss_terms: List[torch.Tensor] = []
        loss_cnn_terms: List[torch.Tensor] = []
        loss_hd_terms: List[torch.Tensor] = []
        for b in range(B):
            logits_s = self._get_pseudo_source_logits(origin_logits_for_select[b], hd_logits_bnk[b])
            logits_t = None if teacher_logits_bnk is None else teacher_logits_bnk[b]
            logits_hd = hd_logits_bnk[b]

            with torch.no_grad():
                sel_idx, pseudo_k, stat = self._select_pseudo_indices(logits_s, logits_t, logits_hd)
            step["total_conf_pass"] += stat["conf_pass"]
            step["total_consistent"] += stat["consistent"]
            step["top1_count"] += stat["top1_count"]
            step["cand_count"] += stat["cand_count"]
            step["agree_count"] += stat["agree_count"]
            step["selected_count"] += stat["selected_count"]

            if sel_idx.numel() == 0:
                continue

            sel_idx, pseudo_k = self._apply_class_cap(sel_idx, pseudo_k)
            if sel_idx.numel() == 0:
                continue

            feat_sel = self._gather_selected_anchor_features(feature_map_for_update, b, sel_idx)
            logits_hd_sel = self._compute_hd_logits_for_selected(feat_sel, sel_idx)
            loss_cnn = F.cross_entropy(origin_logits_bnk[b][sel_idx].float(), pseudo_k.long(), reduction="mean")
            loss_hd = F.cross_entropy(logits_hd_sel.float(), pseudo_k.long(), reduction="mean")
            loss = float(self.cfg.loss_cnn_weight) * loss_cnn + float(self.cfg.loss_hd_weight) * loss_hd
            loss_terms.append(loss)
            loss_cnn_terms.append(loss_cnn.detach())
            loss_hd_terms.append(loss_hd.detach())

            step["total_selected"] += int(sel_idx.numel())
            step["class_count"].index_add_(0, pseudo_k, torch.ones_like(pseudo_k, dtype=torch.long))
            labels_1based = pseudo_k.detach().long() + 1
            step["batch_selected"].append(
                (
                    feat_sel.detach(),
                    labels_1based,
                    logits_hd_sel.detach(),
                )
            )

        if loss_terms:
            loss = torch.stack(loss_terms).mean()
            loss.backward()
            step["loss"] = float(loss.detach().item())
            step["loss_cnn"] = float(torch.stack(loss_cnn_terms).mean().item())
            step["loss_hd"] = float(torch.stack(loss_hd_terms).mean().item())
        else:
            step["loss"] = 0.0

        return step

    def _compute_origin_logits(self, cls_feat: torch.Tensor) -> torch.Tensor:
        logits = self.dense_head.conv_cls_out(cls_feat)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        B, H, W, AK = logits.shape
        A, K = self.num_anchors_per_loc, self.num_classes
        if AK != A * K:
            raise RuntimeError(f"Unexpected origin logits last dim: got {AK}, expected {A*K}")
        return logits.reshape(B, H * W * A, K).float()

    def _compute_hd_logits(self, batch_dict: Dict[str, Any]) -> torch.Tensor:
        feat_map = batch_dict["hd_cls_feat"] if self.cfg.feature_source == "cls" else batch_dict["spatial_features_2d"]
        B, C, H, W = feat_map.shape
        A, K = self.num_anchors_per_loc, self.num_classes
        cell_feat = feat_map.permute(0, 2, 3, 1).contiguous()
        feat_anchor = cell_feat.unsqueeze(3).expand(B, H, W, A, C).reshape(-1, C)
        anchor_ids = torch.arange(A, device=feat_anchor.device).view(1, 1, 1, A).expand(B, H, W, A).reshape(-1)
        feat_anchor = self.hd_core.inject_anchor_context(feat_anchor, anchor_ids, num_anchors=A)
        logits, _ = self.hd_core.compute_hd_logits(feat_anchor)
        return logits.view(B, H * W * A, K).float()

    def _compute_hd_logits_for_selected(self, feat_sel: torch.Tensor, sel_idx: torch.Tensor) -> torch.Tensor:
        if sel_idx.numel() == 0:
            return feat_sel.new_empty((0, self.num_classes))
        A = self.num_anchors_per_loc
        anchor_ids = torch.remainder(sel_idx, A).to(feat_sel.device)
        feat_sel = self.hd_core.inject_anchor_context(feat_sel, anchor_ids, num_anchors=A)
        logits, _ = self.hd_core.compute_hd_logits(feat_sel)
        return logits.float()

    @torch.no_grad()
    def _compute_teacher_logits(self, spatial_features_2d: torch.Tensor) -> torch.Tensor:
        cls_feat = self.teacher_cls_pre(spatial_features_2d)
        logits = self.teacher_cls_out(cls_feat)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        B, H, W, AK = logits.shape
        A, K = self.num_anchors_per_loc, self.num_classes
        if AK != A * K:
            raise RuntimeError(f"Unexpected teacher logits last dim: got {AK}, expected {A*K}")
        return logits.reshape(B, H * W * A, K).float()

    def _get_pseudo_source_logits(self, origin: torch.Tensor, hd: torch.Tensor) -> torch.Tensor:
        src = str(self.cfg.pseudo_logits_source).lower()
        if src == "hd":
            return hd
        return origin

    def _select_pseudo_indices(
        self,
        logits_s: torch.Tensor,
        logits_t: Optional[torch.Tensor],
        logits_hd: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        probs_s = torch.sigmoid(logits_s)
        top_s = torch.topk(probs_s, k=2 if probs_s.shape[1] >= 2 else 1, dim=1)
        conf_s = top_s.values[:, 0]
        margin_s = top_s.values[:, 0] - top_s.values[:, 1] if top_s.values.shape[1] == 2 else conf_s
        cls_s = top_s.indices[:, 0].long()

        cand = (conf_s >= self.cfg.tau_prob) & (margin_s >= self.cfg.tau_margin)
        consistent = torch.zeros_like(cand)
        score = conf_s + margin_s

        if logits_t is not None:
            probs_t = torch.sigmoid(logits_t)
            top_t = torch.topk(probs_t, k=2 if probs_t.shape[1] >= 2 else 1, dim=1)
            conf_t = top_t.values[:, 0]
            margin_t = top_t.values[:, 0] - top_t.values[:, 1] if top_t.values.shape[1] == 2 else conf_t
            cls_t = top_t.indices[:, 0].long()
            cand_t = (conf_t >= self.cfg.tau_prob) & (margin_t >= self.cfg.tau_margin)
            consistent = cls_s == cls_t
            cand = cand & cand_t & consistent
            score = 0.5 * (conf_s + conf_t) + 0.5 * (margin_s + margin_t)
            score = score + self.cfg.consistency_bonus * consistent.float()
            score = score - self.cfg.inconsistency_penalty * (~consistent).float()

        if self.cfg.use_hd_consistency and logits_hd is not None:
            cls_hd = torch.sigmoid(logits_hd).argmax(dim=1).long()
            hd_agree = cls_hd == cls_s
            cand = cand & hd_agree
            consistent = consistent | hd_agree if logits_t is None else consistent & hd_agree

        top1_count = self._count_by_class(cls_s, self.num_classes)
        conf_pass = int(cand.sum().item())
        cand_count = self._count_by_class(cls_s[cand], self.num_classes)
        agree_mask = cand & consistent if (logits_t is not None or self.cfg.use_hd_consistency) else cand
        agree_count = self._count_by_class(cls_s[agree_mask], self.num_classes)
        consistent_cnt = int(agree_mask.sum().item())

        cand_idx = torch.nonzero(cand, as_tuple=False).squeeze(1)
        if cand_idx.numel() == 0:
            empty = cand_idx.new_empty((0,), dtype=torch.long)
            return empty, empty, {
                "conf_pass": conf_pass,
                "consistent": consistent_cnt,
                "top1_count": top1_count,
                "cand_count": cand_count,
                "agree_count": agree_count,
                "selected_count": torch.zeros((self.num_classes,), device=logits_s.device, dtype=torch.long),
            }

        k = max(self.cfg.select_min_k, int(float(cand_idx.numel()) * self.cfg.select_top_ratio))
        k = min(self.cfg.select_max_k, int(cand_idx.numel()))
        sel_idx = self._select_indices_balanced(cand_idx, cls_s[cand_idx], score[cand_idx], k)
        pseudo_k = cls_s[sel_idx]
        return sel_idx, pseudo_k, {
            "conf_pass": conf_pass,
            "consistent": consistent_cnt,
            "top1_count": top1_count,
            "cand_count": cand_count,
            "agree_count": agree_count,
            "selected_count": self._count_by_class(pseudo_k, self.num_classes),
        }

    def _maybe_update_pending(self, reason: str, force: bool = False):
        total_sel = int(self.pending["total_selected"])
        total_conf = int(self.pending["total_conf_pass"])
        total_valid = int(self.pending["total_valid"])
        pending_steps = int(self.pending["pending_steps"])

        if not force:
            if total_sel < self.cfg.min_selected_anchors:
                self._log(
                    f"[PENDING] wait ({reason}) selected={total_sel}/{self.cfg.min_selected_anchors}, "
                    f"conf_pass={total_conf}/{self.cfg.min_confpass_anchors}, valid={total_valid}"
                )
                return
            if total_conf < self.cfg.min_confpass_anchors:
                self._log(
                    f"[PENDING] wait-lowconf ({reason}) selected={total_sel}, "
                    f"conf_pass={total_conf}/{self.cfg.min_confpass_anchors}, valid={total_valid}"
                )
                return

        if self.cfg.class_balance_enable and self.pending["class_count"] is not None:
            req = max(1, min(self.num_classes, int(self.cfg.class_balance_min_classes_to_update)))
            ready = self.pending["class_count"] >= int(self.cfg.min_per_class_for_update)
            n_ready = int(ready.sum().item())
            if n_ready < req:
                timeout = self.cfg.class_balance_max_pending_steps > 0 and pending_steps >= self.cfg.class_balance_max_pending_steps
                if not timeout:
                    if force:
                        self.optimizer.zero_grad(set_to_none=True)
                        self._log(
                            f"[PENDING] class-balance final-drop ({reason}) ready_classes={n_ready}/{req} "
                            f"pending_steps={pending_steps} class_count=[{self._format_count_tensor(self.pending['class_count'])}]"
                        )
                        self._reset_pending()
                        return
                    self._log(
                        f"[PENDING] class-balance wait ({reason}) ready_classes={n_ready}/{req} "
                        f"pending_steps={pending_steps} class_count=[{self._format_count_tensor(self.pending['class_count'])}]"
                    )
                    return
                self._log(f"[PENDING] class-balance timeout ({reason}) ready_classes={n_ready}/{req}, force update")

        if total_sel <= 0:
            old_p, old_m = self.cfg.tau_prob, self.cfg.tau_margin
            self.cfg.tau_prob = max(self.cfg.min_tau_prob, self.cfg.tau_prob - self.cfg.zero_update_tau_prob_step)
            self.cfg.tau_margin = max(self.cfg.min_tau_margin, self.cfg.tau_margin - self.cfg.zero_update_tau_margin_step)
            self.optimizer.zero_grad(set_to_none=True)
            self._log(
                f"[PENDING] zero selected ({reason}), relax tau_prob {old_p:.3f}->{self.cfg.tau_prob:.3f}, "
                f"tau_margin {old_m:.3f}->{self.cfg.tau_margin:.3f}"
            )
            self._reset_pending()
            return

        self.update_idx += 1
        snapshot = self._snapshot_state()
        grad_norm = self._compute_grad_norm()
        if self.cfg.grad_clip > 0:
            clip_grad_norm_(self._trainable_parameters(), max_norm=float(self.cfg.grad_clip))

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        updated_classes = self._apply_hd_memory_update() if self.cfg.update_hd_memory else self._ready_class_count()
        self._apply_source_pullback()
        self._update_teacher_from_student()
        self._set_online_mode()

        avg_loss = float(self.pending["loss_sum"]) / max(1, int(self.pending["loss_count"]))
        avg_loss_cnn = float(self.pending["loss_cnn_sum"]) / max(1, int(self.pending["loss_count"]))
        avg_loss_hd = float(self.pending["loss_hd_sum"]) / max(1, int(self.pending["loss_count"]))
        self._log(
            f"[UPDATE {self.update_idx}] reason={reason} selected={total_sel}/{total_valid} "
            f"conf_pass={total_conf} updated_classes={updated_classes} "
            f"avg_loss={avg_loss:.6f} loss_cnn={avg_loss_cnn:.6f} loss_hd={avg_loss_hd:.6f} "
            f"grad_norm={grad_norm:.6f}"
        )

        rolled_back = False
        metric = float("nan")
        if (
            self.cfg.eval_every_updates > 0
            and self.eval_fn is not None
            and self.eval_loader is not None
            and self.update_idx % self.cfg.eval_every_updates == 0
        ):
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
            if self.best_state is None:
                self.best_state = self._snapshot_state()

        self._append_metric_row(
            {
                "event": "update",
                "step_idx": self.step_idx,
                "update_idx": self.update_idx,
                "metric": metric,
                "best_metric": self.best_metric,
                "rolled_back": int(rolled_back),
                "loss": avg_loss,
                "loss_cnn": avg_loss_cnn,
                "loss_hd": avg_loss_hd,
                "grad_norm": grad_norm,
                "selected": total_sel,
                "conf_pass": total_conf,
                "updated_classes": updated_classes,
            }
        )

        if self.cfg.save_every_updates > 0 and self.update_idx % self.cfg.save_every_updates == 0:
            self._save_state(tag=f"upd{self.update_idx:04d}")
        self._reset_pending()

    def _apply_hd_memory_update(self) -> int:
        class_touch = torch.zeros((self.num_classes,), device=self.device, dtype=torch.long)
        mode = str(self.cfg.update_mode).lower()
        for feat_sel, labels_sel, logits_sel in self.pending["batch_selected"]:
            if labels_sel.numel() == 0:
                continue
            class_touch += self._count_by_class(labels_sel - 1, self.num_classes)
            if mode in ("train", "both"):
                self.hd_core.update_train(feat_sel, labels_sel, selected_anchor_indices=None, alpha=float(self.cfg.alpha))
            if mode in ("retrain", "both"):
                self.hd_core.update_retrain(feat_sel, labels_sel, logits_sel, selected_anchor_indices=None, alpha=float(self.cfg.alpha))
        if self.cfg.normalize_every_updates > 0 and self.update_idx % int(self.cfg.normalize_every_updates) == 0:
            with torch.no_grad():
                self.hd_core.memory.normalize_()
        return int((class_touch > 0).sum().item())

    @torch.no_grad()
    def _apply_source_pullback(self):
        lam = float(self.cfg.source_pullback_lambda)
        if lam <= 0.0 or self.source_memory_state is None:
            return
        mem = self.hd_core.memory
        src = self.source_memory_state
        for name in ("classify_weights", "bg_weight"):
            if hasattr(mem, name) and name in src:
                cur = getattr(mem, name)
                cur.mul_(1.0 - lam).add_(src[name].to(cur.device, dtype=cur.dtype), alpha=lam)
        mem.normalize_()

    def _evaluate_metric(self) -> float:
        self._set_eval_mode()
        if self.eval_fn is None or self.eval_loader is None:
            self._set_online_mode()
            return float("nan")
        out = self.eval_fn(self.model, self.eval_loader)
        self._set_online_mode()
        if isinstance(out, dict):
            key = str(self.cfg.metric_key)
            if key in out:
                return float(out[key])
            for k in ("recall/rcnn_0.3", "recall/rcnn_0.5", "mAP", "map"):
                if k in out:
                    return float(out[k])
            raise RuntimeError(f"eval_fn returned dict without usable metric. keys={list(out.keys())[:20]}")
        return float(out)

    def _snapshot_state(self) -> Dict[str, Any]:
        return {
            "model_state": _state_dict_to_cpu(self.model.state_dict()),
            "optimizer_state": _state_dict_to_cpu(self.optimizer.state_dict()),
            "teacher_cls_pre_state": _state_dict_to_cpu(self.teacher_cls_pre.state_dict()),
            "teacher_cls_out_state": _state_dict_to_cpu(self.teacher_cls_out.state_dict()),
            "step_idx": int(self.step_idx),
            "update_idx": int(self.update_idx),
            "best_metric": float(self.best_metric),
            "last_metric": float(self.last_metric),
            "tau_prob": float(self.cfg.tau_prob),
            "tau_margin": float(self.cfg.tau_margin),
        }

    def _rollback_state(self, state: Dict[str, Any]):
        self.model.load_state_dict(state["model_state"], strict=False)
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.teacher_cls_pre.load_state_dict(state["teacher_cls_pre_state"], strict=False)
        self.teacher_cls_out.load_state_dict(state["teacher_cls_out_state"], strict=False)
        self.best_metric = float(state.get("best_metric", self.best_metric))
        self.last_metric = float(state.get("last_metric", self.last_metric))
        self.cfg.tau_prob = float(state.get("tau_prob", self.cfg.tau_prob))
        self.cfg.tau_margin = float(state.get("tau_margin", self.cfg.tau_margin))
        self.optimizer.zero_grad(set_to_none=True)
        self._set_online_mode()

    def _save_state(self, tag: str):
        if self.output_dir is None:
            return
        payload = self._snapshot_state()
        payload["online_cfg"] = copy.deepcopy(self.cfg.__dict__)
        payload["runner_type"] = "online_all_runner"
        payload["experiment_note"] = self.cfg.experiment_note
        payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        path = self.output_dir / f"{self.state_save_prefix}_{tag}.pth"
        torch.save(payload, path)
        self._log(f"[STATE] saved: {str(path)}")

    def _export_best_model_payload(self):
        if self.output_dir is None or self.best_state is None:
            return
        path = self.output_dir / "best_online_all_model.pth"
        torch.save(self.best_state, path)
        self._log(f"[BEST] exported best ALL-cls+HD model payload: {str(path)}")

    @torch.no_grad()
    def _update_teacher_from_student(self):
        if not self.cfg.use_teacher:
            return
        m = float(self.cfg.teacher_momentum)
        self._ema_module(self.teacher_cls_pre, self.dense_head.conv_cls_pre, m)
        self._ema_module(self.teacher_cls_out, self.dense_head.conv_cls_out, m)

    @staticmethod
    @torch.no_grad()
    def _ema_module(teacher: nn.Module, student: nn.Module, momentum: float):
        for p_t, p_s in zip(teacher.parameters(), student.parameters()):
            p_t.mul_(momentum).add_(p_s.detach(), alpha=1.0 - momentum)
        for b_t, b_s in zip(teacher.buffers(), student.buffers()):
            b_t.copy_(b_s.detach())

    def _set_eval_mode(self):
        self.model.eval()

    def _set_online_mode(self):
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
        self.teacher_cls_pre.eval()
        self.teacher_cls_out.eval()

    def _forward_feature_modules_only(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        module_list = getattr(self.model, "module_list", None)
        if module_list is None and hasattr(self.model, "module"):
            module_list = getattr(self.model.module, "module_list", None)
        if module_list is None:
            raise RuntimeError("Model has no module_list; cannot run feature-only online forward.")
        for cur_module in module_list:
            if cur_module is self.dense_head:
                break
            batch_dict = cur_module(batch_dict)
        return batch_dict

    def _get_feature_map_for_update(self, batch_dict: Dict[str, Any]) -> torch.Tensor:
        return batch_dict["hd_cls_feat"] if self.cfg.feature_source == "cls" else batch_dict["spatial_features_2d"]

    def _gather_selected_anchor_features(
        self,
        feat_map: torch.Tensor,
        batch_idx: int,
        sel_idx: torch.Tensor,
    ) -> torch.Tensor:
        if sel_idx.numel() == 0:
            return feat_map.new_empty((0, feat_map.shape[1]))
        _, C, H, W = feat_map.shape
        A = self.num_anchors_per_loc
        cell_idx = torch.div(sel_idx, A, rounding_mode="floor")
        max_cell = H * W
        if cell_idx.numel() > 0 and int(cell_idx.max().item()) >= max_cell:
            raise RuntimeError(
                f"Selected anchor index out of feature map range: max_cell_idx={int(cell_idx.max().item())}, "
                f"H*W={max_cell}, A={A}"
            )
        feat_cell = feat_map[batch_idx].permute(1, 2, 0).reshape(H * W, C)
        return feat_cell[cell_idx]

    def _select_indices_balanced(self, cand_idx: torch.Tensor, pseudo_k_all: torch.Tensor, score_all: torch.Tensor, k: int) -> torch.Tensor:
        if cand_idx.numel() <= k:
            return cand_idx
        per_class_locals = []
        present_classes = []
        for c in range(self.num_classes):
            idx_c = torch.nonzero(pseudo_k_all == c, as_tuple=False).squeeze(1)
            per_class_locals.append(idx_c)
            if idx_c.numel() > 0:
                present_classes.append(c)
        if len(present_classes) <= 1:
            return cand_idx[torch.topk(score_all, k=k, dim=0).indices]

        selected_locals = []
        chosen_mask = torch.zeros((cand_idx.numel(),), device=cand_idx.device, dtype=torch.bool)
        base_quota = max(1, k // len(present_classes))
        for c in present_classes:
            idx_c = per_class_locals[c]
            take = min(int(idx_c.numel()), base_quota)
            top_local = torch.topk(score_all[idx_c], k=take, dim=0).indices
            chosen = idx_c[top_local]
            selected_locals.append(chosen)
            chosen_mask[chosen] = True
        remaining = k - sum(int(x.numel()) for x in selected_locals)
        if remaining > 0:
            rem = torch.nonzero(~chosen_mask, as_tuple=False).squeeze(1)
            if rem.numel() > 0:
                selected_locals.append(rem[torch.topk(score_all[rem], k=min(remaining, int(rem.numel())), dim=0).indices])
        merged = torch.cat(selected_locals, dim=0)
        if merged.numel() > k:
            merged = merged[torch.topk(score_all[merged], k=k, dim=0).indices]
        return cand_idx[merged]

    def _apply_class_cap(self, sel_idx: torch.Tensor, pseudo_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cap = int(self.cfg.max_per_class_per_step)
        if cap <= 0:
            return sel_idx, pseudo_k
        kept = []
        for c in range(self.num_classes):
            idx_c = torch.nonzero(pseudo_k == c, as_tuple=False).squeeze(1)
            if idx_c.numel() == 0:
                continue
            if idx_c.numel() > cap:
                idx_c = idx_c[torch.randperm(idx_c.numel(), device=idx_c.device)[:cap]]
            kept.append(idx_c)
        if not kept:
            empty = sel_idx.new_empty((0,), dtype=torch.long)
            return empty, empty
        keep = torch.cat(kept, dim=0)
        return sel_idx[keep], pseudo_k[keep]

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

    def _ready_class_count(self) -> int:
        cc = self.pending.get("class_count", None)
        if cc is None:
            return 0
        return int((cc >= int(self.cfg.min_per_class_for_update)).sum().item())

    def _snapshot_memory_state(self) -> Optional[Dict[str, torch.Tensor]]:
        mem = getattr(self.hd_core, "memory", None)
        if mem is None:
            return None
        return {k: v.detach().cpu().clone() for k, v in mem.state_dict().items() if torch.is_tensor(v)}

    def _new_pending_buffer(self) -> Dict[str, Any]:
        return {
            "pending_steps": 0,
            "total_valid": 0,
            "total_selected": 0,
            "total_conf_pass": 0,
            "total_consistent": 0,
            "class_count": None,
            "top1_count": None,
            "cand_count": None,
            "agree_count": None,
            "selected_count": None,
            "loss_sum": 0.0,
            "loss_cnn_sum": 0.0,
            "loss_hd_sum": 0.0,
            "loss_count": 0,
            "batch_selected": [],
        }

    def _accumulate_pending(self, step: Dict[str, Any]):
        self.pending["pending_steps"] += 1
        self.pending["total_valid"] += int(step["total_valid"])
        self.pending["total_selected"] += int(step["total_selected"])
        self.pending["total_conf_pass"] += int(step["total_conf_pass"])
        self.pending["total_consistent"] += int(step["total_consistent"])
        self.pending["loss_sum"] += float(step.get("loss", 0.0))
        self.pending["loss_cnn_sum"] += float(step.get("loss_cnn", 0.0))
        self.pending["loss_hd_sum"] += float(step.get("loss_hd", 0.0))
        self.pending["batch_selected"].extend(step.get("batch_selected", []))
        if step.get("total_selected", 0) > 0:
            self.pending["loss_count"] += 1
        for key in ("class_count", "top1_count", "cand_count", "agree_count", "selected_count"):
            val = step.get(key, None)
            if val is None:
                continue
            if self.pending[key] is None:
                self.pending[key] = val.detach().clone()
            else:
                self.pending[key] += val.detach()

    def _reset_pending(self):
        self.pending = self._new_pending_buffer()

    def _log_pending(self, prefix: str):
        self._log(
            f"{prefix} pending_selected={self.pending['total_selected']} "
            f"pending_conf_pass={self.pending['total_conf_pass']} pending_valid={self.pending['total_valid']} "
            f"pending_steps={self.pending['pending_steps']} "
            f"class_count=[{self._format_count_tensor(self.pending.get('class_count'))}] "
            f"top1=[{self._format_count_tensor(self.pending.get('top1_count'))}] "
            f"cand=[{self._format_count_tensor(self.pending.get('cand_count'))}] "
            f"agree=[{self._format_count_tensor(self.pending.get('agree_count'))}] "
            f"selected=[{self._format_count_tensor(self.pending.get('selected_count'))}]"
        )

    @staticmethod
    def _count_by_class(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        out = torch.zeros((num_classes,), device=labels.device, dtype=torch.long)
        if labels.numel() > 0:
            out.index_add_(0, labels.long(), torch.ones_like(labels, dtype=torch.long))
        return out

    @staticmethod
    def _format_count_tensor(x: Optional[torch.Tensor]) -> str:
        if x is None:
            return "None"
        return ",".join(str(int(v)) for v in x.detach().cpu().tolist())

    @staticmethod
    def _build_cfg(cfg_in: Dict[str, Any]) -> OnlineAllConfig:
        cfg = OnlineAllConfig()
        for k, v in cfg_in.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        cfg.stream_split = str(cfg.stream_split)
        cfg.eval_split = str(cfg.eval_split)
        cfg.stream_ratio = float(max(0.0, min(1.0, cfg.stream_ratio)))
        cfg.max_stream_samples = int(max(0, cfg.max_stream_samples))
        cfg.use_stream_prefix = bool(cfg.use_stream_prefix)
        cfg.stream_seed = int(cfg.stream_seed)

        for name in (
            "train_conv_cls_pre", "train_conv_cls_out", "train_hd_embedder", "update_hd_memory",
            "freeze_vfe", "freeze_map_to_bev", "freeze_backbone_2d", "freeze_box_head", "freeze_dir_head",
            "use_teacher", "use_hd_consistency", "class_balance_enable", "guard_use_best",
            "save_best_model", "save_last_model", "save_to_file",
        ):
            setattr(cfg, name, bool(getattr(cfg, name)))

        cfg.pseudo_logits_source = str(cfg.pseudo_logits_source).lower()
        if cfg.pseudo_logits_source not in ("origin", "hd"):
            cfg.pseudo_logits_source = "origin"
        cfg.tau_prob = float(cfg.tau_prob)
        cfg.tau_margin = float(cfg.tau_margin)
        cfg.min_tau_prob = float(cfg.min_tau_prob)
        cfg.min_tau_margin = float(cfg.min_tau_margin)
        cfg.select_top_ratio = float(max(0.0, cfg.select_top_ratio))
        cfg.select_min_k = int(max(0, cfg.select_min_k))
        cfg.select_max_k = int(max(1, cfg.select_max_k))
        cfg.teacher_momentum = float(max(0.0, min(1.0, cfg.teacher_momentum)))
        cfg.consistency_bonus = float(cfg.consistency_bonus)
        cfg.inconsistency_penalty = float(cfg.inconsistency_penalty)

        cfg.loss_cnn_weight = float(max(0.0, cfg.loss_cnn_weight))
        cfg.loss_hd_weight = float(max(0.0, cfg.loss_hd_weight))
        cfg.grad_clip = float(max(0.0, cfg.grad_clip))
        cfg.update_every_n_steps = int(max(1, cfg.update_every_n_steps))
        cfg.min_selected_anchors = int(max(0, cfg.min_selected_anchors))
        cfg.min_confpass_anchors = int(max(0, cfg.min_confpass_anchors))

        cfg.feature_source = str(cfg.feature_source).lower()
        if cfg.feature_source not in ("cls", "bev"):
            cfg.feature_source = "cls"
        cfg.update_mode = str(cfg.update_mode).lower()
        if cfg.update_mode not in ("train", "retrain", "both"):
            cfg.update_mode = "train"
        cfg.alpha = float(max(0.0, cfg.alpha))
        cfg.normalize_every_updates = int(max(0, cfg.normalize_every_updates))
        cfg.source_pullback_lambda = float(max(0.0, min(1.0, cfg.source_pullback_lambda)))

        cfg.max_per_class_per_step = int(max(0, cfg.max_per_class_per_step))
        cfg.min_per_class_for_update = int(max(0, cfg.min_per_class_for_update))
        cfg.class_balance_min_classes_to_update = int(max(0, cfg.class_balance_min_classes_to_update))
        cfg.class_balance_max_pending_steps = int(max(0, cfg.class_balance_max_pending_steps))

        cfg.eval_every_updates = int(max(0, cfg.eval_every_updates))
        cfg.metric_key = str(cfg.metric_key)
        cfg.guard_max_drop = float(max(0.0, cfg.guard_max_drop))
        cfg.save_every_updates = int(max(0, cfg.save_every_updates))
        cfg.log_interval_steps = int(max(1, cfg.log_interval_steps))
        cfg.zero_update_tau_prob_step = float(max(0.0, cfg.zero_update_tau_prob_step))
        cfg.zero_update_tau_margin_step = float(max(0.0, cfg.zero_update_tau_margin_step))
        cfg.experiment_note = str(cfg.experiment_note)
        return cfg

    def _append_metric_row(self, row: Dict[str, Any]):
        if self.metrics_csv_path is None:
            return
        fieldnames = [
            "event", "step_idx", "update_idx", "metric", "best_metric", "rolled_back",
            "loss", "loss_cnn", "loss_hd", "grad_norm", "selected", "conf_pass", "updated_classes",
        ]
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
