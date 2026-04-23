from __future__ import annotations

import copy
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

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
class OnlinePureCNNConfig:
    # Pseudo-label selection
    tau_prob: float = 0.40
    tau_margin: float = 0.08
    min_tau_prob: float = 0.15
    min_tau_margin: float = 0.01
    select_top_ratio: float = 0.005
    select_min_k: int = 64
    select_max_k: int = 256
    consistency_bonus: float = 0.10
    inconsistency_penalty: float = 0.05

    # Teacher classifier EMA
    use_teacher: bool = True
    teacher_momentum: float = 0.995

    # Update trigger
    update_every_n_steps: int = 32
    min_selected_anchors: int = 1024
    min_confpass_anchors: int = 256

    # Optimizer behavior
    grad_clip: float = 0.0

    # Class balance
    max_per_class_per_step: int = 0
    min_per_class_for_update: int = 64
    class_balance_enable: bool = True
    class_balance_min_classes_to_update: int = 2
    class_balance_max_pending_steps: int = 0

    # Guards
    eval_every_updates: int = 1
    metric_key: str = "recall/rcnn_0.3"
    guard_max_drop: float = 0.01
    guard_use_best: bool = True

    # IO
    save_every_updates: int = 5
    log_interval_steps: int = 10
    save_best_model: bool = True
    save_last_model: bool = True
    pseudo_logits_source: str = "origin"  # final | origin

    # Threshold adaption when update failed
    zero_update_tau_prob_step: float = 0.02
    zero_update_tau_margin_step: float = 0.005


class OnlinePureCNNRunner:
    """
    Unsupervised pure-CNN classifier adaptation runner.

    Design:
    - Use a NO-HD detector and adapt only the CNN classification output layer.
    - Keep the detector frozen except dense_head.conv_cls_out.
    - Select pseudo-label anchors from stream predictions.
    - Accumulate CE gradients on selected pseudo labels, then update conv_cls_out.
    - Use optional EMA teacher classifier, eval guard, rollback, and best-state save.

    Required model assumptions:
    - model.dense_head has conv_cls_pre and conv_cls_out.
    - dense_head.forward_ret_dict includes cls_preds and optionally cls_preds_origin.
    - batch_dict includes hd_cls_feat after dense_head forward. In the pure CNN
      head this name is the penultimate CNN classification feature, not HD memory.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        stream_loader,
        logger,
        cfg: Optional[Dict[str, Any]] = None,
        *,
        val_loader=None,
        eval_fn: Optional[Callable[[torch.nn.Module, Any], float]] = None,
        output_dir: Optional[str] = None,
        state_save_prefix: str = "online_pure_cnn",
        use_amp: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.stream_loader = stream_loader
        self.val_loader = val_loader
        self.eval_fn = eval_fn
        self.logger = logger
        self.use_amp = bool(use_amp)

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
            raise RuntimeError("OnlinePureCNNRunner requires model.dense_head.")
        if not hasattr(self.dense_head, "conv_cls_out"):
            raise RuntimeError("OnlinePureCNNRunner requires dense_head.conv_cls_out.")
        if not hasattr(self.dense_head, "conv_cls_pre"):
            raise RuntimeError("OnlinePureCNNRunner requires dense_head.conv_cls_pre.")

        self.num_anchors_per_loc = int(getattr(self.dense_head, "num_anchors_per_location", 0))
        if self.num_anchors_per_loc <= 0:
            raise RuntimeError("Invalid dense_head.num_anchors_per_location.")

        self.num_classes = int(getattr(self.dense_head, "num_class", 0))
        if self.num_classes <= 0:
            raise RuntimeError("Invalid dense_head.num_class.")

        self._force_baseline_mode()
        self.teacher_cls_out = copy.deepcopy(self.dense_head.conv_cls_out).to(self.device)
        self.teacher_cls_out.eval()
        for p in self.teacher_cls_out.parameters():
            p.requires_grad_(False)

        self.step_idx = 0
        self.update_idx = 0
        self.pending = self._new_pending_buffer()

        self.best_metric = float("-inf")
        self.last_metric = float("-inf")
        self.best_state = None

    # ------------------------------
    # Public APIs
    # ------------------------------
    def run(self, *, max_steps: int = -1):
        self.model.eval()
        self._force_baseline_mode()
        self.optimizer.zero_grad(set_to_none=True)
        self._log("[ONLINE-PURE-CNN] start")
        self._log(
            f"[ONLINE-PURE-CNN] cfg: pseudo_logits_source={self.cfg.pseudo_logits_source}, "
            f"use_teacher={self.cfg.use_teacher}, update_every_n_steps={self.cfg.update_every_n_steps}"
        )

        if self.eval_fn is not None and self.val_loader is not None:
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
                    "grad_norm": "",
                    "selected": 0,
                    "conf_pass": 0,
                    "updated_classes": 0,
                }
            )
            self._save_state(tag="baseline")
            self._log(f"[BASELINE] metric={baseline_metric:.6f} | saved baseline state")
        else:
            self.best_state = self._snapshot_state()

        t0 = time.time()

        for batch in self.stream_loader:
            if max_steps > 0 and self.step_idx >= max_steps:
                break

            self.step_idx += 1
            step_stats = self._process_one_step(batch)
            self._accumulate_pending(step_stats)

            if self.step_idx % self.cfg.update_every_n_steps == 0:
                self._maybe_update_pending(reason=f"step_interval@{self.step_idx}")

            if self.step_idx % self.cfg.log_interval_steps == 0:
                self._log_pending(prefix=f"[STEP {self.step_idx}]")

        self._maybe_update_pending(reason="final_flush", force=True)

        if self.cfg.save_last_model:
            self._save_state(tag="last")
        if self.cfg.save_best_model:
            self._export_best_model_payload()
        self._log(f"[ONLINE-PURE-CNN] finished | steps={self.step_idx} updates={self.update_idx} elapsed={time.time()-t0:.1f}s")

    # ------------------------------
    # Core step logic
    # ------------------------------
    def _process_one_step(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        self._force_baseline_mode()
        self.model.eval()
        load_data_to_gpu(batch_dict)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            _pred, _ret = self.model(batch_dict)

        fr = self.dense_head.forward_ret_dict
        student_logits_bnk = self._extract_student_logits(fr)
        teacher_logits_bnk = None
        if self.cfg.use_teacher:
            teacher_logits_bnk = self._compute_teacher_logits(batch_dict)

        B, N, K = student_logits_bnk.shape
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
        }

        loss_terms = []
        for b in range(B):
            logits_s = student_logits_bnk[b]
            logits_t = None if teacher_logits_bnk is None else teacher_logits_bnk[b]

            sel_idx, pseudo_k, stat = self._select_pseudo_indices(logits_s, logits_t)
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

            logits_sel = logits_s[sel_idx]
            loss_terms.append(F.cross_entropy(logits_sel.float(), pseudo_k.long(), reduction="mean"))
            step["total_selected"] += int(sel_idx.numel())
            step["class_count"].index_add_(0, pseudo_k, torch.ones_like(pseudo_k, dtype=torch.long))

        if loss_terms:
            loss = torch.stack(loss_terms).mean()
            loss.backward()
            step["loss"] = float(loss.detach().item())
        else:
            step["loss"] = 0.0

        return step

    def _extract_student_logits(self, forward_ret: Dict[str, torch.Tensor]) -> torch.Tensor:
        src = self.cfg.pseudo_logits_source.lower()
        if src == "origin" and "cls_preds_origin" in forward_ret:
            cls_hwk = forward_ret["cls_preds_origin"]
        elif src == "final":
            cls_hwk = forward_ret["cls_preds"]
        else:
            cls_hwk = forward_ret.get("cls_preds_origin", forward_ret["cls_preds"])

        if cls_hwk.dim() != 4:
            raise RuntimeError(f"Unexpected cls prediction shape: {tuple(cls_hwk.shape)}")
        B, H, W, AK = cls_hwk.shape
        A = self.num_anchors_per_loc
        K = self.num_classes
        if AK != A * K:
            raise RuntimeError(f"Unexpected cls prediction last dim: got {AK}, expected {A*K}")
        return cls_hwk.reshape(B, H * W * A, K).float()

    @torch.no_grad()
    def _compute_teacher_logits(self, batch_dict: Dict[str, Any]) -> torch.Tensor:
        feat_map = batch_dict.get("hd_cls_feat", None)
        if feat_map is None:
            raise RuntimeError(
                "Expected batch_dict['hd_cls_feat'] to compute teacher CNN logits. "
                "For pure CNN AnchorHeadSingle this field is the conv_cls_pre feature."
            )

        logits = self.teacher_cls_out(feat_map)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        B, H, W, AK = logits.shape
        A = self.num_anchors_per_loc
        K = self.num_classes
        if AK != A * K:
            raise RuntimeError(f"Unexpected teacher cls prediction last dim: got {AK}, expected {A*K}")
        return logits.reshape(B, H * W * A, K).float()

    # ------------------------------
    # Pseudo selection
    # ------------------------------
    def _select_pseudo_indices(
        self,
        logits_s: torch.Tensor,
        logits_t: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        probs_s = torch.sigmoid(logits_s)
        k_s = 2 if probs_s.shape[1] >= 2 else 1
        top_s = torch.topk(probs_s, k=k_s, dim=1)
        conf_s = top_s.values[:, 0]
        margin_s = top_s.values[:, 0] - top_s.values[:, 1] if k_s == 2 else conf_s
        cls_s = top_s.indices[:, 0].long()

        cand = (conf_s >= self.cfg.tau_prob) & (margin_s >= self.cfg.tau_margin)
        consistent = torch.zeros_like(cand)

        if logits_t is not None:
            probs_t = torch.sigmoid(logits_t)
            k_t = 2 if probs_t.shape[1] >= 2 else 1
            top_t = torch.topk(probs_t, k=k_t, dim=1)
            conf_t = top_t.values[:, 0]
            margin_t = top_t.values[:, 0] - top_t.values[:, 1] if k_t == 2 else conf_t
            cls_t = top_t.indices[:, 0].long()

            cand_t = (conf_t >= self.cfg.tau_prob) & (margin_t >= self.cfg.tau_margin)
            consistent = cls_s == cls_t
            cand = cand & cand_t & consistent
            score = 0.5 * (conf_s + conf_t) + 0.5 * (margin_s + margin_t)
            score = score + self.cfg.consistency_bonus * consistent.float()
            score = score - self.cfg.inconsistency_penalty * (~consistent).float()
        else:
            score = conf_s + margin_s

        top1_count = self._count_by_class(cls_s, self.num_classes)
        conf_pass = int(cand.sum().item())
        cand_count = self._count_by_class(cls_s[cand], self.num_classes)
        consistent_mask = cand & consistent if logits_t is not None else cand
        agree_count = self._count_by_class(cls_s[consistent_mask], self.num_classes)
        consistent_cnt = int(consistent_mask.sum().item())

        cand_idx = torch.nonzero(cand, as_tuple=False).squeeze(1)
        if cand_idx.numel() == 0:
            return (
                cand_idx.new_empty((0,), dtype=torch.long),
                cand_idx.new_empty((0,), dtype=torch.long),
                {
                    "conf_pass": conf_pass,
                    "consistent": consistent_cnt,
                    "top1_count": top1_count,
                    "cand_count": cand_count,
                    "agree_count": agree_count,
                    "selected_count": torch.zeros((self.num_classes,), device=logits_s.device, dtype=torch.long),
                },
            )

        k_ratio = int(float(cand_idx.numel()) * self.cfg.select_top_ratio)
        k = max(self.cfg.select_min_k, k_ratio)
        k = min(self.cfg.select_max_k, int(cand_idx.numel()))

        sel_idx = self._select_indices_balanced(
            cand_idx=cand_idx,
            pseudo_k_all=cls_s[cand_idx],
            score_all=score[cand_idx],
            k=k,
        )
        pseudo_k = cls_s[sel_idx]
        selected_count = self._count_by_class(pseudo_k, self.num_classes)

        return sel_idx, pseudo_k, {
            "conf_pass": conf_pass,
            "consistent": consistent_cnt,
            "top1_count": top1_count,
            "cand_count": cand_count,
            "agree_count": agree_count,
            "selected_count": selected_count,
        }

    def _select_indices_balanced(
        self,
        cand_idx: torch.Tensor,
        pseudo_k_all: torch.Tensor,
        score_all: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
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
            local = torch.topk(score_all, k=k, dim=0).indices
            return cand_idx[local]

        selected_locals = []
        chosen_mask = torch.zeros((cand_idx.numel(),), device=cand_idx.device, dtype=torch.bool)
        base_quota = max(1, k // len(present_classes))

        for c in present_classes:
            idx_c = per_class_locals[c]
            take = min(int(idx_c.numel()), base_quota)
            if take <= 0:
                continue
            top_local = torch.topk(score_all[idx_c], k=take, dim=0).indices
            chosen = idx_c[top_local]
            selected_locals.append(chosen)
            chosen_mask[chosen] = True

        selected_num = sum(int(x.numel()) for x in selected_locals)
        remaining = k - selected_num
        if remaining > 0:
            remaining_idx = torch.nonzero(~chosen_mask, as_tuple=False).squeeze(1)
            if remaining_idx.numel() > 0:
                top_remaining = torch.topk(
                    score_all[remaining_idx],
                    k=min(remaining, int(remaining_idx.numel())),
                    dim=0,
                ).indices
                selected_locals.append(remaining_idx[top_remaining])

        if not selected_locals:
            local = torch.topk(score_all, k=k, dim=0).indices
            return cand_idx[local]

        merged = torch.cat(selected_locals, dim=0)
        if merged.numel() > k:
            top_merged = torch.topk(score_all[merged], k=k, dim=0).indices
            merged = merged[top_merged]
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
                perm = torch.randperm(idx_c.numel(), device=idx_c.device)[:cap]
                idx_c = idx_c[perm]
            kept.append(idx_c)

        if not kept:
            empty = sel_idx.new_empty((0,), dtype=torch.long)
            return empty, empty

        keep_local = torch.cat(kept, dim=0)
        return sel_idx[keep_local], pseudo_k[keep_local]

    # ------------------------------
    # Update / guard
    # ------------------------------
    def _maybe_update_pending(self, reason: str, force: bool = False):
        total_sel = int(self.pending["total_selected"])
        total_conf = int(self.pending["total_conf_pass"])
        total_valid = int(self.pending["total_valid"])
        pending_steps = int(self.pending.get("pending_steps", 0))

        if not force:
            if total_sel < int(self.cfg.min_selected_anchors):
                self._log(
                    f"[PENDING] wait ({reason}) selected={total_sel}/{self.cfg.min_selected_anchors}, "
                    f"conf_pass={total_conf}/{self.cfg.min_confpass_anchors}, valid={total_valid}"
                )
                return
            if total_conf < int(self.cfg.min_confpass_anchors):
                self._log(
                    f"[PENDING] wait-lowconf ({reason}) selected={total_sel}, "
                    f"conf_pass={total_conf}/{self.cfg.min_confpass_anchors}, valid={total_valid}"
                )
                return

        if self.cfg.class_balance_enable:
            class_count = self.pending.get("class_count", None)
            if class_count is None:
                self._log(f"[PENDING] class-balance enabled but class_count is None ({reason}), waiting")
                return

            req = int(self.cfg.class_balance_min_classes_to_update)
            if req <= 0:
                req = int(self.num_classes)
            req = max(1, min(req, int(self.num_classes)))

            enough = class_count >= int(self.cfg.min_per_class_for_update)
            n_ready = int(enough.sum().item())
            class_count_str = self._format_count_tensor(class_count)

            if n_ready < req:
                force_due_to_timeout = (
                    int(self.cfg.class_balance_max_pending_steps) > 0 and
                    pending_steps >= int(self.cfg.class_balance_max_pending_steps)
                )
                if not force_due_to_timeout:
                    self._log(
                        f"[PENDING] class-balance wait ({reason}) ready_classes={n_ready}/{req} "
                        f"min_per_class={self.cfg.min_per_class_for_update} pending_steps={pending_steps} "
                        f"class_count=[{class_count_str}]"
                    )
                    return
                self._log(
                    f"[PENDING] class-balance timeout ({reason}) ready_classes={n_ready}/{req} "
                    f"pending_steps={pending_steps}, force update"
                )

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
        self._force_baseline_mode()
        self._update_teacher_from_student()

        avg_loss = float(self.pending["loss_sum"]) / max(1, int(self.pending["loss_count"]))
        updated_classes = int((self.pending["class_count"] >= int(self.cfg.min_per_class_for_update)).sum().item())
        self._log(
            f"[UPDATE {self.update_idx}] reason={reason} selected={total_sel}/{total_valid} "
            f"conf_pass={total_conf} updated_classes={updated_classes} "
            f"avg_loss={avg_loss:.6f} grad_norm={grad_norm:.6f}"
        )

        rolled_back = False
        metric = float("nan")
        if (
            self.cfg.eval_every_updates > 0
            and self.eval_fn is not None
            and self.val_loader is not None
            and (self.update_idx % self.cfg.eval_every_updates == 0)
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
                "step_idx": int(self.step_idx),
                "update_idx": int(self.update_idx),
                "metric": float(metric),
                "best_metric": float(self.best_metric),
                "rolled_back": int(rolled_back),
                "loss": float(avg_loss),
                "grad_norm": float(grad_norm),
                "selected": int(total_sel),
                "conf_pass": int(total_conf),
                "updated_classes": int(updated_classes),
            }
        )

        if self.cfg.save_every_updates > 0 and (self.update_idx % self.cfg.save_every_updates == 0):
            self._save_state(tag=f"upd{self.update_idx:04d}")

        self._reset_pending()

    def _evaluate_metric(self) -> float:
        self._force_baseline_mode()
        if self.eval_fn is None or self.val_loader is None:
            return float("nan")
        out = self.eval_fn(self.model, self.val_loader)
        if isinstance(out, dict):
            key = str(self.cfg.metric_key)
            if key in out:
                return float(out[key])
            for k in ("recall/rcnn_0.3", "recall/rcnn_0.5", "mAP", "map"):
                if k in out:
                    return float(out[k])
            raise RuntimeError(f"eval_fn returned dict without usable metric. keys={list(out.keys())[:20]}")
        return float(out)

    # ------------------------------
    # State / teacher helpers
    # ------------------------------
    def _snapshot_state(self) -> Dict[str, Any]:
        return {
            "model_state": _state_dict_to_cpu(self.model.state_dict()),
            "optimizer_state": _state_dict_to_cpu(self.optimizer.state_dict()),
            "teacher_state": _state_dict_to_cpu(self.teacher_cls_out.state_dict()),
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
        self.teacher_cls_out.load_state_dict(state["teacher_state"], strict=False)
        self.best_metric = float(state.get("best_metric", self.best_metric))
        self.last_metric = float(state.get("last_metric", self.last_metric))
        self.cfg.tau_prob = float(state.get("tau_prob", self.cfg.tau_prob))
        self.cfg.tau_margin = float(state.get("tau_margin", self.cfg.tau_margin))
        self._force_baseline_mode()
        self.optimizer.zero_grad(set_to_none=True)

    def _save_state(self, tag: str):
        if self.output_dir is None:
            return
        payload = self._snapshot_state()
        payload["online_cfg"] = copy.deepcopy(self.cfg.__dict__)
        payload["runner_type"] = "online_pure_cnn_runner"
        payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        path = self.output_dir / f"{self.state_save_prefix}_{tag}.pth"
        torch.save(payload, path)
        self._log(f"[STATE] saved: {str(path)}")

    def _export_best_model_payload(self):
        if self.output_dir is None or self.best_state is None:
            return
        path = self.output_dir / "best_online_pure_cnn_model.pth"
        torch.save(self.best_state, path)
        self._log(f"[BEST] exported best CNN model payload: {str(path)}")

    @torch.no_grad()
    def _update_teacher_from_student(self):
        if not self.cfg.use_teacher:
            return
        m = float(self.cfg.teacher_momentum)
        for p_t, p_s in zip(self.teacher_cls_out.parameters(), self.dense_head.conv_cls_out.parameters()):
            p_t.mul_(m).add_((1.0 - m) * p_s.detach())
        for b_t, b_s in zip(self.teacher_cls_out.buffers(), self.dense_head.conv_cls_out.buffers()):
            b_t.copy_(b_s.detach())

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

    # ------------------------------
    # Pending / logging / cfg
    # ------------------------------
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
            "loss_count": 0,
        }

    def _accumulate_pending(self, step: Dict[str, Any]):
        self.pending["pending_steps"] += 1
        self.pending["total_valid"] += int(step["total_valid"])
        self.pending["total_selected"] += int(step["total_selected"])
        self.pending["total_conf_pass"] += int(step["total_conf_pass"])
        self.pending["total_consistent"] += int(step["total_consistent"])
        self.pending["loss_sum"] += float(step.get("loss", 0.0))
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
            f"class_count=[{self._format_count_tensor(self.pending.get('class_count', None))}] "
            f"top1=[{self._format_count_tensor(self.pending.get('top1_count', None))}] "
            f"cand=[{self._format_count_tensor(self.pending.get('cand_count', None))}] "
            f"agree=[{self._format_count_tensor(self.pending.get('agree_count', None))}] "
            f"selected=[{self._format_count_tensor(self.pending.get('selected_count', None))}]"
        )

    @staticmethod
    def _count_by_class(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        out = torch.zeros((num_classes,), device=labels.device, dtype=torch.long)
        if labels.numel() == 0:
            return out
        out.index_add_(0, labels.long(), torch.ones_like(labels, dtype=torch.long))
        return out

    @staticmethod
    def _format_count_tensor(x: Optional[torch.Tensor]) -> str:
        if x is None:
            return "None"
        return ",".join([str(int(v)) for v in x.detach().cpu().tolist()])

    @staticmethod
    def _build_cfg(cfg_in: Dict[str, Any]) -> OnlinePureCNNConfig:
        cfg = OnlinePureCNNConfig()
        for k, v in cfg_in.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        cfg.tau_prob = float(cfg.tau_prob)
        cfg.tau_margin = float(cfg.tau_margin)
        cfg.min_tau_prob = float(cfg.min_tau_prob)
        cfg.min_tau_margin = float(cfg.min_tau_margin)
        cfg.select_top_ratio = float(max(0.0, cfg.select_top_ratio))
        cfg.select_min_k = int(max(0, cfg.select_min_k))
        cfg.select_max_k = int(max(1, cfg.select_max_k))
        cfg.consistency_bonus = float(cfg.consistency_bonus)
        cfg.inconsistency_penalty = float(cfg.inconsistency_penalty)

        cfg.use_teacher = bool(cfg.use_teacher)
        cfg.teacher_momentum = float(max(0.0, min(1.0, cfg.teacher_momentum)))

        cfg.update_every_n_steps = int(max(1, cfg.update_every_n_steps))
        cfg.min_selected_anchors = int(max(0, cfg.min_selected_anchors))
        cfg.min_confpass_anchors = int(max(0, cfg.min_confpass_anchors))
        cfg.grad_clip = float(max(0.0, cfg.grad_clip))

        cfg.max_per_class_per_step = int(max(0, cfg.max_per_class_per_step))
        cfg.min_per_class_for_update = int(max(0, cfg.min_per_class_for_update))
        cfg.class_balance_enable = bool(cfg.class_balance_enable)
        cfg.class_balance_min_classes_to_update = int(max(0, cfg.class_balance_min_classes_to_update))
        cfg.class_balance_max_pending_steps = int(max(0, cfg.class_balance_max_pending_steps))

        cfg.eval_every_updates = int(max(0, cfg.eval_every_updates))
        cfg.metric_key = str(cfg.metric_key)
        cfg.guard_max_drop = float(max(0.0, cfg.guard_max_drop))
        cfg.guard_use_best = bool(cfg.guard_use_best)

        cfg.save_every_updates = int(max(0, cfg.save_every_updates))
        cfg.log_interval_steps = int(max(1, cfg.log_interval_steps))
        cfg.save_best_model = bool(cfg.save_best_model)
        cfg.save_last_model = bool(cfg.save_last_model)
        cfg.pseudo_logits_source = str(cfg.pseudo_logits_source).lower()
        if cfg.pseudo_logits_source not in ("final", "origin"):
            cfg.pseudo_logits_source = "origin"

        cfg.zero_update_tau_prob_step = float(max(0.0, cfg.zero_update_tau_prob_step))
        cfg.zero_update_tau_margin_step = float(max(0.0, cfg.zero_update_tau_margin_step))
        return cfg

    def _append_metric_row(self, row: Dict[str, Any]):
        if self.metrics_csv_path is None:
            return
        fieldnames = [
            "event", "step_idx", "update_idx", "metric", "best_metric", "rolled_back",
            "loss", "grad_norm", "selected", "conf_pass", "updated_classes",
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
