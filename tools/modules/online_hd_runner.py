from __future__ import annotations

import copy
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from pcdet.models import load_data_to_gpu
from modules.online_state_io import OnlineStateIO


@dataclass
class OnlineHDConfig:
    # Pseudo label selection
    tau_prob: float = 0.35
    tau_margin: float = 0.05
    min_tau_prob: float = 0.15
    min_tau_margin: float = 0.01
    select_top_ratio: float = 0.02
    select_min_k: int = 256
    select_max_k: int = 4096
    consistency_bonus: float = 0.10
    inconsistency_penalty: float = 0.05

    # Teacher
    use_teacher: bool = True
    teacher_momentum: float = 0.995

    # Update trigger
    update_every_n_steps: int = 5
    min_selected_anchors: int = 4096
    min_confpass_anchors: int = 1024

    # Update behavior
    update_mode: str = "train"  # train | retrain | both
    alpha: float = 1.0
    source_pullback_lambda: float = 0.015
    max_per_class_per_step: int = 0
    min_per_class_for_update: int = 32
    class_balance_enable: bool = False
    class_balance_min_classes_to_update: int = 0   # <=0 means require all classes
    class_balance_max_pending_steps: int = 0       # <=0 means no forced timeout

    # Replay
    replay_enable: bool = True
    replay_cap_per_class: int = 512
    replay_per_class: int = 64
    replay_alpha_scale: float = 0.5

    # Guards
    eval_every_updates: int = 1
    guard_max_drop: float = 0.005
    guard_use_best: bool = True

    # IO
    save_every_updates: int = 1
    log_interval_steps: int = 20
    feat_source: str = "cls"  # cls | bev
    pseudo_logits_source: str = "final"  # final | origin | hd

    # Threshold adaption when update failed
    zero_update_tau_prob_step: float = 0.03
    zero_update_tau_margin_step: float = 0.01


class OnlineHDRunner:
    """
    HyperRadar online unsupervised HD adaptation runner.

    Design:
    - Keep detector frozen; update only HD memory/prototypes.
    - Use student/teacher consistency with adaptive top-k selection.
    - Support source-anchor pullback, replay, guard+rollback, and resume.

    Required model assumptions:
    - model.dense_head.hd_core exists.
    - dense_head.forward_ret_dict includes cls_preds and optionally cls_preds_origin/cls_preds_hd.
    - batch_dict after forward includes either hd_cls_feat (preferred) or spatial_features_2d.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        stream_loader,
        logger,
        cfg: Optional[Dict[str, Any]] = None,
        *,
        val_loader=None,
        eval_fn: Optional[Callable[[torch.nn.Module, Any], float]] = None,
        output_dir: Optional[str] = None,
        state_save_prefix: str = "online_hd",
        use_amp: bool = False,
    ):
        self.model = model
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

        self.device = next(self.model.parameters()).device

        self.dense_head = getattr(self.model, "dense_head", None)
        if self.dense_head is None:
            raise RuntimeError("OnlineHDRunner requires model.dense_head.")

        self.hd_core = getattr(self.dense_head, "hd_core", None)
        if self.hd_core is None:
            raise RuntimeError("OnlineHDRunner requires dense_head.hd_core.")

        self.num_anchors_per_loc = int(getattr(self.dense_head, "num_anchors_per_location", 0))
        if self.num_anchors_per_loc <= 0:
            raise RuntimeError("Invalid dense_head.num_anchors_per_location.")

        self.num_classes = int(getattr(self.dense_head, "num_class", 0))
        if self.num_classes <= 0:
            raise RuntimeError("Invalid dense_head.num_class.")

        # Runtime states
        self.step_idx = 0
        self.update_idx = 0
        self.pending = self._new_pending_buffer()

        self.best_metric = float("-inf")
        self.last_metric = float("-inf")
        self.best_state = None

        # Teacher/source memories (prototype space, normalized rows)
        with torch.no_grad():
            self.hd_core.memory.normalize_()
            self.teacher_prototypes = self.hd_core.memory.prototypes.detach().clone()
            self.source_prototypes = self.hd_core.memory.prototypes.detach().clone()
            self.teacher_bg_prototype = self.hd_core.memory.bg_prototype.detach().clone()
            self.source_bg_prototype = self.hd_core.memory.bg_prototype.detach().clone()

        # Replay buffer in HV space, per class (CPU tensors)
        self.mem_hv: Dict[int, Optional[torch.Tensor]] = {c: None for c in range(self.num_classes)}
        self._metrics_header_written = False

    # ------------------------------
    # Public APIs
    # ------------------------------
    def run(self, *, max_steps: int = -1):
        self.model.eval()
        self._log("[ONLINE] start")
        self._log(
            f"[ONLINE] cfg: feat_source={self.cfg.feat_source}, pseudo_logits_source={self.cfg.pseudo_logits_source}, "
            f"update_mode={self.cfg.update_mode}, use_teacher={self.cfg.use_teacher}"
        )

        # Baseline evaluation before any online update.
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

            step_stats = self._process_one_step(batch)
            self._accumulate_pending(step_stats)

            # Trigger update by step interval
            if self.step_idx % self.cfg.update_every_n_steps == 0:
                self._maybe_update_pending(reason=f"step_interval@{self.step_idx}")

            if self.step_idx % self.cfg.log_interval_steps == 0:
                class_count = self.pending.get("class_count", None)
                self._log(
                    f"[STEP {self.step_idx}] pending_selected={self.pending['total_selected']} "
                    f"pending_conf_pass={self.pending['total_conf_pass']} pending_valid={self.pending['total_valid']} "
                    f"pending_steps={self.pending['pending_steps']} class_count=[{self._format_count_tensor(class_count)}] "
                    f"top1=[{self._format_count_tensor(self.pending.get('top1_count', None))}] "
                    f"cand=[{self._format_count_tensor(self.pending.get('cand_count', None))}] "
                    f"agree=[{self._format_count_tensor(self.pending.get('agree_count', None))}] "
                    f"selected=[{self._format_count_tensor(self.pending.get('selected_count', None))}]"
                )

        # Flush at end
        self._maybe_update_pending(reason="final_flush", force=True)

        # Final state save
        self._save_state(tag="last")
        self._export_best_memory_payload()

        self._log(f"[ONLINE] finished | steps={self.step_idx} updates={self.update_idx} elapsed={time.time()-t0:.1f}s")

    def load_state(self, ckpt_path: str):
        ckpt = OnlineStateIO.load(ckpt_path)
        restored = OnlineStateIO.apply_payload(
            hd_core=self.hd_core,
            payload=ckpt,
            device=self.device,
            num_classes=self.num_classes,
            strict_memory=False,
        )

        if "teacher_prototypes" in restored:
            self.teacher_prototypes = restored["teacher_prototypes"]
        if "teacher_bg_prototype" in restored:
            self.teacher_bg_prototype = restored["teacher_bg_prototype"]
        if "source_prototypes" in restored:
            self.source_prototypes = restored["source_prototypes"]
        if "source_bg_prototype" in restored:
            self.source_bg_prototype = restored["source_bg_prototype"]

        self.step_idx = int(restored.get("step_idx", 0))
        self.update_idx = int(restored.get("update_idx", 0))
        self.best_metric = float(restored.get("best_metric", float("-inf")))
        self.last_metric = float(restored.get("last_metric", float("-inf")))

        if restored.get("tau_prob", None) is not None:
            self.cfg.tau_prob = float(restored["tau_prob"])
        if restored.get("tau_margin", None) is not None:
            self.cfg.tau_margin = float(restored["tau_margin"])
        if restored.get("alpha", None) is not None:
            self.cfg.alpha = float(restored["alpha"])

        self.mem_hv = restored.get("mem_hv", self.mem_hv)
        self.best_state = restored.get("best_state", None)
        if self.best_state is None:
            self.best_state = self._snapshot_state()
        self._log(f"[STATE] loaded: {ckpt_path} | step={self.step_idx} update={self.update_idx}")

    # ------------------------------
    # Core step logic
    # ------------------------------
    def _process_one_step(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        load_data_to_gpu(batch_dict)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # model forward writes into dense_head.forward_ret_dict and batch_dict
                _pred, _ret = self.model(batch_dict)

            fr = self.dense_head.forward_ret_dict
            feat_bnc, student_logits_bnk = self._extract_anchor_features_and_student_logits(batch_dict, fr)

            B, N, C = feat_bnc.shape
            K = self.num_classes

            step = {
                "total_valid": int(B * N),
                "total_selected": 0,
                "total_conf_pass": 0,
                "total_consistent": 0,
                "class_sum": torch.zeros((K, C), device=self.device, dtype=feat_bnc.dtype),
                "class_count": torch.zeros((K,), device=self.device, dtype=torch.long),
                "top1_count": torch.zeros((K,), device=self.device, dtype=torch.long),
                "cand_count": torch.zeros((K,), device=self.device, dtype=torch.long),
                "agree_count": torch.zeros((K,), device=self.device, dtype=torch.long),
                "selected_count": torch.zeros((K,), device=self.device, dtype=torch.long),
                "batch_selected": [],
            }

            for b in range(B):
                feat_n_c = feat_bnc[b]
                logits_s = student_logits_bnk[b]

                logits_t = None
                if self.cfg.use_teacher:
                    logits_t = self._compute_teacher_logits(feat_n_c)

                sel_idx, pseudo_k, stat = self._select_pseudo_indices(logits_s, logits_t)
                step["total_conf_pass"] += stat["conf_pass"]
                step["total_consistent"] += stat["consistent"]
                step["top1_count"] += stat["top1_count"]
                step["cand_count"] += stat["cand_count"]
                step["agree_count"] += stat["agree_count"]
                step["selected_count"] += stat["selected_count"]

                if sel_idx.numel() == 0:
                    continue

                # Optional class cap per step
                sel_idx, pseudo_k = self._apply_class_cap(sel_idx, pseudo_k)
                if sel_idx.numel() == 0:
                    continue

                # Aggregate selected features for update trigger
                feat_sel = feat_n_c[sel_idx]
                step["total_selected"] += int(sel_idx.numel())
                logits_sel = logits_s[sel_idx]
                labels_sel = pseudo_k + 1  # hd_core positives are 1..K
                # Keep only selected anchors to avoid retaining full-map tensors in pending buffer.
                step["batch_selected"].append((feat_sel, logits_sel, labels_sel))

                # Track per-class stats in feature space (used for diagnostics; update uses hd_core)
                step["class_sum"].index_add_(0, pseudo_k, feat_sel)
                ones = torch.ones_like(pseudo_k, dtype=torch.long)
                step["class_count"].index_add_(0, pseudo_k, ones)

                # Add replay memory in HV space
                hv_sel = self._encode_features_to_hv(feat_sel)
                self._replay_add(hv_sel, pseudo_k)

            return step

    def _extract_anchor_features_and_student_logits(
        self,
        batch_dict: Dict[str, Any],
        forward_ret: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Feature map source
        feat_source = self.cfg.feat_source.lower()
        if feat_source == "cls":
            feat_map = batch_dict.get("hd_cls_feat", None)
            if feat_map is None:
                raise RuntimeError("Expected batch_dict['hd_cls_feat'] for feat_source='cls'.")
        elif feat_source == "bev":
            feat_map = batch_dict.get("spatial_features_2d", None)
            if feat_map is None:
                raise RuntimeError("Expected batch_dict['spatial_features_2d'] for feat_source='bev'.")
        else:
            raise RuntimeError(f"Unknown feat_source={self.cfg.feat_source}")

        # Choose student logits source
        src = self.cfg.pseudo_logits_source.lower()
        if src == "origin" and "cls_preds_origin" in forward_ret:
            cls_hwk = forward_ret["cls_preds_origin"]
        elif src == "hd" and "cls_preds_hd" in forward_ret:
            cls_hwk = forward_ret["cls_preds_hd"]
        else:
            cls_hwk = forward_ret["cls_preds"]

        B, C, H, W = feat_map.shape
        A = self.num_anchors_per_loc
        K = self.num_classes

        if cls_hwk.shape[0] != B or cls_hwk.shape[1] != H or cls_hwk.shape[2] != W or cls_hwk.shape[3] != A * K:
            raise RuntimeError(
                f"Shape mismatch: feat_map={tuple(feat_map.shape)} cls_preds={tuple(cls_hwk.shape)} A={A} K={K}"
            )

        # [B,C,H,W] -> [B,H,W,C] -> [B,H,W,A,C] -> [B,N,C]
        feat_cell = feat_map.permute(0, 2, 3, 1).contiguous()
        feat_anchor = feat_cell.unsqueeze(3).expand(B, H, W, A, C).reshape(B, H * W * A, C).contiguous()

        # logits [B,H,W,A*K] -> [B,N,K]
        logits = cls_hwk.view(B, H * W * A, K).float()
        return feat_anchor, logits

    # ------------------------------
    # Pseudo selection
    # ------------------------------
    def _select_pseudo_indices(
        self,
        logits_s: torch.Tensor,
        logits_t: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
        # Use sigmoid for anchor classification logits
        probs_s = torch.sigmoid(logits_s)
        k_s = 2 if probs_s.shape[1] >= 2 else 1
        top_s = torch.topk(probs_s, k=k_s, dim=1)
        conf_s = top_s.values[:, 0]
        if k_s == 2:
            margin_s = top_s.values[:, 0] - top_s.values[:, 1]
        else:
            margin_s = conf_s
        cls_s = top_s.indices[:, 0].long()

        cand = (conf_s >= self.cfg.tau_prob) & (margin_s >= self.cfg.tau_margin)

        consistent = torch.zeros_like(cand)
        if logits_t is not None:
            probs_t = torch.sigmoid(logits_t)
            k_t = 2 if probs_t.shape[1] >= 2 else 1
            top_t = torch.topk(probs_t, k=k_t, dim=1)
            conf_t = top_t.values[:, 0]
            if k_t == 2:
                margin_t = top_t.values[:, 0] - top_t.values[:, 1]
            else:
                margin_t = conf_t
            cls_t = top_t.indices[:, 0].long()

            cand_t = (conf_t >= self.cfg.tau_prob) & (margin_t >= self.cfg.tau_margin)
            cand = cand & cand_t
            consistent = cls_s == cls_t

            score = 0.5 * (conf_s + conf_t) + 0.5 * (margin_s + margin_t)
            score = score + self.cfg.consistency_bonus * consistent.float()
            score = score - self.cfg.inconsistency_penalty * (~consistent).float()
            cand = cand & consistent
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
        k = min(k, self.cfg.select_max_k, int(cand_idx.numel()))

        sel_idx = self._select_indices_balanced(
            cand_idx=cand_idx,
            pseudo_k_all=cls_s[cand_idx],
            score_all=score[cand_idx],
            k=k,
        )
        pseudo_k = cls_s[sel_idx]  # 0..K-1
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
                    score_all[remaining_idx], k=min(remaining, int(remaining_idx.numel())), dim=0
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
    # Update core
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

        # Class-balance gate: accumulate until enough per-class support.
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
            class_count_str = ",".join([str(int(x)) for x in class_count.detach().cpu().tolist()])

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
            # No update, relax thresholds slightly to avoid deadlock
            old_p, old_m = self.cfg.tau_prob, self.cfg.tau_margin
            self.cfg.tau_prob = max(self.cfg.min_tau_prob, self.cfg.tau_prob - self.cfg.zero_update_tau_prob_step)
            self.cfg.tau_margin = max(self.cfg.min_tau_margin, self.cfg.tau_margin - self.cfg.zero_update_tau_margin_step)
            self._log(
                f"[PENDING] zero selected ({reason}), relax tau_prob {old_p:.3f}->{self.cfg.tau_prob:.3f}, "
                f"tau_margin {old_m:.3f}->{self.cfg.tau_margin:.3f}"
            )
            self._reset_pending()
            return

        self.update_idx += 1
        snapshot = self._snapshot_state()

        updated_classes = self._apply_online_update_from_pending()
        self._update_teacher_from_student()
        self._apply_source_pullback()

        self._log(
            f"[UPDATE {self.update_idx}] reason={reason} selected={total_sel}/{total_valid} "
            f"conf_pass={total_conf} updated_classes={updated_classes} alpha={self.cfg.alpha:.4f}"
        )

        rolled_back = False
        metric = float("nan")
        # Guard by eval callback
        if self.eval_fn is not None and self.val_loader is not None and (self.update_idx % self.cfg.eval_every_updates == 0):
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
                self._save_state(tag="best")

            self._log(
                f"[UPDATE {self.update_idx}] post-eval metric={metric:.6f}, best={self.best_metric:.6f}"
            )

        self._append_metric_row(
            {
                "event": "update",
                "step_idx": int(self.step_idx),
                "update_idx": int(self.update_idx),
                "metric": float(metric),
                "best_metric": float(self.best_metric),
                "rolled_back": int(rolled_back),
                "selected": int(total_sel),
                "conf_pass": int(total_conf),
                "updated_classes": int(updated_classes),
            }
        )

        # Periodic save
        if self.cfg.save_every_updates > 0 and (self.update_idx % self.cfg.save_every_updates == 0):
            self._save_state(tag=f"upd{self.update_idx:04d}")

        self._reset_pending()

    def _apply_online_update_from_pending(self) -> int:
        class_touch = torch.zeros((self.num_classes,), device=self.device, dtype=torch.long)

        for feat_sel, logits_sel, labels_sel in self.pending["batch_selected"]:
            if labels_sel.numel() == 0:
                continue

            pseudo_k = labels_sel - 1
            class_touch.index_add_(0, pseudo_k, torch.ones_like(pseudo_k, dtype=torch.long))

            mode = self.cfg.update_mode.lower()
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

        # Replay
        if self.cfg.replay_enable:
            self._replay_apply(alpha_scale=self.cfg.replay_alpha_scale)

        # Ensure normalized prototypes
        with torch.no_grad():
            self.hd_core.memory.normalize_()

        # Enforce minimum per-class support (optional pruning by class count in this update)
        # Here we keep it simple: report only classes with enough selected anchors.
        updated_classes = int((class_touch >= int(self.cfg.min_per_class_for_update)).sum().item())
        return updated_classes

    # ------------------------------
    # Teacher/source/replay helpers
    # ------------------------------
    def _compute_teacher_logits(self, feat_n_c: torch.Tensor) -> torch.Tensor:
        A = self.num_anchors_per_loc
        N, C = feat_n_c.shape

        # Build deterministic anchor ids 0..A-1 repeated
        anchor_ids = torch.arange(A, device=feat_n_c.device, dtype=torch.long).repeat(N // A)
        if anchor_ids.numel() < N:
            extra = torch.arange(N - anchor_ids.numel(), device=feat_n_c.device, dtype=torch.long)
            anchor_ids = torch.cat([anchor_ids, extra], dim=0)

        feat_ctx = self.hd_core.inject_anchor_context(feat_n_c, anchor_ids, num_anchors=A)
        hv = self._encode_features_to_hv(feat_ctx)

        # teacher logits in cosine space
        proto = self._normalize_rows(self.teacher_prototypes)
        hv_n = self._normalize_rows(hv)
        logits = hv_n @ proto.t()
        temp = float(getattr(self.hd_core.cfg, "temperature", 1.0))
        if temp != 1.0:
            logits = logits / temp

        if bool(getattr(self.hd_core.cfg, "bg_enabled", False)):
            bg = self._normalize_rows(self.teacher_bg_prototype.view(1, -1)).view(-1)
            bg_logit = hv_n @ bg
            logits = logits - float(getattr(self.hd_core.cfg, "bg_margin_scale", 1.0)) * bg_logit.unsqueeze(1)

        return logits

    def _encode_features_to_hv(self, feat_n_c: torch.Tensor) -> torch.Tensor:
        chunk = int(getattr(self.hd_core.cfg, "encode_chunk", 0))
        if chunk <= 0:
            chunk = feat_n_c.shape[0]
        hv = self.hd_core.embedder.forward_chunked(
            feat_n_c,
            chunk=chunk,
            quantize=bool(getattr(self.hd_core.cfg, "quantize", False)),
        )
        hv = self._normalize_rows(hv)
        return hv

    def _update_teacher_from_student(self):
        if not self.cfg.use_teacher:
            return
        m = float(self.cfg.teacher_momentum)
        with torch.no_grad():
            p = self.hd_core.memory.prototypes.detach()
            self.teacher_prototypes.mul_(m)
            self.teacher_prototypes.add_((1.0 - m) * p)
            self.teacher_prototypes.copy_(self._normalize_rows(self.teacher_prototypes))

            if hasattr(self.hd_core.memory, "bg_prototype"):
                bg = self.hd_core.memory.bg_prototype.detach()
                self.teacher_bg_prototype.mul_(m)
                self.teacher_bg_prototype.add_((1.0 - m) * bg)
                self.teacher_bg_prototype.copy_(self._normalize_rows(self.teacher_bg_prototype.view(1, -1)).view(-1))

    def _apply_source_pullback(self):
        lam = float(self.cfg.source_pullback_lambda)
        if lam <= 0:
            return

        with torch.no_grad():
            cur = self.hd_core.memory.prototypes.detach()
            mix = (1.0 - lam) * cur + lam * self.source_prototypes.to(cur.device)
            mix = self._normalize_rows(mix)
            self.hd_core.memory.prototypes.copy_(mix)
            self.hd_core.memory.classify_weights.copy_(mix)

            if hasattr(self.hd_core.memory, "bg_prototype"):
                bg = self.hd_core.memory.bg_prototype.detach()
                bg_mix = (1.0 - lam) * bg + lam * self.source_bg_prototype.to(bg.device)
                bg_mix = self._normalize_rows(bg_mix.view(1, -1)).view(-1)
                self.hd_core.memory.bg_prototype.copy_(bg_mix)
                if hasattr(self.hd_core.memory, "bg_weight"):
                    self.hd_core.memory.bg_weight.copy_(bg_mix)

            self.hd_core.memory.normalize_()

    def _replay_add(self, hv: torch.Tensor, pseudo_k: torch.Tensor):
        if not self.cfg.replay_enable:
            return

        cap = int(self.cfg.replay_cap_per_class)
        hv_cpu = hv.detach().float().cpu()
        lb_cpu = pseudo_k.detach().long().cpu()

        for c in range(self.num_classes):
            idx = torch.nonzero(lb_cpu == c, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            cur = hv_cpu[idx]
            old = self.mem_hv[c]
            merged = cur if old is None else torch.cat([old, cur], dim=0)
            if cap > 0 and merged.shape[0] > cap:
                merged = merged[-cap:]
            self.mem_hv[c] = merged

    def _replay_apply(self, alpha_scale: float):
        per_class = int(self.cfg.replay_per_class)
        if per_class <= 0:
            return

        hv_list = []
        lb_list = []
        for c in range(self.num_classes):
            buf = self.mem_hv[c]
            if buf is None or buf.shape[0] == 0:
                continue
            n = min(per_class, buf.shape[0])
            idx = torch.randperm(buf.shape[0])[:n]
            hv_list.append(buf[idx])
            lb_list.append(torch.full((n,), c, dtype=torch.long))

        if not hv_list:
            return

        hv = torch.cat(hv_list, dim=0).to(self.device)
        lb = torch.cat(lb_list, dim=0).to(self.device)

        # Direct memory replay with labels in 0..K-1
        with torch.no_grad():
            self.hd_core.memory.add_(lb, hv, alpha=float(self.cfg.alpha) * float(alpha_scale))
            self.hd_core.memory.normalize_()

    # ------------------------------
    # Eval / guard / state
    # ------------------------------
    def _evaluate_metric(self) -> float:
        out = self.eval_fn(self.model, self.val_loader)
        if isinstance(out, dict):
            # common key fallback
            for k in ("recall/rcnn_0.3", "recall/rcnn_0.5", "mAP", "map"):
                if k in out:
                    return float(out[k])
            raise RuntimeError(f"eval_fn returned dict without known metric keys: {list(out.keys())[:20]}")
        return float(out)

    def _snapshot_state(self) -> Dict[str, Any]:
        mem_state = {}
        for k, v in self.hd_core.memory.state_dict().items():
            if torch.is_tensor(v):
                mem_state[k] = v.detach().cpu().clone()
            else:
                mem_state[k] = copy.deepcopy(v)

        mem_hv = {}
        for c in range(self.num_classes):
            t = self.mem_hv[c]
            mem_hv[c] = None if t is None else t.clone()

        return {
            "memory_state": mem_state,
            "teacher_prototypes": self.teacher_prototypes.detach().cpu().clone(),
            "teacher_bg_prototype": self.teacher_bg_prototype.detach().cpu().clone(),
            "source_prototypes": self.source_prototypes.detach().cpu().clone(),
            "source_bg_prototype": self.source_bg_prototype.detach().cpu().clone(),
            "tau_prob": float(self.cfg.tau_prob),
            "tau_margin": float(self.cfg.tau_margin),
            "alpha": float(self.cfg.alpha),
            "mem_hv": mem_hv,
            "step_idx": int(self.step_idx),
            "update_idx": int(self.update_idx),
            "best_metric": float(self.best_metric),
            "last_metric": float(self.last_metric),
        }

    def _rollback_state(self, state: Dict[str, Any]):
        with torch.no_grad():
            self.hd_core.memory.load_state_dict(state["memory_state"], strict=False)
            self.hd_core.memory.normalize_()
            self.teacher_prototypes = state["teacher_prototypes"].to(self.device).clone()
            self.teacher_bg_prototype = state["teacher_bg_prototype"].to(self.device).clone()
            self.source_prototypes = state["source_prototypes"].to(self.device).clone()
            self.source_bg_prototype = state["source_bg_prototype"].to(self.device).clone()

        self.cfg.tau_prob = float(state["tau_prob"])
        self.cfg.tau_margin = float(state["tau_margin"])
        self.cfg.alpha = float(state["alpha"])

        restored = {}
        for c in range(self.num_classes):
            t = state["mem_hv"].get(c, None)
            restored[c] = None if t is None else t.clone()
        self.mem_hv = restored

    def _save_state(self, tag: str):
        if self.output_dir is None:
            return

        payload = OnlineStateIO.build_payload(
            hd_core=self.hd_core,
            teacher_prototypes=self.teacher_prototypes,
            teacher_bg_prototype=self.teacher_bg_prototype,
            source_prototypes=self.source_prototypes,
            source_bg_prototype=self.source_bg_prototype,
            tau_prob=self.cfg.tau_prob,
            tau_margin=self.cfg.tau_margin,
            alpha=self.cfg.alpha,
            step_idx=self.step_idx,
            update_idx=self.update_idx,
            best_metric=self.best_metric,
            last_metric=self.last_metric,
            best_state=self.best_state,
            mem_hv=self.mem_hv,
            online_cfg=self.cfg.__dict__.copy(),
        )
        path = self.output_dir / f"{self.state_save_prefix}_{tag}.pth"
        OnlineStateIO.save(str(path), payload)
        self._log(f"[STATE] saved: {str(path)}")

    def _export_best_memory_payload(self):
        if self.output_dir is None:
            return
        if self.best_state is None:
            self._log("[BEST] no best_state found, skip best memory export")
            return

        mem_state = self.best_state.get("memory_state", None)
        if not isinstance(mem_state, dict):
            self._log("[BEST] best_state has no valid memory_state, skip best memory export")
            return

        payload: Dict[str, Any] = {
            "memory": copy.deepcopy(mem_state),
            "meta": {
                "source": "online_hd_runner.best_state",
                "best_metric": float(self.best_metric),
                "step_idx": int(self.best_state.get("step_idx", self.step_idx)),
                "update_idx": int(self.best_state.get("update_idx", self.update_idx)),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }

        # embedder itself is static during online memory update, keep current snapshot for portability
        embedder = getattr(self.hd_core, "embedder", None)
        if embedder is not None:
            emb_state = {}
            for k, v in embedder.state_dict().items():
                if torch.is_tensor(v):
                    emb_state[k] = v.detach().cpu().clone()
                else:
                    emb_state[k] = copy.deepcopy(v)
            payload["embedder"] = emb_state

        # provide simplified compatibility fields
        for k in ("classify_weights", "prototypes", "bg_weight", "bg_prototype"):
            if k in mem_state:
                v = mem_state[k]
                payload[k] = v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)

        path = self.output_dir / "best_hd_memory.pth"
        OnlineStateIO.save(str(path), payload)
        self._log(f"[BEST] exported best memory payload: {str(path)}")

    # ------------------------------
    # Pending accumulator
    # ------------------------------
    def _new_pending_buffer(self) -> Dict[str, Any]:
        return {
            "pending_steps": 0,
            "total_valid": 0,
            "total_selected": 0,
            "total_conf_pass": 0,
            "total_consistent": 0,
            "class_sum": None,
            "class_count": None,
            "top1_count": None,
            "cand_count": None,
            "agree_count": None,
            "selected_count": None,
            "batch_selected": [],
        }

    def _accumulate_pending(self, step: Dict[str, Any]):
        self.pending["pending_steps"] += 1
        self.pending["total_valid"] += int(step["total_valid"])
        self.pending["total_selected"] += int(step["total_selected"])
        self.pending["total_conf_pass"] += int(step["total_conf_pass"])
        self.pending["total_consistent"] += int(step["total_consistent"])

        if self.pending["class_sum"] is None:
            self.pending["class_sum"] = step["class_sum"].detach().clone()
            self.pending["class_count"] = step["class_count"].detach().clone()
            self.pending["top1_count"] = step["top1_count"].detach().clone()
            self.pending["cand_count"] = step["cand_count"].detach().clone()
            self.pending["agree_count"] = step["agree_count"].detach().clone()
            self.pending["selected_count"] = step["selected_count"].detach().clone()
        else:
            self.pending["class_sum"] += step["class_sum"]
            self.pending["class_count"] += step["class_count"]
            self.pending["top1_count"] += step["top1_count"]
            self.pending["cand_count"] += step["cand_count"]
            self.pending["agree_count"] += step["agree_count"]
            self.pending["selected_count"] += step["selected_count"]

        self.pending["batch_selected"].extend(step["batch_selected"])

    def _reset_pending(self):
        self.pending = self._new_pending_buffer()

    # ------------------------------
    # Utils
    # ------------------------------
    @staticmethod
    def _normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        if x.dim() == 1:
            return x / x.norm().clamp_min(eps)
        return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)

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
    def _build_cfg(cfg_in: Dict[str, Any]) -> OnlineHDConfig:
        cfg = OnlineHDConfig()
        for k, v in cfg_in.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        # sanitize
        cfg.update_mode = str(cfg.update_mode).lower()
        if cfg.update_mode not in ("train", "retrain", "both"):
            cfg.update_mode = "train"
        cfg.feat_source = str(cfg.feat_source).lower()
        if cfg.feat_source not in ("cls", "bev"):
            cfg.feat_source = "cls"
        cfg.pseudo_logits_source = str(cfg.pseudo_logits_source).lower()
        if cfg.pseudo_logits_source not in ("final", "origin", "hd"):
            cfg.pseudo_logits_source = "final"

        cfg.tau_prob = float(max(0.0, min(1.0, cfg.tau_prob)))
        cfg.tau_margin = float(max(0.0, min(1.0, cfg.tau_margin)))
        cfg.min_tau_prob = float(max(0.0, min(1.0, cfg.min_tau_prob)))
        cfg.min_tau_margin = float(max(0.0, min(1.0, cfg.min_tau_margin)))
        cfg.teacher_momentum = float(max(0.0, min(1.0, cfg.teacher_momentum)))
        cfg.source_pullback_lambda = float(max(0.0, min(1.0, cfg.source_pullback_lambda)))
        cfg.guard_max_drop = float(max(0.0, cfg.guard_max_drop))
        cfg.alpha = float(max(0.0, cfg.alpha))
        cfg.replay_alpha_scale = float(max(0.0, cfg.replay_alpha_scale))

        cfg.select_top_ratio = float(max(0.0, cfg.select_top_ratio))
        cfg.select_min_k = int(max(1, cfg.select_min_k))
        cfg.select_max_k = int(max(cfg.select_min_k, cfg.select_max_k))

        cfg.update_every_n_steps = int(max(1, cfg.update_every_n_steps))
        cfg.eval_every_updates = int(max(1, cfg.eval_every_updates))
        cfg.log_interval_steps = int(max(1, cfg.log_interval_steps))
        cfg.save_every_updates = int(max(0, cfg.save_every_updates))

        cfg.min_selected_anchors = int(max(0, cfg.min_selected_anchors))
        cfg.min_confpass_anchors = int(max(0, cfg.min_confpass_anchors))
        cfg.min_per_class_for_update = int(max(0, cfg.min_per_class_for_update))
        cfg.class_balance_min_classes_to_update = int(max(0, cfg.class_balance_min_classes_to_update))
        cfg.class_balance_max_pending_steps = int(max(0, cfg.class_balance_max_pending_steps))
        cfg.replay_cap_per_class = int(max(0, cfg.replay_cap_per_class))
        cfg.replay_per_class = int(max(0, cfg.replay_per_class))

        return cfg

    def _log(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def _append_metric_row(self, row: Dict[str, Any]):
        if self.metrics_csv_path is None:
            return
        fieldnames = [
            "event", "step_idx", "update_idx", "metric", "best_metric",
            "rolled_back", "selected", "conf_pass", "updated_classes"
        ]
        self.metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (not self._metrics_header_written) and (not self.metrics_csv_path.exists())
        with open(self.metrics_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                self._metrics_header_written = True
            writer.writerow({k: row.get(k, "") for k in fieldnames})
