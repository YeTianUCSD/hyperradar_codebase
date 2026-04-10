#!/usr/bin/env python3
# tools/online_hd_test.py
# --------------------------------------------------------
# Online HD testing / adaptation runner for HyperRadar (OpenPCDet-style).
#
# Key idea:
#   - Run detector in eval mode
#   - Still run assign_targets() in eval (requires HD.ASSIGN_TARGETS_IN_EVAL=True)
#   - Use GT anchor labels (box_cls_labels) to update HD prototypes online
#   - Optionally do "retrain" correction for misclassified anchors
#
# Requirements:
#   - You have added pcdet/models/hd/hd_core.py
#   - You have modified AnchorHeadSingle to include HDCore and optional eval target assignment
#   - Your dense head forward_ret_dict contains:
#       - 'cls_preds' (final logits, [B,H,W,A*K])
#       - 'box_preds'
#       - 'box_cls_labels' and 'box_reg_targets' (when assign_targets is executed)
#
# Notes:
#   - This script assumes single-head anchor layout matches:
#       cls_preds.view(B, num_anchors, num_class)
#     and box_cls_labels is [B, num_anchors].
#   - Path-1 features: feat_mid = spatial_features_2d (per-cell feature),
#     expanded to anchor-level by sharing per-cell features across anchors.
# --------------------------------------------------------

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Online HD test (GT-based adaptation)")

    parser.add_argument("--cfg_file", type=str, required=True, help="Config file for the detector")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path for the detector weights")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (online typically 1)")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--extra_tag", type=str, default="online_hd", help="Extra tag for output directory")
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Optional ckpt dir (OpenPCDet style)")

    # HD runtime overrides
    parser.add_argument("--hd_mode", type=str, default=None,
                        choices=["baseline", "hd_only", "fused"],
                        help="Override HD fusion mode")
    parser.add_argument("--hd_lambda", type=float, default=None, help="Override fusion lambda")

    # Online update options
    parser.add_argument("--update", type=str, default="train",
                        choices=["none", "train", "retrain", "both"],
                        help="Online update policy")
    parser.add_argument("--alpha", type=float, default=1.0, help="Update step size for prototype updates")
    parser.add_argument("--percentage", type=float, default=None, help="Override sampler percentage (pos anchors)")
    parser.add_argument("--hard_ratio", type=float, default=None, help="Override hard ratio in sampling")
    parser.add_argument("--min_pos", type=int, default=None, help="Override minimum positives to keep per batch")

    parser.add_argument("--hard_from", type=str, default="origin",
                        choices=["origin", "final", "hd"],
                        help="Which logits to compute hardness (for sampling / retrain)")
    parser.add_argument("--normalize_every", type=int, default=None,
                        help="Override normalize_every in HDCore")
    parser.add_argument("--max_iters", type=int, default=-1, help="Max iterations to run (-1 = all)")
    parser.add_argument("--log_interval", type=int, default=20, help="Logging interval")

    # Memory persistence
    parser.add_argument("--load_memory", type=str, default=None, help="Load HD memory state from file")
    parser.add_argument("--save_memory", type=str, default=None, help="Save HD memory state to file (end of run)")
    parser.add_argument("--save_memory_every", type=int, default=0,
                        help="If >0, save memory every N iters (use save_memory as prefix)")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()
    return args


def find_hd_core(model) -> Optional[torch.nn.Module]:
    """
    Find the first submodule that has attribute 'hd_core'.
    This supports:
      - model.dense_head.hd_core
      - model.dense_head.heads_list[i].hd_core
      - other nested layouts
    """
    for m in model.modules():
        if hasattr(m, "hd_core") and getattr(m, "hd_core") is not None:
            return getattr(m, "hd_core")
    return None


def infer_anchor_layout(
    dense_head,
    spatial_features_2d: torch.Tensor,
    cls_preds_hwk: torch.Tensor
) -> Tuple[int, int, int, int, int]:
    """
    Infer B, H, W, A, K for cls_preds shaped [B,H,W,A*K].
    """
    B, C_in, H, W = spatial_features_2d.shape
    A = int(getattr(dense_head, "num_anchors_per_location"))
    K = int(getattr(dense_head, "num_class"))
    # Sanity check: cls last dim should be A*K
    assert cls_preds_hwk.shape[0] == B and cls_preds_hwk.shape[1] == H and cls_preds_hwk.shape[2] == W, \
        f"Shape mismatch: spatial_features_2d is (B,H,W)=({B},{H},{W}), cls_preds is {tuple(cls_preds_hwk.shape)}"
    assert cls_preds_hwk.shape[3] == A * K, \
        f"Expected cls_preds last dim A*K={A*K}, got {cls_preds_hwk.shape[3]}"
    return B, H, W, A, K


@torch.no_grad()
def build_anchor_features_from_cell_features(
    spatial_features_2d: torch.Tensor,
    num_anchors_per_loc: int
) -> torch.Tensor:
    """
    Path-1: feat_mid = spatial_features_2d (per-cell feature), expanded to anchor-level
    by sharing per-cell feature across anchors.

    Args:
        spatial_features_2d: [B, C, H, W]
        num_anchors_per_loc: A

    Returns:
        feat_anchor: [B, num_anchors, C] where num_anchors = H*W*A
    """
    B, C, H, W = spatial_features_2d.shape
    A = int(num_anchors_per_loc)

    # [B, H, W, C]
    cell_feat = spatial_features_2d.permute(0, 2, 3, 1).contiguous()
    # [B, H, W, A, C] (expand does not allocate new memory)
    anchor_feat = cell_feat.unsqueeze(3).expand(B, H, W, A, C)
    # [B, H*W*A, C]
    feat_anchor = anchor_feat.reshape(B, H * W * A, C).contiguous()
    return feat_anchor


@torch.no_grad()
def select_update_indices(
    labels: torch.Tensor,               # [num_anchors]
    logits_for_hard: torch.Tensor,      # [num_anchors, K]
    percentage: float,
    hard_ratio: float,
    min_pos: int
) -> torch.Tensor:
    """
    Hard + random sampling over positive anchors based on "hardness" = best_other - true_score (larger = harder).
    """
    # positives: label > 0; ignore: label < 0; bg: label == 0
    pos_mask = labels > 0
    pos_idx = torch.nonzero(pos_mask, as_tuple=False).view(-1)
    if pos_idx.numel() == 0:
        return labels.new_empty((0,), dtype=torch.long)

    pos_labels = labels[pos_idx].long()                 # [Npos]
    pos_logits = logits_for_hard[pos_idx]               # [Npos, K]

    # Gather true score
    true_score = pos_logits.gather(1, pos_labels.view(-1, 1)).squeeze(1)  # [Npos]

    # Mask true class and take best-other
    mask = torch.zeros_like(pos_logits, dtype=torch.bool)
    mask.scatter_(1, pos_labels.view(-1, 1), True)
    other = pos_logits.masked_fill(mask, float("-inf"))
    best_other = other.max(dim=1).values
    hardness = best_other - true_score                  # [Npos], larger => harder

    Npos = pos_idx.numel()
    keep_n = max(int(Npos * float(percentage)), int(min_pos))
    keep_n = min(keep_n, Npos)
    if keep_n <= 0:
        return labels.new_empty((0,), dtype=torch.long)

    hard_n = int(keep_n * float(hard_ratio))
    hard_n = min(hard_n, keep_n)
    rand_n = keep_n - hard_n

    # Hard: top hardness
    _, order = torch.sort(hardness, descending=True)
    hard_sel = order[:hard_n]

    if rand_n <= 0:
        sel_pos = hard_sel
    else:
        # Random from the remaining positives
        remain_mask = torch.ones(Npos, device=labels.device, dtype=torch.bool)
        remain_mask[hard_sel] = False
        remain = torch.nonzero(remain_mask, as_tuple=False).view(-1)
        if remain.numel() == 0:
            sel_pos = hard_sel
        elif remain.numel() <= rand_n:
            sel_pos = torch.cat([hard_sel, remain], dim=0)
        else:
            perm = torch.randperm(remain.numel(), device=labels.device)
            sel_pos = torch.cat([hard_sel, remain[perm[:rand_n]]], dim=0)

    selected_anchor_indices = pos_idx[sel_pos]  # indices in [0..num_anchors-1]
    return selected_anchor_indices


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])  # used by logger

    output_dir = Path(cfg.ROOT_DIR) / "output" / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"log_online_hd_{time.strftime('%Y%m%d-%H%M%S')}.txt"
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Config: {args.cfg_file}")
    logger.info(f"Checkpoint: {args.ckpt}")

    # Force some HD-friendly runtime switches if they exist in cfg:
    #  - enable assign_targets in eval for GT online updates
    try:
        if cfg.MODEL.DENSE_HEAD.get("HD", None) is not None:
            cfg.MODEL.DENSE_HEAD.HD.ASSIGN_TARGETS_IN_EVAL = True
    except Exception:
        pass

    # Build dataloader (test mode)
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False
    )

    # Build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.cuda()
    model.eval()

    # Load checkpoint (OpenPCDet-style)
    if hasattr(model, "load_params_from_file"):
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    else:
        # fallback
        ckpt = torch.load(args.ckpt, map_location="cpu")
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=False)
        elif "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

    dense_head = getattr(model, "dense_head", None)
    if dense_head is None:
        raise RuntimeError("Cannot find model.dense_head. This script assumes an anchor-based dense head.")

    hd_core = find_hd_core(model)
    if hd_core is None:
        raise RuntimeError("Cannot find hd_core in the model. Did you modify AnchorHeadSingle and add HDCore?")

    logger.info("Found HDCore. Online adaptation enabled.")

    # Optional runtime overrides for HD fusion
    if args.hd_mode is not None:
        try:
            hd_core.cfg.mode = args.hd_mode
            logger.info(f"Override HD mode: {args.hd_mode}")
        except Exception:
            logger.warning("Failed to override HD mode (cfg missing).")

    if args.hd_lambda is not None:
        try:
            hd_core.cfg.lam = float(args.hd_lambda)
            logger.info(f"Override HD lambda: {args.hd_lambda}")
        except Exception:
            logger.warning("Failed to override HD lambda (cfg missing).")

    # Optional runtime overrides for sampling/update params
    if args.percentage is not None:
        hd_core.cfg.sample_percentage = float(args.percentage)
    if args.hard_ratio is not None:
        hd_core.cfg.hard_ratio = float(args.hard_ratio)
    if args.min_pos is not None:
        hd_core.cfg.min_pos = int(args.min_pos)
    if args.normalize_every is not None:
        hd_core.cfg.normalize_every = int(args.normalize_every)

    # Load memory if provided
    if args.load_memory is not None:
        logger.info(f"Loading HD memory from: {args.load_memory}")
        hd_core.load_memory(args.load_memory, strict=False, map_location="cpu")

    # Ensure prototypes normalized before run
    with torch.no_grad():
        hd_core.memory.normalize_()

    update_policy = args.update.lower()
    logger.info(f"Online update policy: {update_policy}")
    logger.info(f"Hardness logits source: {args.hard_from}")
    logger.info(f"Sampler: percentage={hd_core.cfg.sample_percentage}, hard_ratio={hd_core.cfg.hard_ratio}, min_pos={hd_core.cfg.min_pos}")
    logger.info(f"Normalize every: {hd_core.cfg.normalize_every}")

    # Main loop
    start_time = time.time()
    iters = 0
    total_pos = 0
    total_sel = 0
    total_wrong = 0

    for batch_idx, data_dict in enumerate(test_loader):
        if args.max_iters > 0 and iters >= args.max_iters:
            break

        iters += 1
        load_data_to_gpu(data_dict)

        with torch.no_grad():
            # Forward
            pred_dicts, ret_dict = model(data_dict)

            # Access forward_ret_dict from dense head
            fr = dense_head.forward_ret_dict

            if "box_cls_labels" not in fr:
                # If you are in eval mode, this appears only if ASSIGN_TARGETS_IN_EVAL is enabled and gt_boxes exists.
                logger.warning(
                    "Missing 'box_cls_labels' in forward_ret_dict. "
                    "Ensure HD.ASSIGN_TARGETS_IN_EVAL=True and dataset provides gt_boxes in eval."
                )
                continue

            # Gather required tensors
            # cls_preds: [B,H,W,A*K] (final, after fusion)
            cls_preds_final_hwk = fr["cls_preds"]
            cls_preds_origin_hwk = fr.get("cls_preds_origin", None)
            cls_preds_hd_hwk = fr.get("cls_preds_hd", None)

            box_cls_labels = fr["box_cls_labels"]  # [B, num_anchors]

            # Need spatial_features_2d to build Path-1 anchor features
            # Usually model keeps it in data_dict; if not, you can enable EXPORT_FOR_ONLINE and use 'hd_cell_feat'.
            spatial_features_2d = data_dict.get("spatial_features_2d", None)
            if spatial_features_2d is None:
                spatial_features_2d = data_dict.get("hd_cell_feat", None)
            if spatial_features_2d is None:
                logger.warning(
                    "Cannot find 'spatial_features_2d' in data_dict. "
                    "Enable HD.EXPORT_FOR_ONLINE to export 'hd_cell_feat' from dense head."
                )
                continue

            # Infer layout
            B, H, W, A, K = infer_anchor_layout(dense_head, spatial_features_2d, cls_preds_final_hwk)
            num_anchors = H * W * A

            # Convert logits to [B, num_anchors, K]
            cls_final = cls_preds_final_hwk.view(B, num_anchors, K)

            if cls_preds_origin_hwk is not None:
                cls_origin = cls_preds_origin_hwk.view(B, num_anchors, K)
            else:
                cls_origin = cls_final  # fallback

            if cls_preds_hd_hwk is not None:
                cls_hd = cls_preds_hd_hwk.view(B, num_anchors, K)
            else:
                cls_hd = None

            # Select which logits to use for hardness / retrain
            if args.hard_from == "origin":
                logits_for_hard = cls_origin
            elif args.hard_from == "final":
                logits_for_hard = cls_final
            elif args.hard_from == "hd":
                if cls_hd is None:
                    logits_for_hard = cls_final
                else:
                    logits_for_hard = cls_hd
            else:
                logits_for_hard = cls_origin

            # Build anchor-level features: [B, num_anchors, C_in]
            feat_anchor_bnc = build_anchor_features_from_cell_features(spatial_features_2d, A)

            # Per-batch online update
            for b in range(B):
                labels_b = box_cls_labels[b].long()          # [num_anchors]
                logits_b = logits_for_hard[b].float()        # [num_anchors, K]
                feat_b = feat_anchor_bnc[b]                  # [num_anchors, C_in]

                # Positives / ignore stats
                pos_cnt = int((labels_b > 0).sum().item())
                total_pos += pos_cnt
                if pos_cnt == 0:
                    continue

                # Sampling indices in [0..num_anchors-1]
                sel_idx = select_update_indices(
                    labels=labels_b,
                    logits_for_hard=logits_b,
                    percentage=float(hd_core.cfg.sample_percentage),
                    hard_ratio=float(hd_core.cfg.hard_ratio),
                    min_pos=int(hd_core.cfg.min_pos),
                )
                total_sel += int(sel_idx.numel())

                # Train-update: prototype[y] += hv
                if update_policy in ["train", "both"]:
                    # Apply multiple steps if configured
                    steps = max(int(hd_core.cfg.update_steps), 1)
                    for _ in range(steps):
                        hd_core.update_train(
                            feat_mid=feat_b,               # [num_anchors, C]
                            labels=labels_b,               # [num_anchors]
                            selected_anchor_indices=sel_idx,
                            alpha=float(args.alpha),
                        )

                # Retrain-update: for wrong samples, true += hv, pred -= hv
                if update_policy in ["retrain", "both"]:
                    # Count wrong on selected positives (for logging)
                    pred_sel = logits_b[sel_idx].argmax(dim=1).long()
                    true_sel = labels_b[sel_idx].long()
                    wrong = pred_sel != true_sel
                    total_wrong += int(wrong.sum().item())

                    hd_core.update_retrain(
                        feat_mid=feat_b,
                        labels=labels_b,
                        logits_origin=logits_b,
                        selected_anchor_indices=sel_idx,
                        alpha=float(args.alpha),
                    )

            # Optional periodic memory save
            if args.save_memory_every > 0 and args.save_memory is not None:
                if iters % int(args.save_memory_every) == 0:
                    save_path = f"{args.save_memory}.iter{iters}.pth"
                    hd_core.save_memory(save_path)
                    logger.info(f"[Iter {iters}] Saved HD memory to: {save_path}")

        # Logging
        if iters % args.log_interval == 0:
            elapsed = time.time() - start_time
            pos_avg = total_pos / max(iters, 1)
            sel_avg = total_sel / max(iters, 1)
            wrong_avg = total_wrong / max(iters, 1)
            logger.info(
                f"[Iter {iters}] elapsed={elapsed:.1f}s | "
                f"pos_avg={pos_avg:.1f} | sel_avg={sel_avg:.1f} | wrong_avg={wrong_avg:.1f} | "
                f"HD(mode={getattr(hd_core.cfg, 'mode', 'NA')}, lam={getattr(hd_core.cfg, 'lam', 'NA')})"
            )

    # End of run
    total_time = time.time() - start_time
    logger.info(f"Done. iters={iters}, total_time={total_time:.1f}s")

    if args.save_memory is not None:
        logger.info(f"Saving final HD memory to: {args.save_memory}")
        hd_core.save_memory(args.save_memory)

    logger.info("Finished online HD test.")


if __name__ == "__main__":
    main()
