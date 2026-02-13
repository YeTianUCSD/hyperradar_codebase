import _init_path
import argparse
import datetime
import os
import re
from pathlib import Path
import warnings
import logging, sys

warnings.filterwarnings("ignore")

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description="Build HD memory (prototypes) on train split (cell-level)")

    parser.add_argument("--cfg_file", type=str, required=True,
                        help="cfg file, e.g. cfgs/kitti_models/pointpillar_vod_hd.yaml")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="checkpoint path, e.g. .../checkpoint_epoch_110.pth")

    parser.add_argument("--batch_size", type=int, default=None, help="batch size")
    parser.add_argument("--workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--extra_tag", type=str, default="default", help="extra tag for output dir")

    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"], default="none")
    parser.add_argument("--tcp_port", type=int, default=18888)
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER,
                        help="set extra config keys if needed")

    # Memory build options
    parser.add_argument("--save_path", type=str, default=None,
                        help="explicit path to save hd_memory.pth. If None, save under output/.../hd_memory/")
    parser.add_argument("--max_batches", type=int, default=-1,
                        help="limit number of batches for quick debug (-1 means full pass)")
    parser.add_argument("--use_fp16", action="store_true", default=False,
                        help="use autocast fp16 for forward (feature extraction)")
    parser.add_argument("--chunk_cells", type=int, default=65536,
                        help="chunk size for cell features when encoding (avoid peak memory)")

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])

    np.random.seed(1024)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def _infer_epoch_id_from_ckpt(ckpt_path: str) -> str:
    """Extract epoch id from checkpoint filename."""
    num_list = re.findall(r"\d+", ckpt_path)
    return num_list[-1] if len(num_list) > 0 else "no_number"


def _get_num_anchors_per_loc(dense_head) -> int:
    """Get anchors per spatial location from dense head."""
    if hasattr(dense_head, "num_anchors_per_location"):
        v = dense_head.num_anchors_per_location
        if isinstance(v, (list, tuple)):
            return int(sum(v))
        return int(v)
    raise RuntimeError("dense_head has no num_anchors_per_location attribute.")


@torch.no_grad()
def _ensure_hd_memory_ready(dense_head, logger):
    """Ensure dense_head.hd_core and its memory exist."""
    hd_core = getattr(dense_head, "hd_core", None)
    if hd_core is None:
        raise RuntimeError("dense_head.hd_core is None. Enable HD in cfg and rebuild model.")
    if not hasattr(hd_core, "embedder") or not hasattr(hd_core, "memory"):
        raise RuntimeError("hd_core missing embedder/memory. Please use your HDCore implementation.")
    mem = hd_core.memory
    if not hasattr(mem, "classify_weights") or not hasattr(mem, "prototypes"):
        raise RuntimeError("hd_core.memory missing classify_weights/prototypes.")

    logger.info(
        f"HD memory tensors: classify_weights={tuple(mem.classify_weights.shape)}, "
        f"prototypes={tuple(mem.prototypes.shape)}"
    )
    return hd_core



@torch.no_grad()
def _accumulate_from_batch_cell_level(
    dense_head,
    spatial_features_2d: torch.Tensor,
    box_cls_labels: torch.Tensor,
    anchors_per_loc: int,
    num_classes: int,
    chunk_cells: int,
    logger=None
):
    """
    Accumulate prototypes using cell features (NO anchor feature expansion).

    spatial_features_2d: [B, C, H, W]
    box_cls_labels:      [B, H*W*A]  (A fastest)
      - background: 0
      - ignored:    -1
      - positives:  1..K

    We reshape labels -> [B, H*W, A], and for each anchor index a,
    update memory using the SAME cell feature for anchors whose label is positive.

    This matches AnchorHeadSingle behavior:
      - HD logits are computed per cell then repeated across anchors.
    """
    hd_core = dense_head.hd_core
    embedder = hd_core.embedder
    memory = hd_core.memory

    assert spatial_features_2d.dim() == 4
    B, C, H, W = spatial_features_2d.shape
    HW = H * W
    A = int(anchors_per_loc)
    K = int(num_classes)

    # [B, C, H, W] -> [B, H, W, C] -> [B, HW, C]
    cell_feat = spatial_features_2d.permute(0, 2, 3, 1).contiguous().view(B, HW, C)

    # labels [B, HW*A] -> [B, HW, A]
    if box_cls_labels.shape[1] != HW * A:
        raise RuntimeError(
            f"Label/feature mismatch: box_cls_labels={tuple(box_cls_labels.shape)}, expected second dim={HW*A}. "
            f"Got H={H}, W={W}, A={A}. Check anchor order / feature map stride."
        )
    labels_hw_a = box_cls_labels.view(B, HW, A).long()

    total_pos = 0

    for a in range(A):
        lab = labels_hw_a[:, :, a]  # [B, HW]
        pos_mask = lab > 0  # positives only (1..K)
        if not pos_mask.any():
            continue

        feat_sel = cell_feat[pos_mask]     # [Npos, C]
        lab_sel = lab[pos_mask] - 1        # -> [0..K-1]

        if (lab_sel.min() < 0) or (lab_sel.max() >= K):
            raise RuntimeError(
                f"Shifted labels out of range: min={lab_sel.min().item()}, max={lab_sel.max().item()}, K={K}"
            )

        Npos = feat_sel.shape[0]
        if chunk_cells <= 0:
            chunk_cells = Npos

        start = 0
        while start < Npos:
            end = min(start + chunk_cells, Npos)
            f_chunk = feat_sel[start:end]
            y_chunk = lab_sel[start:end]

            hv = embedder(f_chunk)              # [n, HD_DIM]
            memory.add_(y_chunk, hv, alpha=1.0) # class accumulator

            start = end

        total_pos += int(Npos)

    memory.normalize_()
    return total_pos


def main():
    args, cfg_local = parse_config()

    # Distributed init (same as tools/test.py)
    if args.launcher == "none":
        dist = False
    else:
        _, cfg_local.LOCAL_RANK = getattr(common_utils, "init_dist_%s" % args.launcher)(
            args.tcp_port, args.local_rank, backend="nccl"
        )
        dist = True

    if args.batch_size is None:
        args.batch_size = cfg_local.OPTIMIZATION.BATCH_SIZE_PER_GPU

    output_dir = cfg_local.ROOT_DIR / "output" / cfg_local.EXP_GROUP_PATH / cfg_local.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_id = _infer_epoch_id_from_ckpt(args.ckpt)

    if args.save_path is None:
        mem_dir = output_dir / "hd_memory"
        mem_dir.mkdir(parents=True, exist_ok=True)
        save_path = mem_dir / f"hd_memory_epoch_{epoch_id}.pth"
    else:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Logger
    log_file = output_dir / "hd_memory" / f"log_build_hd_memory_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(log_file, rank=cfg_local.LOCAL_RANK)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    logger.propagate = False
    logger.info("[STDOUT] logger stream handler attached")

    logger.info("********************** Start build_hd_memory.py **********************")
    logger.info(f"cfg_file: {args.cfg_file}")
    logger.info(f"ckpt: {args.ckpt}")
    logger.info(f"save_path: {str(save_path)}")
    logger.info(f"batch_size: {args.batch_size}, workers: {args.workers}, launcher: {args.launcher}")
    logger.info(f"max_batches: {args.max_batches}, use_fp16: {args.use_fp16}, chunk_cells: {args.chunk_cells}")
    log_config_to_file(cfg_local, logger=logger)

    # Build TRAIN split dataloader
    train_set, train_loader, _sampler = build_dataloader(
        dataset_cfg=cfg_local.DATA_CONFIG,
        class_names=cfg_local.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist,
        workers=args.workers,
        logger=logger,
        training=True,
    )

    model = build_network(model_cfg=cfg_local.MODEL, num_class=len(cfg_local.CLASS_NAMES), dataset=train_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist)
    model.cuda()
    model.eval()

    dense_head = getattr(model, "dense_head", None)
    if dense_head is None:
        raise RuntimeError("Model has no `dense_head`. Cannot build HD memory.")

    anchors_per_loc = _get_num_anchors_per_loc(dense_head)
    num_classes = len(cfg_local.CLASS_NAMES)
    logger.info(f"anchors_per_loc: {anchors_per_loc}, num_classes: {num_classes}")

    hd_core = _ensure_hd_memory_ready(dense_head, logger)
    hd_core.memory.reset()

    total_pos = 0
    total_seen = 0
    use_autocast = bool(args.use_fp16)

    with torch.no_grad():
        for it, batch_dict in enumerate(train_loader):
            if args.max_batches > 0 and it >= args.max_batches:
                break

            # Move to GPU
            try:
                from pcdet.models import load_data_to_gpu as _load_data_to_gpu
                _load_data_to_gpu(batch_dict)
            except Exception:
                for k, v in batch_dict.items():
                    if torch.is_tensor(v):
                        batch_dict[k] = v.cuda(non_blocking=True)

            # Forward once
            if use_autocast:
                with torch.cuda.amp.autocast(enabled=True):
                    model_ret = model(batch_dict)
            else:
                model_ret = model(batch_dict)

            # ---- reduce log spam: every 1000 iters ----
            if cfg_local.LOCAL_RANK == 0 and (it % 1000 == 0):
                logger.info(
                    f"[Iter {it}] type(model_return)={type(model_ret)} "
                    f"keys_in_batch_dict_head={list(batch_dict.keys())[:30]}"
                )

            data_dict = batch_dict

            if "spatial_features_2d" not in data_dict:
                if isinstance(model_ret, dict):
                    data_dict = model_ret
                elif isinstance(model_ret, (tuple, list)):
                    for item in model_ret:
                        if isinstance(item, dict) and "spatial_features_2d" in item:
                            data_dict = item
                            break

            spatial = data_dict.get("spatial_features_2d", None)
            if spatial is None:
                raise RuntimeError(
                    "Cannot find 'spatial_features_2d'. "
                    "Model forward did not populate it in batch_dict nor return a dict containing it."
                )

            gt_boxes = data_dict.get("gt_boxes", None)
            if gt_boxes is None:
                gt_boxes = batch_dict.get("gt_boxes", None)
            if gt_boxes is None:
                raise RuntimeError("Cannot find 'gt_boxes'. Cannot assign anchor labels.")

            targets = dense_head.assign_targets(gt_boxes=gt_boxes)
            if "box_cls_labels" not in targets:
                raise RuntimeError("assign_targets() output has no key 'box_cls_labels'.")

            box_cls_labels = targets["box_cls_labels"]  # [B, num_anchors]

            # ---- label sanity log: only first 2 iters ----
            if it < 2 and cfg_local.LOCAL_RANK == 0:
                lab = box_cls_labels
                u = torch.unique(lab)
                logger.info(
                    f"[LABEL] box_cls_labels shape={tuple(lab.shape)} "
                    f"min={lab.min().item()} max={lab.max().item()} unique={u.tolist()}"
                )
                logger.info(
                    f"[LABEL] counts: ignore(-1)={(lab==-1).sum().item()} "
                    f"bg(0)={(lab==0).sum().item()} pos(>0)={(lab>0).sum().item()}"
                )
                if (lab > 0).any():
                    lab_shift = lab[lab > 0] - 1
                    logger.info(
                        f"[LABEL] shifted_pos_labels (lab-1): "
                        f"min={lab_shift.min().item()} max={lab_shift.max().item()} "
                        f"expected_range=[0,{num_classes-1}]"
                    )

            cared = box_cls_labels.view(-1) >= 0
            total_seen += int(cared.sum().item())

            pos_added = _accumulate_from_batch_cell_level(
                dense_head=dense_head,
                spatial_features_2d=spatial,
                box_cls_labels=box_cls_labels,
                anchors_per_loc=anchors_per_loc,
                num_classes=num_classes,
                chunk_cells=int(args.chunk_cells),
                logger=logger
            )
            total_pos += int(pos_added)

            # optional progress (also avoid spam)
            if cfg_local.LOCAL_RANK == 0 and ((it + 1) % 1000 == 0):
                logger.info(f"[Iter {it+1}] accumulated_pos={total_pos}, total_seen={total_seen}")

    # Final normalize + sanity
    hd_core.memory.normalize_()
    if cfg_local.LOCAL_RANK == 0:
        mem = hd_core.memory
        try:
            logger.info(
                f"[MEM] proto_norm_mean={mem.prototypes.norm(dim=1).mean().item():.6f}, "
                f"w_norm_mean={mem.classify_weights.norm(dim=1).mean().item():.6f}"
            )
        except Exception:
            pass

        # ---------------- Save FULL payload (embedder + memory) ----------------
    # This avoids projection mismatch between build and inference.
    embedder_sd = None
    try:
        embedder_sd = hd_core.embedder.state_dict()
    except Exception as e:
        logger.warning(f"[SAVE] cannot get embedder.state_dict(): {repr(e)}")

    memory_sd = None
    try:
        memory_sd = hd_core.memory.state_dict()
    except Exception as e:
        logger.warning(f"[SAVE] cannot get memory.state_dict(): {repr(e)}")

    # Optional: add a quick fingerprint so you can sanity check consistency
    def _tensor_fingerprint(x: torch.Tensor, n: int = 16):
        x = x.detach().float().flatten()
        if x.numel() == 0:
            return {"numel": 0}
        idx = torch.linspace(0, x.numel() - 1, steps=min(n, x.numel())).long()
        vals = x[idx].cpu().numpy().tolist()
        return {
            "numel": int(x.numel()),
            "mean": float(x.mean().item()),
            "std": float(x.std().item()),
            "samples": vals,
        }

    proj_fp = None
    try:
        if hasattr(hd_core.embedder, "projection") and hasattr(hd_core.embedder.projection, "weight"):
            proj_fp = _tensor_fingerprint(hd_core.embedder.projection.weight, n=16)
    except Exception:
        proj_fp = None

    save_obj = {
        # Full payload (preferred by loader)
        "embedder": embedder_sd,   # state_dict
        "memory": memory_sd,       # state_dict

        # Keep the old flat keys too (backward compatible)
        "classify_weights": hd_core.memory.classify_weights.detach().float().cpu(),
        "prototypes": hd_core.memory.prototypes.detach().float().cpu(),

        "meta": {
            "source": "hd_core.memory (cell-level accumulation, repeated to anchors)",
            "num_classes": num_classes,
            "anchors_per_loc": int(anchors_per_loc),
            "ckpt": args.ckpt,
            "cfg_file": args.cfg_file,
            "epoch_id": epoch_id,
            "total_pos_anchors": int(total_pos),
            "total_seen_anchors": int(total_seen),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            # add important HD cfg snapshots for debugging
            "hd_cfg": {
                "HD_DIM": int(getattr(cfg_local.MODEL.DENSE_HEAD.HD, "HD_DIM", -1)),
                "ENCODER": str(getattr(cfg_local.MODEL.DENSE_HEAD.HD, "ENCODER", "unknown")),
                "QUANTIZE": bool(getattr(cfg_local.MODEL.DENSE_HEAD.HD, "QUANTIZE", False)),
                "TEMPERATURE": float(getattr(cfg_local.MODEL.DENSE_HEAD.HD, "TEMPERATURE", 1.0)),
                "SEED": int(getattr(cfg_local.MODEL.DENSE_HEAD.HD, "SEED", 0)),
            },
            "projection_fingerprint": proj_fp,
        }
    }

    torch.save(save_obj, str(save_path))
    logger.info(f"Saved HD memory to: {str(save_path)}")
    logger.info(f"meta: {save_obj['meta']}")

    logger.info(f"Saved HD memory to: {str(save_path)}")
    logger.info(f"meta: {save_obj['meta']}")
    logger.info("********************** Finished build_hd_memory.py **********************")


if __name__ == "__main__":
    main()
