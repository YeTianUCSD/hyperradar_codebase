#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
python /home/code/hyperradar/hyperradar_codebase/tools/visualization/draw_evalResults_3.py \
  --cls_eval_root /home/code/hyperradar/hyperradar_codebase/output/kitti_models/pointpillar_vod_hd/run500_cls/eval \
  --bev_log /home/code/hyperradar/hyperradar_codebase/runs/logs/eval_100epoch20260203-100632.log \
  --out /home/code/hyperradar/hyperradar_codebase/runs/logs/recall03_cls_vs_bev_epoch1-20.png \
  --epoch_min 1 --epoch_max 20 \
  --verbose

'''

import re
import argparse
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import matplotlib
matplotlib.use("Agg")  # 服务器无GUI更稳
import matplotlib.pyplot as plt


RECALL03_PAT = re.compile(r"recall_rcnn_0\.3:\s*([0-9]*\.?[0-9]+)")
EPOCH_HDR_PAT = re.compile(r"Performance of EPOCH\s+(\d+)", re.IGNORECASE)


def parse_recall03_from_text(text: str) -> Optional[float]:
    ms = RECALL03_PAT.findall(text)
    if not ms:
        return None
    return float(ms[-1])  # 取最后一次出现


def parse_epoch_from_path(p: Path) -> Optional[int]:
    # .../eval/epoch_12/val/default/log_eval_xxx.txt
    for part in p.parts:
        if part.startswith("epoch_"):
            try:
                return int(part.split("_", 1)[1])
            except Exception:
                return None
    return None


def newest_log_in_epoch_dir(epoch_dir: Path) -> Optional[Path]:
    # 默认匹配：epoch_x/val/default/log_eval_*.txt
    cands = list(epoch_dir.rglob("log_eval_*.txt"))
    if not cands:
        # fallback：任意txt
        cands = list(epoch_dir.rglob("*.txt"))
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1]


def load_cls_curve(eval_root: Path, e_min: int, e_max: int, verbose: bool) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for e in range(e_min, e_max + 1):
        epoch_dir = eval_root / f"epoch_{e}"
        if not epoch_dir.exists():
            if verbose:
                print(f"[CLS][SKIP] missing dir: {epoch_dir}")
            continue

        lp = newest_log_in_epoch_dir(epoch_dir)
        if lp is None:
            if verbose:
                print(f"[CLS][SKIP] no log under: {epoch_dir}")
            continue

        txt = lp.read_text(errors="ignore")
        v = parse_recall03_from_text(txt)
        if v is None:
            if verbose:
                print(f"[CLS][SKIP] no recall_rcnn_0.3 in: {lp}")
            continue

        out[e] = v
        if verbose:
            print(f"[CLS][OK] epoch={e} recall03={v:.6f}  ({lp})")
    return out


def load_bev_curve(big_log: Path, e_min: int, e_max: int, verbose: bool) -> Dict[int, float]:
    """
    Parse from a single big log that contains:
      *************** Performance of EPOCH k *****************
      recall_rcnn_0.3: ...
    """
    text = big_log.read_text(errors="ignore")

    cur_epoch: Optional[int] = None
    out: Dict[int, float] = {}

    for line in text.splitlines():
        m = EPOCH_HDR_PAT.search(line)
        if m:
            cur_epoch = int(m.group(1))
            continue

        if cur_epoch is None:
            continue

        if cur_epoch < e_min or cur_epoch > e_max:
            continue

        m = RECALL03_PAT.search(line)
        if m:
            out[cur_epoch] = float(m.group(1))  # 覆盖即可，取最后一次
            continue

    if verbose:
        for e in range(e_min, e_max + 1):
            if e in out:
                print(f"[BEV][OK] epoch={e} recall03={out[e]:.6f}")
            else:
                print(f"[BEV][SKIP] epoch={e} not found in big log")
    return out


def dict_to_series(d: Dict[int, float], e_min: int, e_max: int) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for e in range(e_min, e_max + 1):
        if e in d:
            xs.append(e)
            ys.append(d[e])
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls_eval_root", required=True, help=".../run500_cls/eval")
    ap.add_argument("--bev_log", required=True, help=".../runs/logs/eval_100epoch....log")
    ap.add_argument("--out", required=True, help="output png/pdf path")
    ap.add_argument("--epoch_min", type=int, default=1)
    ap.add_argument("--epoch_max", type=int, default=20)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cls_root = Path(args.cls_eval_root)
    bev_log = Path(args.bev_log)
    out_path = Path(args.out)

    if not cls_root.exists():
        raise FileNotFoundError(f"cls_eval_root not found: {cls_root}")
    if not bev_log.exists():
        raise FileNotFoundError(f"bev_log not found: {bev_log}")

    e_min, e_max = args.epoch_min, args.epoch_max

    cls_map = load_cls_curve(cls_root, e_min, e_max, args.verbose)
    bev_map = load_bev_curve(bev_log, e_min, e_max, args.verbose)

    x_cls, y_cls = dict_to_series(cls_map, e_min, e_max)
    x_bev, y_bev = dict_to_series(bev_map, e_min, e_max)

    if len(x_cls) == 0 and len(x_bev) == 0:
        raise RuntimeError("No points parsed for both cls and bev. Check paths/log format.")

    plt.figure(figsize=(10, 5))
    if len(x_cls) > 0:
        plt.plot(x_cls, y_cls, marker="o", label="cls")
    if len(x_bev) > 0:
        plt.plot(x_bev, y_bev, marker="o", label="bev")

    plt.xlabel("Epoch")
    plt.ylabel("Recall (rcnn @ 0.3)")
    plt.title("recall_0.3 vs epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200)
    print(f"[OK] saved figure -> {out_path}")

    if args.show:
        # 服务器无GUI可能会报错；需要的话你可以在本地跑 --show
        plt.show()


if __name__ == "__main__":
    main()
