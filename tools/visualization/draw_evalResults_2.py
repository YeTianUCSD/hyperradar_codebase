#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example:
python /home/code/hyperradar/hyperradar_codebase/tools/visualization/draw_evalResults_2.py \
  --eval_root /home/code/hyperradar/hyperradar_codebase/output/kitti_models/pointpillar_vod_hd/run500_cls/eval \
  --out /home/code/hyperradar/hyperradar_codebase/runs/logs/recall_rcnn_curve_run500_cls.png \
  --show
"""

import re
import argparse
from pathlib import Path
from typing import Optional

# headless friendly
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


R03_PAT = re.compile(r"recall_rcnn_0\.3:\s*([0-9]*\.?[0-9]+)")
R05_PAT = re.compile(r"recall_rcnn_0\.5:\s*([0-9]*\.?[0-9]+)")
R07_PAT = re.compile(r"recall_rcnn_0\.7:\s*([0-9]*\.?[0-9]+)")


def parse_epoch_from_path(p: Path) -> Optional[int]:
    """
    Parse epoch id from path parts like .../epoch_12/val/default/log_eval_*.txt
    """
    for part in p.parts:
        if part.startswith("epoch_"):
            try:
                return int(part.split("_", 1)[1])
            except Exception:
                return None
    return None


def parse_single_log(log_path: Path):
    """
    Return (r03, r05, r07) or None if missing.
    Use the LAST occurrence in file (some logs might print multiple times).
    """
    text = log_path.read_text(errors="ignore")
    m03 = R03_PAT.findall(text)
    m05 = R05_PAT.findall(text)
    m07 = R07_PAT.findall(text)

    if not (m03 and m05 and m07):
        return None

    return float(m03[-1]), float(m05[-1]), float(m07[-1])


def collect_logs(eval_root: Path, pattern: str):
    """
    Find logs under eval_root. Default pattern targets your structure:
      epoch_*/val/default/log_eval_*.txt
    But we implement via rglob so it's robust.
    """
    # If user passes something like "epoch_*/val/default/log_eval_*.txt"
    # we can emulate by scanning all and filtering on Path.match.
    all_txt = list(eval_root.rglob("*.txt"))
    matched = [p for p in all_txt if p.match(pattern)]
    return matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True, help=".../runXXX/eval")
    ap.add_argument("--pattern", default="**/epoch_*/val/default/log_eval_*.txt",
                    help="glob-like Path.match pattern relative to eval_root (default matches your layout)")
    ap.add_argument("--out", default="recall_rcnn_curve.png", help="output figure path")
    ap.add_argument("--show", action="store_true", help="show interactive window")
    ap.add_argument("--verbose", action="store_true", help="print per-epoch parsing status")
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    if not eval_root.exists():
        raise FileNotFoundError(f"eval_root not found: {eval_root}")

    logs = collect_logs(eval_root, args.pattern)
    if len(logs) == 0:
        raise RuntimeError(f"No logs matched under {eval_root} with pattern: {args.pattern}")

    # For each epoch, keep the newest log file (mtime max)
    best_by_epoch: dict[int, Path] = {}
    for lp in logs:
        ep = parse_epoch_from_path(lp)
        if ep is None:
            continue
        if ep not in best_by_epoch:
            best_by_epoch[ep] = lp
        else:
            if lp.stat().st_mtime > best_by_epoch[ep].stat().st_mtime:
                best_by_epoch[ep] = lp

    epochs, r03, r05, r07 = [], [], [], []
    skipped = []

    for ep in sorted(best_by_epoch.keys()):
        lp = best_by_epoch[ep]
        vals = parse_single_log(lp)
        if vals is None:
            skipped.append((ep, str(lp)))
            if args.verbose:
                print(f"[SKIP] epoch={ep} missing recall_rcnn_0.x in {lp}")
            continue

        v03, v05, v07v = vals
        epochs.append(ep)
        r03.append(v03)
        r05.append(v05)
        r07.append(v07v)

        if args.verbose:
            print(f"[OK] epoch={ep} r03={v03:.6f} r05={v05:.6f} r07={v07v:.6f}  ({lp.name})")

    if len(epochs) == 0:
        raise RuntimeError("Parsed 0 epochs with complete recall_rcnn_0.3/0.5/0.7. Check log format or pattern.")

    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, r03, marker="o", label="recall_rcnn_0.3")
    plt.plot(epochs, r05, marker="o", label="recall_rcnn_0.5")
    plt.plot(epochs, r07, marker="o", label="recall_rcnn_0.7")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("RCNN Recall vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"[OK] Parsed {len(epochs)} epochs from folder: {eval_root}")
    print(f"[OK] Saved figure to: {out_path}")
    print("[Preview] First 5 points:", list(zip(epochs[:5], r03[:5], r05[:5], r07[:5])))

    if skipped:
        print(f"[Warn] Skipped {len(skipped)} epochs (missing keys). Use --verbose to see details.")

    if args.show:
        # If you're on a machine with display; otherwise keep --show off
        plt.show()


if __name__ == "__main__":
    main()
