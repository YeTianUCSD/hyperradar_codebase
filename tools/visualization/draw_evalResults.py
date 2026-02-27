#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
python /home/code/hyperradar/hyperradar_codebase/tools/visualization/draw_evalResults.py \
  --log /home/code/hyperradar/hyperradar_codebase/output/kitti_models/pointpillar_vod_hd_retrain_hdonly/run500_cls_retrain_hd/retrain_hdonly_from_CNN_checkpoint_epoch17_2026-02-14_120548.log \
  --out /home/code/hyperradar/hyperradar_codebase/output/kitti_models/pointpillar_vod_hd_retrain_hdonly/run500_cls_retrain_hd/retrain_hdonly_epoch50.png \
  --show
'''
import re
import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log(log_path: Path):
    text = log_path.read_text(errors="ignore")

    # 例子：
    # 2026-02-03 ... INFO  *************** Performance of EPOCH 1 *****************
    epoch_pat = re.compile(r"Performance of EPOCH\s+(\d+)", re.IGNORECASE)

    r03_pat = re.compile(r"recall_rcnn_0\.3:\s*([0-9]*\.?[0-9]+)")
    r05_pat = re.compile(r"recall_rcnn_0\.5:\s*([0-9]*\.?[0-9]+)")
    r07_pat = re.compile(r"recall_rcnn_0\.7:\s*([0-9]*\.?[0-9]+)")

    epochs, r03, r05, r07 = [], [], [], []

    cur_epoch = None
    cur_vals = {}

    for line in text.splitlines():
        m = epoch_pat.search(line)
        if m:
            # flush 上一个 epoch（如果完整）
            if cur_epoch is not None and {"0.3", "0.5", "0.7"} <= set(cur_vals.keys()):
                epochs.append(cur_epoch)
                r03.append(cur_vals["0.3"])
                r05.append(cur_vals["0.5"])
                r07.append(cur_vals["0.7"])

            cur_epoch = int(m.group(1))
            cur_vals = {}
            continue

        if cur_epoch is None:
            continue

        m = r03_pat.search(line)
        if m:
            cur_vals["0.3"] = float(m.group(1))
            continue
        m = r05_pat.search(line)
        if m:
            cur_vals["0.5"] = float(m.group(1))
            continue
        m = r07_pat.search(line)
        if m:
            cur_vals["0.7"] = float(m.group(1))
            continue

    # flush 最后一个 epoch
    if cur_epoch is not None and {"0.3", "0.5", "0.7"} <= set(cur_vals.keys()):
        epochs.append(cur_epoch)
        r03.append(cur_vals["0.3"])
        r05.append(cur_vals["0.5"])
        r07.append(cur_vals["0.7"])

    return epochs, r03, r05, r07


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log",
        required=True,
        help="path to evaluation log, e.g. /home/code/.../eval_100epoch....log",
    )
    ap.add_argument(
        "--out",
        default="recall_rcnn_curve.png",
        help="output figure file name (png/pdf/etc.)",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="show interactive window",
    )
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    epochs, r03, r05, r07 = parse_log(log_path)

    if len(epochs) == 0:
        raise RuntimeError(
            "Parsed 0 epochs. Check log format (must contain 'Performance of EPOCH k' and recall_rcnn_0.x lines)."
        )

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

    plt.savefig(args.out, dpi=200)
    print(f"[OK] Parsed {len(epochs)} epochs from: {log_path}")
    print(f"[OK] Saved figure to: {args.out}")
    print("[Preview] First 5 points:", list(zip(epochs[:5], r03[:5], r05[:5], r07[:5])))

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
