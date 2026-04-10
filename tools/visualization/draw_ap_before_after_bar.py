#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Draw grouped bar charts for AP comparison:
  - Pre-trained
  - Online updated

This version saves 3 figures:
  - Easy
  - Moderate
  - Hard

Each figure contains two subplots:
  - Strict IoU
  - Loose IoU

All numbers are embedded in this file so they can be edited directly later.

Example:
python /home/code/hyperradar/hyperradar_codebase/tools/visualization/draw_ap_before_after_bar.py \
  --out_dir /home/code/hyperradar/hyperradar_codebase/tools/visualization/online_update_compare
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPERIMENT_DATA: Dict[str, Dict[str, Dict[str, List[float]]]] = {
    "strict": {
        "title": "Strict IoU 3D AP",
        "subtitle": "Car@0.70 / Pedestrian@0.50 / Cyclist@0.50",
        "pretrained": {
            "Car": [1.9534, 5.1734, 4.1769],
            "Pedestrian": [3.1984, 2.6043, 2.1668],
            "Cyclist": [3.1738, 2.8051, 2.4522],
        },
        "online_updated": {
            "Car": [3.1889, 7.9528, 6.7804],
            "Pedestrian": [2.8365, 2.7982, 2.4425],
            "Cyclist": [2.8655, 2.7338, 2.4193],
        },
    },
    "loose": {
        "title": "Loose IoU 3D AP",
        "subtitle": "Car@0.50 / Pedestrian@0.25 / Cyclist@0.25",
        "pretrained": {
            "Car": [14.1729, 27.4622, 22.7976],
            "Pedestrian": [23.2627, 21.6553, 19.2601],
            "Cyclist": [36.6540, 32.6724, 28.9865],
        },
        "online_updated": {
            "Car": [20.8006, 39.5165, 33.1556],
            "Pedestrian": [23.3646, 21.9872, 20.5099],
            "Cyclist": [42.8288, 39.6498, 36.1601],
        },
    },
}


CLASS_ORDER = ["Car", "Pedestrian", "Cyclist"]
METHOD_ORDER = ["pretrained", "online_updated"]
METHOD_LABELS = {
    "pretrained": "Pre-trained",
    "online_updated": "Online updated",
}
DIFFICULTIES = ["Easy", "Moderate", "Hard"]


def parse_args():
    parser = argparse.ArgumentParser(description="Draw grouped bar charts for AP before/after online update.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/code/hyperradar/hyperradar_codebase/tools/visualization/online_update_compare",
        help="Directory to save output figures",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI",
    )
    return parser.parse_args()


def _validate_data():
    for metric_name, metric_cfg in EXPERIMENT_DATA.items():
        for method_name in METHOD_ORDER:
            if method_name not in metric_cfg:
                raise RuntimeError(f"{metric_name} missing method '{method_name}'")
            for cls_name in CLASS_ORDER:
                if cls_name not in metric_cfg[method_name]:
                    raise RuntimeError(f"{metric_name}/{method_name} missing class '{cls_name}'")
                vals = metric_cfg[method_name][cls_name]
                if not isinstance(vals, (list, tuple)) or len(vals) != 3:
                    raise RuntimeError(
                        f"{metric_name}/{method_name}/{cls_name} must be a 3-value list [Easy, Moderate, Hard]"
                    )


def _values_for(metric_cfg: Dict, method_name: str, difficulty_idx: int) -> List[float]:
    return [float(metric_cfg[method_name][cls_name][difficulty_idx]) for cls_name in CLASS_ORDER]


def _draw_subplot(ax, metric_cfg: Dict, difficulty_idx: int):
    x = np.arange(len(CLASS_ORDER), dtype=float)
    width = 0.34

    before_vals = _values_for(metric_cfg, "pretrained", difficulty_idx)
    after_vals = _values_for(metric_cfg, "online_updated", difficulty_idx)

    bars_before = ax.bar(
        x - width / 2,
        before_vals,
        width,
        label=METHOD_LABELS["pretrained"],
        color="#6c8ebf",
        edgecolor="#315b8a",
        linewidth=1.2,
    )
    bars_after = ax.bar(
        x + width / 2,
        after_vals,
        width,
        label=METHOD_LABELS["online_updated"],
        color="#e28743",
        edgecolor="#a65118",
        linewidth=1.2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylabel("AP")
    ax.set_title(f"{metric_cfg['title']}\n{metric_cfg['subtitle']}")
    ax.grid(axis="y", alpha=0.25, linestyle="--")

    ymax = max(before_vals + after_vals)
    ax.set_ylim(0, ymax * 1.24 if ymax > 0 else 1.0)

    for bars in (bars_before, bars_after):
        for rect in bars:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                h + ymax * 0.03 if ymax > 0 else 0.03,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    return bars_before, bars_after


def _draw_one_difficulty(difficulty_idx: int, out_dir: Path, dpi: int):
    diff_name = DIFFICULTIES[difficulty_idx]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=False)

    bars_before, bars_after = _draw_subplot(axes[0], EXPERIMENT_DATA["strict"], difficulty_idx)
    _draw_subplot(axes[1], EXPERIMENT_DATA["loose"], difficulty_idx)

    fig.suptitle(f"3D AP Comparison ({diff_name})", fontsize=18, y=1.02)
    fig.legend(
        [bars_before[0], bars_after[0]],
        [METHOD_LABELS["pretrained"], METHOD_LABELS["online_updated"]],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    fig.tight_layout()

    stem = f"{diff_name.lower()}_ap_compare"
    out_png = out_dir / f"{stem}.png"
    out_pdf = out_dir / f"{stem}.pdf"
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out_png}")
    print(f"[OK] saved: {out_pdf}")


def main():
    args = parse_args()
    _validate_data()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for difficulty_idx in range(3):
        _draw_one_difficulty(difficulty_idx, out_dir=out_dir, dpi=int(args.dpi))


if __name__ == "__main__":
    main()
