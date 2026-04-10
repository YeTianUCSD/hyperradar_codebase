#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example:
python /home/code/hyperradar/hyperradar_codebase/tools/visualization/draw_online_update_recall_curve.py \
  --out /home/code/hyperradar/hyperradar_codebase/tools/visualization/update_recall_from_list.png \
  --title "Recall vs Update Count"
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output figure path (png/pdf)")
    ap.add_argument("--title", default="Recall vs Update Count", help="plot title")
    args = ap.parse_args()

    # Your provided values (10 updates)
    y = [0.4543, 0.2889, 0.3115, 0.4225, 0.3025, 0.2845, 0.3552, 0.3881, 0.2443, 0.2995]
    x = list(range(1, len(y) + 1))  # 1..10

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker="o", label="recall")
    plt.xlabel("Update Count")
    plt.ylabel("Recall")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    print("[OK] Saved figure to:", out_path)
    print("[Values]", y)


if __name__ == "__main__":
    main()