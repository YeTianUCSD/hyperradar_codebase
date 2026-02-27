#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example:
python /home/code/hyperradar/hyperradar_codebase/tools/visualization/draw_epochAP.py \
  --log /home/code/hyperradar/hyperradar_codebase/output/kitti_models/pointpillar_vod_hd_retrain_hdonly/run500_cls_retrain_hd/retrain_hdonly_from_CNN_checkpoint_epoch17_2026-02-14_120548.log \
  --out /home/code/hyperradar/hyperradar_codebase/output/kitti_models/pointpillar_vod_hd_retrain_hdonly/run500_cls_retrain_hd/retrainHD_hdonlyepoch50_AP.png \
  --epoch_min 1 --epoch_max 50 \
  --verbose
"""

import re
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- patterns ---
EPOCH_PAT = re.compile(r"Performance of EPOCH\s+(\d+)", re.IGNORECASE)

# Match e.g. "3d   AP:4.1988, 7.0321, 5.7293"
AP3_PAT = re.compile(
    r"\b3d\s+AP:\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE
)

# Section headers we care about (strict IoU)
CAR_HDR = re.compile(r"^Car\s+AP_R40@0\.70,\s*0\.70,\s*0\.70\s*:", re.IGNORECASE)
PED_HDR = re.compile(r"^Pedestrian\s+AP_R40@0\.50,\s*0\.50,\s*0\.50\s*:", re.IGNORECASE)
CYC_HDR = re.compile(r"^Cyclist\s+AP_R40@0\.50,\s*0\.50,\s*0\.50\s*:", re.IGNORECASE)


def parse_log_ap(log_path: Path, epoch_min: int, epoch_max: int, verbose: bool) -> Dict[int, Dict[str, Tuple[float, float, float]]]:
    """
    Return:
      epoch -> {
        "car": (easy, mod, hard),
        "ped": (easy, mod, hard),
        "cyc": (easy, mod, hard),
      }
    If an epoch appears multiple times, keep the LAST values.
    """
    text = log_path.read_text(errors="ignore")

    out: Dict[int, Dict[str, Tuple[float, float, float]]] = {}

    cur_epoch: Optional[int] = None
    cur_section: Optional[str] = None  # "car" | "ped" | "cyc" | None

    # We'll keep a working dict for current epoch and overwrite as we find new values
    cur_vals: Dict[str, Tuple[float, float, float]] = {}

    def flush_epoch():
        nonlocal cur_epoch, cur_vals
        if cur_epoch is None:
            return
        if cur_epoch < epoch_min or cur_epoch > epoch_max:
            return
        # store whatever we have (maybe incomplete); plotting will handle missing
        out[cur_epoch] = dict(cur_vals)

    for line in text.splitlines():
        m = EPOCH_PAT.search(line)
        if m:
            # new epoch starts -> flush previous
            flush_epoch()
            cur_epoch = int(m.group(1))
            cur_section = None
            cur_vals = {}
            continue

        if cur_epoch is None:
            continue

        # Only care requested epoch range
        if cur_epoch < epoch_min or cur_epoch > epoch_max:
            continue

        s = line.strip()

        # detect which block we are in
        if CAR_HDR.match(s):
            cur_section = "car"
            continue
        if PED_HDR.match(s):
            cur_section = "ped"
            continue
        if CYC_HDR.match(s):
            cur_section = "cyc"
            continue

        if cur_section is None:
            continue

        m3 = AP3_PAT.search(s)
        if m3:
            easy = float(m3.group(1))
            mod  = float(m3.group(2))
            hard = float(m3.group(3))
            cur_vals[cur_section] = (easy, mod, hard)
            if verbose:
                print(f"[OK] epoch={cur_epoch:>3} section={cur_section}  easy={easy:.4f} mod={mod:.4f} hard={hard:.4f}")
            # keep section active until another header appears
            continue

    # flush last epoch
    flush_epoch()
    return out


def series_from_map(ap_map: Dict[int, Dict[str, Tuple[float, float, float]]],
                    key: str, idx: int,
                    epoch_min: int, epoch_max: int) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for e in range(epoch_min, epoch_max + 1):
        if e in ap_map and key in ap_map[e]:
            xs.append(e)
            ys.append(ap_map[e][key][idx])
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to a single eval log")
    ap.add_argument("--out", required=True, help="output figure path (png/pdf)")
    ap.add_argument("--epoch_min", type=int, default=1)
    ap.add_argument("--epoch_max", type=int, default=20)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"log not found: {log_path}")

    epoch_min, epoch_max = args.epoch_min, args.epoch_max
    ap_map = parse_log_ap(log_path, epoch_min, epoch_max, args.verbose)

    if not ap_map:
        raise RuntimeError("Parsed 0 epochs. Check whether log contains 'Performance of EPOCH k' lines.")

    # Build 9 curves
        # --- Color families (same class = same colormap family) ---
    cmap_car = plt.get_cmap("Blues")
    cmap_ped = plt.get_cmap("Greens")
    cmap_cyc = plt.get_cmap("Oranges")

    # 3 shades for easy/mod/hard
    car_cols = [cmap_car(0.45), cmap_car(0.65), cmap_car(0.85)]
    ped_cols = [cmap_ped(0.45), cmap_ped(0.65), cmap_ped(0.85)]
    cyc_cols = [cmap_cyc(0.45), cmap_cyc(0.65), cmap_cyc(0.85)]

    # line style for difficulty (consistent across classes)
    styles = ["-", "--", ":"]   # Easy, Mod, Hard
    diff_names = ["Easy", "Mod", "Hard"]

    plt.figure(figsize=(11, 6))
    plotted = 0

    for cls_name, key, cols in [
        ("Car", "car", car_cols),
        ("Ped", "ped", ped_cols),
        ("Cyc", "cyc", cyc_cols),
    ]:
        for i in range(3):  # 0/1/2 = Easy/Mod/Hard
            xs, ys = series_from_map(ap_map, key, i, epoch_min, epoch_max)
            if not xs:
                continue
            plt.plot(
                xs, ys,
                marker="o",
                linestyle=styles[i],
                color=cols[i],
                label=f"{cls_name}-{diff_names[i]}",
            )
            plotted += 1

    if plotted == 0:
        raise RuntimeError("No AP points found for the selected sections. Check header patterns / log content.")

    plt.xlabel("Epoch")
    plt.ylabel("AP (3D, R40)")
    plt.title("AP_R40 3D vs Epoch (Car@0.70 / Ped@0.50 / Cyc@0.50)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3)
    plt.tight_layout()


    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200)
    print(f"[OK] Saved figure to: {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
