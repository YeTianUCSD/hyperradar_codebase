#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   cd tools
#   bash sweep_hd_calib.sh \
#     --cfg_file cfgs/kitti_models/pointpillar_vod_hd.yaml \
#     --ckpt /abs/path/checkpoint_epoch_3.pth \
#     --hd_memory /abs/path/hd_memory_epoch_3.pth \
#     --extra_tag pointpillar_vod_hd_eval/sweep_epoch3 \
#     --workers 1 --batch_size 16

CFG_FILE=""
CKPT=""
HD_MEMORY=""
EXTRA_TAG="pointpillar_vod_hd_eval/sweep"
BATCH_SIZE=16
WORKERS=1
HD_MODE="hd_only"
EVAL_TAG="default"
TEMPS="1.0 1.2"
THRS="0.60 0.65 0.68"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cfg_file) CFG_FILE="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --hd_memory) HD_MEMORY="$2"; shift 2 ;;
    --extra_tag) EXTRA_TAG="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --hd_mode) HD_MODE="$2"; shift 2 ;;
    --eval_tag) EVAL_TAG="$2"; shift 2 ;;
    --temps) TEMPS="$2"; shift 2 ;;
    --score_thrs) THRS="$2"; shift 2 ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$CFG_FILE" || -z "$CKPT" || -z "$HD_MEMORY" ]]; then
  echo "Missing required args: --cfg_file --ckpt --hd_memory" >&2
  exit 1
fi

ROOT_OUT="$(dirname "$CKPT")/../.."
RUN_LOG_DIR="${ROOT_OUT}/sweep_logs"
mkdir -p "$RUN_LOG_DIR"
CSV="${RUN_LOG_DIR}/sweep_hd_calib_$(date +%Y%m%d-%H%M%S).csv"
echo "temperature,score_thresh,recall_rcnn_0.3,recall_rcnn_0.5,recall_rcnn_0.7,avg_pred_objects,log_file" > "$CSV"

echo "[SWEEP] csv: $CSV"

extract_last_value() {
  local pattern="$1"
  local file="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -o "$pattern" "$file" | tail -n1 | awk '{print $NF}'
  else
    grep -Eo "$pattern" "$file" | tail -n1 | awk '{print $NF}'
  fi
}

for T in $TEMPS; do
  for S in $THRS; do
    TAG="${EXTRA_TAG}/t${T}_s${S}"
    LOG="${RUN_LOG_DIR}/run_t${T}_s${S}_$(date +%Y%m%d-%H%M%S).log"
    echo "[SWEEP] T=${T} S=${S} -> ${LOG}"

    python -u test_hd.py \
      --cfg_file "$CFG_FILE" \
      --ckpt "$CKPT" \
      --hd_memory "$HD_MEMORY" \
      --hd_mode "$HD_MODE" \
      --hd_temperature "$T" \
      --batch_size "$BATCH_SIZE" \
      --workers "$WORKERS" \
      --extra_tag "$TAG" \
      --eval_tag "$EVAL_TAG" \
      --fast_recall_only \
      --set MODEL.POST_PROCESSING.SCORE_THRESH "$S" \
      > "$LOG" 2>&1

    R03=$(extract_last_value "recall_rcnn_0.3: [0-9.]+" "$LOG")
    R05=$(extract_last_value "recall_rcnn_0.5: [0-9.]+" "$LOG")
    R07=$(extract_last_value "recall_rcnn_0.7: [0-9.]+" "$LOG")
    AVG=$(extract_last_value "Average predicted number of objects\\([0-9]+ samples\\): [0-9.]+" "$LOG")

    R03=${R03:-NA}
    R05=${R05:-NA}
    R07=${R07:-NA}
    AVG=${AVG:-NA}

    echo "${T},${S},${R03},${R05},${R07},${AVG},${LOG}" >> "$CSV"
  done
done

echo "[SWEEP] done. csv: $CSV"
