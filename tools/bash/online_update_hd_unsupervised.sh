#!/usr/bin/env bash

set -euo pipefail

cd /home/code/hyperradar/hyperradar_codebase/tools

export NUMBA_DISABLE_JIT=1
export NUMBA_CACHE_DIR=/tmp/numba_cache
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

OUT_ROOT=/home/code/hyperradar/hyperradar_codebase/output/kitti_models/online_hd_unsupervised
RUN_TAG=online_run_v14_T_hd_unsup_retraincfg_ckpt50_memory50
LOG=$OUT_ROOT/$RUN_TAG/run_$(date +%F_%H%M%S).log

mkdir -p "$OUT_ROOT/$RUN_TAG"

python -u main_online_hd.py \
  --cfg_file cfgs/kitti_models/pointpillar_vod_hd_retrain_hdonly.yaml \
  --ckpt /home/code/hyperradar/hyperradar_codebase/output/kitti_models/pointpillar_vod_hd_retrain_hdonly/run500_cls_retrain_hd/ckpt/checkpoint_epoch_50.pth \
  --source_memory /home/code/hyperradar/hyperradar_codebase/output/kitti_models/pointpillar_vod_hd_eval/run500_cls/hd_only/epoch_50/hd_memory/hd_memory_epoch_50.pth \
  --online_cfg cfgs/online/pointpillar_vod_hd_online.yaml \
  --output_root "$OUT_ROOT" \
  --extra_tag "$RUN_TAG" \
  --stream_split test \
  --val_split val \
  --batch_size 1 \
  --workers 0 \
  --max_steps -1 \
  --guard_metric recall/rcnn_0.3 \
  --guard_fast_recall_only 1 \
  --final_full_eval 1 \
  2>&1 | tee "$LOG"

