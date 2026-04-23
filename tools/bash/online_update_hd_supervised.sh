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

OUT_ROOT=/home/code/hyperradar/hyperradar_codebase/output/kitti_models/online_hd_supervised
RUN_TAG=online_run_v19_T_20data_by_sence_norandom_memory50_ckpt50
LOG=$OUT_ROOT/$RUN_TAG/nohup_$(date +%F_%H%M%S).log

mkdir -p "$OUT_ROOT/$RUN_TAG"

python -u main_online_hd_supervised.py \
  --cfg_file cfgs/kitti_models/pointpillar_vod_hd_retrain_hdonly.yaml \
  --ckpt /home/code/hyperradar/hyperradar_codebase/output/kitti_models/pointpillar_vod_hd_retrain_hdonly/run500_cls_retrain_hd/ckpt/checkpoint_epoch_50.pth \
  --source_memory /home/code/hyperradar/hyperradar_codebase/output/kitti_models/pointpillar_vod_hd_eval/run500_cls/hd_only/epoch_50/hd_memory/hd_memory_epoch_50.pth \
  --online_cfg cfgs/online/pointpillar_vod_hd_supervised.yaml \
  --output_root "$OUT_ROOT" \
  --extra_tag "$RUN_TAG" \
  --batch_size 8 \
  --workers 2 \
  --eval_batch_size 8 \
  --eval_workers 2 \
  --stream_ratio 1.0 \
  --max_steps -1
