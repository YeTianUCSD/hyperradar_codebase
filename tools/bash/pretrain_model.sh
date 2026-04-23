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

LOG=/home/code/hyperradar/hyperradar_codebase/output/kitti_models/pointpillar_vod_hd_retrain_hdonly/run500_cls_retrain_hd/retrain_hdonly_from_CNN_checkpoint_epoch18_$(date +%F_%H%M%S).log

nohup python -u retrainHD.py \
  --cfg_file cfgs/kitti_models/pointpillar_vod_hd_retrain_hdonly.yaml \
  --base_ckpt /home/code/hyperradar/hyperradar_codebase/output/kitti_models/pointpillar_vod_hd/run100_cls/ckpt/checkpoint_epoch_18.pth \
  --extra_tag run100_cls_retrain_hd_0319 \
  --batch_size 1 \
  --workers 0 \
  --eval_workers 0 \
  --eval_batch_size 1 \
  --epochs 50 \
  --auto_eval_each_epoch \
  --eval_hd_mode baseline \
  > "$LOG" 2>&1 &

echo "PID=$!"
echo "LOG=$LOG"
