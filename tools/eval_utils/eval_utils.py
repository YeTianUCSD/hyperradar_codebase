import pickle
import time
import json
import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (
            metric['recall_roi_%s' % str(min_thresh)],
            metric['recall_rcnn_%s' % str(min_thresh)],
            metric['gt_num']
        )


def _jsonable(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _sync_if_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    """
    Detailed timing breakdown:
      - dataloader_wait: time spent waiting for next batch from dataloader iterator
      - to_gpu:          load_data_to_gpu + device transfers
      - forward:         model forward + decode (true GPU time with cuda synchronize)
      - gen_pred:        dataset.generate_prediction_dicts (and optional file write)
      - eval_metric:     dataset.evaluation (KITTI metrics etc.)
      - total:           whole eval_one_epoch wall clock
    """
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {'gt_num': 0}
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric[f'recall_roi_{cur_thresh}'] = 0
        metric[f'recall_rcnn_{cur_thresh}'] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    # ---- infer_time meter (ms) ----
    if getattr(args, 'infer_time', False):
        # Only start measuring after a few warmup iters to avoid first-iter overhead
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)

    # ---- DDP wrapper ----
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % max(num_gpus, 1)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False
        )

    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)

    # -------------------- timing accumulators --------------------
    t_total0 = time.perf_counter()

    t_dataloader_wait = 0.0
    t_to_gpu = 0.0
    t_forward = 0.0
    t_gen_pred = 0.0
    # (evaluation happens after loop)
    t_eval_metric = 0.0

    # for dataloader wait: measure time gap between end of last iter and start of next iter
    last_iter_end = time.perf_counter()

    # optional: print per-iter average every N iters
    log_every = int(getattr(args, "time_log_every", 10))  # you can add arg if you want; default 10

    # -------------------- main loop --------------------
    for i, batch_dict in enumerate(dataloader):
        # dataloader wait
        now = time.perf_counter()
        t_dataloader_wait += (now - last_iter_end)

        # to GPU
        t0 = time.perf_counter()
        load_data_to_gpu(batch_dict)
        _sync_if_cuda()
        t_to_gpu += (time.perf_counter() - t0)

        # forward (true GPU time)
        if getattr(args, 'infer_time', False) and i >= start_iter:
            # measure inference latency (ms) after warmup
            t0 = time.perf_counter()
            with torch.no_grad():
                pred_dicts, ret_dict = model(batch_dict)
            _sync_if_cuda()
            infer_ms = (time.perf_counter() - t0) * 1000.0
            infer_time_meter.update(infer_ms)
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                pred_dicts, ret_dict = model(batch_dict)
            _sync_if_cuda()

        t_forward += (time.perf_counter() - t0)

        # disp
        disp_dict = {}
        if getattr(args, 'infer_time', False):
            # show current/avg in ms
            disp_dict['infer_time_ms'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})' if i >= start_iter else 'warmup'

        # update recall metrics
        statistics_info(cfg, ret_dict, metric, disp_dict)

        # generate prediction dicts (CPU-heavy + optional IO)
        t0 = time.perf_counter()
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        t_gen_pred += (time.perf_counter() - t0)

        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

            # periodic detailed timing
            if log_every > 0 and (i + 1) % log_every == 0:
                it = i + 1
                logger.info(
                    f"[TIME-ITER] it={it}/{len(dataloader)} | "
                    f"avg_wait={t_dataloader_wait/it:.4f}s | "
                    f"avg_to_gpu={t_to_gpu/it:.4f}s | "
                    f"avg_forward={t_forward/it:.4f}s | "
                    f"avg_gen_pred={t_gen_pred/it:.4f}s | "
                    f"(bs={getattr(args, 'batch_size', 'NA')})"
                )

        last_iter_end = time.perf_counter()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    # -------------------- dist merge --------------------
    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    # -------------------- summary + recall --------------------
    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)

    # total wall time for generating labels (before evaluation)
    t_after_loop = time.perf_counter()
    t_gen_total = t_after_loop - t_total0
    sec_per_example = t_gen_total / max(1, len(dataloader.dataset))
    logger.info('Generate label finished(sec_per_example: %.6f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        # merge metric across ranks
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric[f'recall_roi_{cur_thresh}'] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric[f'recall_rcnn_{cur_thresh}'] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict[f'recall/roi_{cur_thresh}'] = cur_roi_recall
        ret_dict[f'recall/rcnn_{cur_thresh}'] = cur_rcnn_recall

    total_pred_objects = sum(len(anno['name']) for anno in det_annos)
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    # save pkl
    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    if getattr(args, 'fast_recall_only', False):
        logger.info('*************** FAST_RECALL_ONLY: skip dataset.evaluation() *****************')
        quick_path = result_dir / 'metrics_recall_only.json'
        quick_obj = {
            'recall': {
                f'roi_{k}': float(ret_dict[f'recall/roi_{k}'])
                for k in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST
            },
            'recall_rcnn': {
                f'rcnn_{k}': float(ret_dict[f'recall/rcnn_{k}'])
                for k in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST
            },
            'avg_pred_objects': float(total_pred_objects / max(1, len(det_annos))),
            'num_samples': int(len(det_annos))
        }
        with open(quick_path, 'w') as f:
            json.dump(quick_obj, f, indent=2)
        logger.info('Saved recall-only metrics to %s' % str(quick_path))

        t_total1 = time.perf_counter()
        total_s = t_total1 - t_total0
        n_iter = max(1, len(dataloader))
        avg_wait = t_dataloader_wait / n_iter
        avg_to = t_to_gpu / n_iter
        avg_fw = t_forward / n_iter
        avg_gp = t_gen_pred / n_iter
        logger.info(
            f"[TIME-SUM] wait={t_dataloader_wait:.2f}s (avg {avg_wait:.4f}s/it) | "
            f"to_gpu={t_to_gpu:.2f}s (avg {avg_to:.4f}s/it) | "
            f"forward={t_forward:.2f}s (avg {avg_fw:.4f}s/it) | "
            f"gen_pred={t_gen_pred:.2f}s (avg {avg_gp:.4f}s/it) | "
            f"eval_metric=SKIPPED | "
            f"TOTAL={total_s:.2f}s ({total_s/60.0:.2f}min)"
        )
        logger.info('Result is saved to %s' % result_dir)
        logger.info('****************Evaluation done.*****************')
        return ret_dict

    # -------------------- evaluation metric timing --------------------
    logger.info('*************** Running dataset.evaluation() *****************')
    logger.info('EVAL_METRIC=%s' % str(cfg.MODEL.POST_PROCESSING.get('EVAL_METRIC', None)))
    logger.info('output_path=%s' % str(final_output_dir))
    logger.info('Entering dataset.evaluation() ...')

    t0 = time.perf_counter()
    eval_ret = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )
    t_eval_metric = time.perf_counter() - t0
    logger.info('dataset.evaluation() finished in %.2f sec' % t_eval_metric)

    # Support return formats
    result_str, result_dict = '', {}
    if isinstance(eval_ret, tuple) and len(eval_ret) == 2:
        result_str, result_dict = eval_ret
    elif isinstance(eval_ret, dict):
        result_dict = eval_ret
        result_str = str(eval_ret)
    elif isinstance(eval_ret, str):
        result_str = eval_ret
    else:
        result_str = f'Unexpected evaluation return type: {type(eval_ret)}'
        result_dict = {}

    logger.info('*************** Evaluation Summary (raw) *****************')
    if result_str is None:
        result_str = ''
    if len(result_str.strip()) > 0:
        logger.info('\n' + result_str)
    else:
        logger.info('[EMPTY result_str] dataset.evaluation returned empty string')

    if isinstance(result_dict, dict):
        logger.info('Evaluation result_dict keys (%d): %s' % (len(result_dict), sorted(list(result_dict.keys()))))
    else:
        logger.info('Evaluation result_dict is not a dict: %s' % str(type(result_dict)))

    # save metrics json
    metrics_path = result_dir / 'metrics_eval.json'
    with open(metrics_path, 'w') as f:
        if isinstance(result_dict, dict):
            json.dump({k: _jsonable(v) for k, v in result_dict.items()}, f, indent=2)
        else:
            json.dump({"result_str": result_str}, f, indent=2)
    logger.info('Saved evaluation metrics to %s' % str(metrics_path))

    # update ret_dict for tb
    if isinstance(result_dict, dict):
        ret_dict.update({
            f"eval/{k}": float(v)
            for k, v in result_dict.items()
            if isinstance(v, (int, float, np.floating, np.integer))
        })

    # -------------------- final time summary --------------------
    t_total1 = time.perf_counter()
    total_s = t_total1 - t_total0

    n_iter = max(1, len(dataloader))
    # per-iter averages
    avg_wait = t_dataloader_wait / n_iter
    avg_to = t_to_gpu / n_iter
    avg_fw = t_forward / n_iter
    avg_gp = t_gen_pred / n_iter

    logger.info(
        f"[TIME-SUM] wait={t_dataloader_wait:.2f}s (avg {avg_wait:.4f}s/it) | "
        f"to_gpu={t_to_gpu:.2f}s (avg {avg_to:.4f}s/it) | "
        f"forward={t_forward:.2f}s (avg {avg_fw:.4f}s/it) | "
        f"gen_pred={t_gen_pred:.2f}s (avg {avg_gp:.4f}s/it) | "
        f"eval_metric={t_eval_metric:.2f}s | "
        f"TOTAL={total_s:.2f}s ({total_s/60.0:.2f}min)"
    )

    if getattr(args, 'infer_time', False):
        logger.info(
            f"[TIME-INFER] infer_time_ms: last={infer_time_meter.val:.2f} | avg={infer_time_meter.avg:.2f} "
            f"(start_iter={int(len(dataloader)*0.1)})"
        )

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
