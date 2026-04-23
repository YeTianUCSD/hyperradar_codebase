import _init_path
import argparse
import datetime
from pathlib import Path
from typing import Dict

import torch
import yaml

from modules.online_all_runner import OnlineAllRunner
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.models import build_network
from pcdet.utils import common_utils

from main_online_all_supervised import (
    _build_eval_fn,
    _build_loader_for_split,
    _build_optimizer,
    _build_subset_loader,
    _configure_trainable_scope,
    _load_memory_payload_into_hd_core,
    _load_runner_best_state_into_model,
    _make_stream_subset_indices,
    _safe_len,
)


def parse_config():
    parser = argparse.ArgumentParser(description="HyperRadar Unsupervised Online ALL-cls+HD Adaptation")

    parser.add_argument("--cfg_file", type=str, required=True, help="Detector config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Detector checkpoint")
    parser.add_argument("--source_memory", type=str, default=None,
                        help="Optional HD memory/prototype file used to initialize the online run")
    parser.add_argument("--online_cfg", type=str, required=True,
                        help="Path to unsupervised ALL online yaml config. Expected root key: ALL_ONLINE")

    parser.add_argument("--extra_tag", type=str, default="online_all", help="Output tag")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Optional output root dir. If set, outputs go to <output_root>/<extra_tag>")

    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for stream loader")
    parser.add_argument("--workers", type=int, default=1, help="Workers for stream loader")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Batch size for eval loader")
    parser.add_argument("--eval_workers", type=int, default=None, help="Workers for eval loader")

    parser.add_argument("--max_steps", type=int, default=-1, help="Max stream steps (-1 for all)")
    parser.add_argument("--save_to_file", action="store_true", default=False,
                        help="Save eval predictions to files")

    # Split / subset overrides
    parser.add_argument("--stream_split", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--stream_ratio", type=float, default=None)
    parser.add_argument("--max_stream_samples", type=int, default=None)
    parser.add_argument("--use_stream_prefix", type=int, choices=[0, 1], default=None)
    parser.add_argument("--stream_seed", type=int, default=None)
    parser.add_argument("--stream_info_paths", type=str, nargs='+', default=None)
    parser.add_argument("--eval_info_paths", type=str, nargs='+', default=None)

    # Trainable scope overrides
    parser.add_argument("--train_conv_cls_pre", type=int, choices=[0, 1], default=None)
    parser.add_argument("--train_conv_cls_out", type=int, choices=[0, 1], default=None)
    parser.add_argument("--train_hd_embedder", type=int, choices=[0, 1], default=None)
    parser.add_argument("--update_hd_memory", type=int, choices=[0, 1], default=None)
    parser.add_argument("--freeze_vfe", type=int, choices=[0, 1], default=None)
    parser.add_argument("--freeze_map_to_bev", type=int, choices=[0, 1], default=None)
    parser.add_argument("--freeze_backbone_2d", type=int, choices=[0, 1], default=None)
    parser.add_argument("--freeze_box_head", type=int, choices=[0, 1], default=None)
    parser.add_argument("--freeze_dir_head", type=int, choices=[0, 1], default=None)

    # Pseudo labels / loss / optimizer overrides
    parser.add_argument("--pseudo_logits_source", type=str, choices=["origin", "hd"], default=None)
    parser.add_argument("--tau_prob", type=float, default=None)
    parser.add_argument("--tau_margin", type=float, default=None)
    parser.add_argument("--select_top_ratio", type=float, default=None)
    parser.add_argument("--select_min_k", type=int, default=None)
    parser.add_argument("--select_max_k", type=int, default=None)
    parser.add_argument("--use_teacher", type=int, choices=[0, 1], default=None)
    parser.add_argument("--teacher_momentum", type=float, default=None)
    parser.add_argument("--use_hd_consistency", type=int, choices=[0, 1], default=None)
    parser.add_argument("--loss_hd_weight", type=float, default=None)
    parser.add_argument("--loss_cnn_weight", type=float, default=None)

    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"], default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)

    # Update / HD memory overrides
    parser.add_argument("--update_every_n_steps", type=int, default=None)
    parser.add_argument("--min_selected_anchors", type=int, default=None)
    parser.add_argument("--min_confpass_anchors", type=int, default=None)
    parser.add_argument("--feature_source", type=str, choices=["cls", "bev"], default=None)
    parser.add_argument("--update_mode", type=str, choices=["train", "retrain", "both"], default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--normalize_every_updates", type=int, default=None)
    parser.add_argument("--source_pullback_lambda", type=float, default=None)

    parser.add_argument("--max_per_class_per_step", type=int, default=None)
    parser.add_argument("--min_per_class_for_update", type=int, default=None)
    parser.add_argument("--class_balance_enable", type=int, choices=[0, 1], default=None)
    parser.add_argument("--class_balance_min_classes_to_update", type=int, default=None)
    parser.add_argument("--class_balance_max_pending_steps", type=int, default=None)

    # Eval / save overrides
    parser.add_argument("--eval_every_updates", type=int, default=None)
    parser.add_argument("--metric_key", type=str, default=None)
    parser.add_argument("--fast_recall_only", type=int, choices=[0, 1], default=None)
    parser.add_argument("--final_full_eval", type=int, choices=[0, 1], default=None)
    parser.add_argument("--guard_max_drop", type=float, default=None)
    parser.add_argument("--guard_use_best", type=int, choices=[0, 1], default=None)
    parser.add_argument("--save_every_updates", type=int, default=None)
    parser.add_argument("--log_interval_steps", type=int, default=None)
    parser.add_argument("--save_best_model", type=int, choices=[0, 1], default=None)
    parser.add_argument("--save_last_model", type=int, choices=[0, 1], default=None)
    parser.add_argument("--zero_update_tau_prob_step", type=float, default=None)
    parser.add_argument("--zero_update_tau_margin_step", type=float, default=None)
    parser.add_argument("--experiment_note", type=str, default=None)

    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra detector config keys if needed')

    args = parser.parse_args()

    if not Path(args.cfg_file).is_file():
        raise FileNotFoundError(f'cfg_file not found: {args.cfg_file}')
    if not Path(args.ckpt).is_file():
        raise FileNotFoundError(f'ckpt not found: {args.ckpt}')
    if args.source_memory is not None and (not Path(args.source_memory).is_file()):
        raise FileNotFoundError(f'source_memory not found: {args.source_memory}')
    if not Path(args.online_cfg).is_file():
        raise FileNotFoundError(f'online_cfg not found: {args.online_cfg}')

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def _build_output_and_logger(args):
    if args.output_root is not None:
        if str(args.output_root).strip() == "":
            raise RuntimeError("output_root must be a non-empty path")
        output_dir = Path(args.output_root) / args.extra_tag
    else:
        output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('online_all_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start unsupervised online ALL-cls+HD logging**********************')
    logger.info(f'cfg_file={args.cfg_file}')
    logger.info(f'ckpt={args.ckpt}')
    logger.info(f'source_memory={args.source_memory}')
    logger.info(f'online_cfg={args.online_cfg}')
    logger.info(f'torch.__version__={torch.__version__}')
    logger.info(f'cuda_available={torch.cuda.is_available()}')
    if torch.cuda.is_available():
        logger.info(f'cuda_device_count={torch.cuda.device_count()}')
        logger.info(f'cuda_current_device={torch.cuda.current_device()}')
        logger.info(f'cuda_device_name={torch.cuda.get_device_name(torch.cuda.current_device())}')
    for key, val in vars(args).items():
        logger.info('{:28} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    return output_dir, logger


def _load_all_online_cfg_file(path: str) -> Dict:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise RuntimeError(f'Invalid online cfg file: expected dict root, got {type(raw)}')
    online_cfg = raw.get('ALL_ONLINE', raw)
    if not isinstance(online_cfg, dict):
        raise RuntimeError(f'Invalid ALL_ONLINE section type: {type(online_cfg)}')
    return dict(online_cfg)


def _build_online_cfg_dict(args) -> Dict:
    mapping = {
        'stream_split': args.stream_split,
        'eval_split': args.eval_split,
        'stream_ratio': args.stream_ratio,
        'max_stream_samples': args.max_stream_samples,
        'stream_info_paths': args.stream_info_paths,
        'eval_info_paths': args.eval_info_paths,
        'pseudo_logits_source': args.pseudo_logits_source,
        'tau_prob': args.tau_prob,
        'tau_margin': args.tau_margin,
        'select_top_ratio': args.select_top_ratio,
        'select_min_k': args.select_min_k,
        'select_max_k': args.select_max_k,
        'teacher_momentum': args.teacher_momentum,
        'loss_hd_weight': args.loss_hd_weight,
        'loss_cnn_weight': args.loss_cnn_weight,
        'optimizer': args.optimizer,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'grad_clip': args.grad_clip,
        'update_every_n_steps': args.update_every_n_steps,
        'min_selected_anchors': args.min_selected_anchors,
        'min_confpass_anchors': args.min_confpass_anchors,
        'feature_source': args.feature_source,
        'update_mode': args.update_mode,
        'alpha': args.alpha,
        'normalize_every_updates': args.normalize_every_updates,
        'source_pullback_lambda': args.source_pullback_lambda,
        'max_per_class_per_step': args.max_per_class_per_step,
        'min_per_class_for_update': args.min_per_class_for_update,
        'class_balance_min_classes_to_update': args.class_balance_min_classes_to_update,
        'class_balance_max_pending_steps': args.class_balance_max_pending_steps,
        'eval_every_updates': args.eval_every_updates,
        'metric_key': args.metric_key,
        'guard_max_drop': args.guard_max_drop,
        'save_every_updates': args.save_every_updates,
        'log_interval_steps': args.log_interval_steps,
        'zero_update_tau_prob_step': args.zero_update_tau_prob_step,
        'zero_update_tau_margin_step': args.zero_update_tau_margin_step,
        'experiment_note': args.experiment_note,
    }

    bool_args = [
        'use_stream_prefix',
        'train_conv_cls_pre',
        'train_conv_cls_out',
        'train_hd_embedder',
        'update_hd_memory',
        'freeze_vfe',
        'freeze_map_to_bev',
        'freeze_backbone_2d',
        'freeze_box_head',
        'freeze_dir_head',
        'use_teacher',
        'use_hd_consistency',
        'class_balance_enable',
        'fast_recall_only',
        'final_full_eval',
        'guard_use_best',
        'save_best_model',
        'save_last_model',
    ]
    for name in bool_args:
        val = getattr(args, name)
        if val is not None:
            mapping[name] = bool(val)

    if args.stream_seed is not None:
        mapping['stream_seed'] = int(args.stream_seed)

    return {k: v for k, v in mapping.items() if v is not None}


def _save_run_artifacts(output_dir: Path, args, online_cfg: Dict, logger):
    meta = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cfg_file': str(args.cfg_file),
        'ckpt': str(args.ckpt),
        'source_memory': None if args.source_memory is None else str(args.source_memory),
        'online_cfg': str(args.online_cfg),
        'extra_tag': str(args.extra_tag),
        'batch_size': args.batch_size,
        'workers': args.workers,
        'eval_batch_size': args.eval_batch_size,
        'eval_workers': args.eval_workers,
        'max_steps': args.max_steps,
        'save_to_file': bool(args.save_to_file),
        'stream_info_paths': args.stream_info_paths,
        'eval_info_paths': args.eval_info_paths,
    }
    with open(output_dir / 'run_meta.yaml', 'w') as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    with open(output_dir / 'online_cfg_resolved.yaml', 'w') as f:
        yaml.safe_dump({'ALL_ONLINE': online_cfg}, f, sort_keys=False)
    logger.info(f'[ARTIFACT] saved run meta and resolved online cfg to {str(output_dir)}')


def main():
    args, _cfg = parse_config()
    output_dir, logger = _build_output_and_logger(args)

    online_cfg = _load_all_online_cfg_file(args.online_cfg)
    logger.info(f'[ONLINE_CFG] loaded from file: {args.online_cfg}')
    logger.info(f'[ONLINE_CFG] file keys: {sorted(list(online_cfg.keys()))}')
    online_cfg.update(_build_online_cfg_dict(args))
    logger.info(f'[ONLINE_CFG] final keys: {sorted(list(online_cfg.keys()))}')
    logger.info(f'[ONLINE_CFG] final values:\n{yaml.safe_dump(online_cfg, sort_keys=True)}')
    _save_run_artifacts(output_dir, args, online_cfg, logger)

    stream_split = str(online_cfg.get('stream_split', 'test'))
    eval_split = str(online_cfg.get('eval_split', 'val'))
    stream_info_paths = online_cfg.get('stream_info_paths', None)
    eval_info_paths = online_cfg.get('eval_info_paths', None)

    stream_bs = args.batch_size if args.batch_size is not None else 1
    eval_bs = args.eval_batch_size if args.eval_batch_size is not None else stream_bs
    stream_workers = int(args.workers)
    eval_workers = int(args.eval_workers) if args.eval_workers is not None else stream_workers

    logger.info(f'[DATA] stream split={stream_split}')
    if stream_info_paths is not None:
        logger.info(f'[DATA] stream info paths={stream_info_paths}')
    stream_set_full, _stream_loader_full = _build_loader_for_split(
        stream_split, stream_bs, stream_workers, logger, info_paths=stream_info_paths
    )
    logger.info(f'[DATA] full stream dataset size={_safe_len(stream_set_full)} loader_len={_safe_len(_stream_loader_full)}')
    if _safe_len(stream_set_full) <= 0:
        raise RuntimeError(f'stream split "{stream_split}" produced an empty dataset')

    stream_indices = _make_stream_subset_indices(stream_set_full, online_cfg, logger)
    stream_set, stream_loader = _build_subset_loader(
        stream_set_full,
        stream_indices,
        batch_size=stream_bs,
        workers=stream_workers,
        seed=int(online_cfg.get('stream_seed', 0)),
    )
    logger.info(f'[DATA] effective stream subset size={_safe_len(stream_set)} loader_len={_safe_len(stream_loader)}')

    logger.info(f'[DATA] eval split={eval_split}')
    if eval_info_paths is not None:
        logger.info(f'[DATA] eval info paths={eval_info_paths}')
    eval_set, eval_loader = _build_loader_for_split(
        eval_split, eval_bs, eval_workers, logger, info_paths=eval_info_paths
    )
    logger.info(f'[DATA] eval dataset size={_safe_len(eval_set)} loader_len={_safe_len(eval_loader)}')
    if _safe_len(eval_set) <= 0:
        raise RuntimeError(f'eval split "{eval_split}" produced an empty dataset')

    logger.info('[MODEL] building detector...')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=stream_set_full)
    logger.info('[MODEL] loading checkpoint...')
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    logger.info('[MODEL] model moved to CUDA and set to eval')

    dense_head = getattr(model, 'dense_head', None)
    hd_core = getattr(dense_head, 'hd_core', None) if dense_head is not None else None
    if hd_core is None:
        raise RuntimeError('Cannot find dense_head.hd_core in model.')
    if hasattr(dense_head, 'hd_mode'):
        logger.info(f'[MODEL] dense_head.hd_mode={dense_head.hd_mode}')

    if args.source_memory is not None:
        logger.info(f'[MEM] loading source memory: {args.source_memory}')
        _load_memory_payload_into_hd_core(hd_core, args.source_memory, logger)

    _configure_trainable_scope(model, online_cfg, logger)
    optimizer = _build_optimizer(model, online_cfg, logger)

    eval_fn, run_eval = _build_eval_fn(args, logger, output_dir, online_cfg, eval_bs)

    runner = OnlineAllRunner(
        model=model,
        optimizer=optimizer,
        stream_loader=stream_loader,
        logger=logger,
        cfg=online_cfg,
        eval_loader=eval_loader,
        eval_fn=eval_fn,
        output_dir=str(output_dir / 'online_state'),
        state_save_prefix='online_all',
    )

    logger.info('[ONLINE] unsupervised ALL-cls+HD runner start')
    runner.run(max_steps=args.max_steps)
    logger.info('[ONLINE] unsupervised ALL-cls+HD runner finished')

    if bool(online_cfg.get('final_full_eval', True)) and _safe_len(eval_loader) > 0:
        _load_runner_best_state_into_model(runner, logger)
        logger.info('[EVAL] start final full official evaluation once')
        tb_dict = run_eval(model, eval_loader, fast_recall_only=False, eval_name='final_full')
        if isinstance(tb_dict, dict):
            metric_key = str(online_cfg.get('metric_key', 'recall/rcnn_0.3'))
            if metric_key in tb_dict:
                logger.info(f'[EVAL][FINAL] {metric_key}={float(tb_dict[metric_key]):.6f}')
            for k in ('recall/rcnn_0.3', 'recall/rcnn_0.5', 'mAP', 'map'):
                if k in tb_dict:
                    logger.info(f'[EVAL][FINAL] {k}={float(tb_dict[k]):.6f}')
        logger.info('[EVAL] final full official evaluation finished')
    elif bool(online_cfg.get('final_full_eval', True)):
        logger.warning('[EVAL] skip final full eval because eval loader is empty')

    logger.info('**********************End unsupervised online ALL-cls+HD**********************')


if __name__ == '__main__':
    main()
