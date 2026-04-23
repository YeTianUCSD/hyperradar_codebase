import _init_path
import argparse
import copy
import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import torch
import yaml

from eval_utils import eval_utils
from modules.online_cnn_runner import OnlineCNNRunner
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def _safe_len(x):
    try:
        return len(x)
    except Exception:
        return -1


def parse_config():
    parser = argparse.ArgumentParser(description="HyperRadar Online CNN Classifier Adaptation")

    parser.add_argument("--cfg_file", type=str, required=True, help="Detector config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Detector checkpoint")
    parser.add_argument("--online_cfg", type=str, required=True,
                        help="Path to online yaml config. Expected root key: CNN_ONLINE")

    parser.add_argument("--extra_tag", type=str, default="online_cnn", help="Output tag")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Optional output root dir. If set, outputs go to <output_root>/<extra_tag>")

    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--workers", type=int, default=1, help="Dataloader workers")

    parser.add_argument("--stream_split", type=str, default=None,
                        help="Split name used for stream (e.g., train/val/test). If None, keep cfg default test split")
    parser.add_argument("--val_split", type=str, default=None,
                        help="Split name used for guard evaluation (e.g., val/test). If None, disable guard eval")

    parser.add_argument("--max_steps", type=int, default=-1, help="Max stream steps (-1 for all)")
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use mixed precision for forward")

    parser.add_argument("--guard_metric", type=str, default="recall/rcnn_0.3",
                        help="Metric key used from eval dict for guard")
    parser.add_argument("--guard_fast_recall_only", type=int, choices=[0, 1], default=1,
                        help="Use recall-only eval during online guard checks")
    parser.add_argument("--final_full_eval", type=int, choices=[0, 1], default=1,
                        help="Run one full official evaluation after online updates finish")
    parser.add_argument("--save_to_file", action="store_true", default=False,
                        help="Save eval predictions to files")

    # Online config overrides (map to OnlineCNNConfig fields)
    parser.add_argument("--tau_prob", type=float, default=None)
    parser.add_argument("--tau_margin", type=float, default=None)
    parser.add_argument("--select_top_ratio", type=float, default=None)
    parser.add_argument("--select_min_k", type=int, default=None)
    parser.add_argument("--select_max_k", type=int, default=None)

    parser.add_argument("--use_teacher", type=int, choices=[0, 1], default=None)
    parser.add_argument("--teacher_momentum", type=float, default=None)

    parser.add_argument("--update_every_n_steps", type=int, default=None)
    parser.add_argument("--min_selected_anchors", type=int, default=None)
    parser.add_argument("--min_confpass_anchors", type=int, default=None)

    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"], default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)

    parser.add_argument("--max_per_class_per_step", type=int, default=None)
    parser.add_argument("--min_per_class_for_update", type=int, default=None)
    parser.add_argument("--class_balance_enable", type=int, choices=[0, 1], default=None)
    parser.add_argument("--class_balance_min_classes_to_update", type=int, default=None)
    parser.add_argument("--class_balance_max_pending_steps", type=int, default=None)

    parser.add_argument("--eval_every_updates", type=int, default=None)
    parser.add_argument("--guard_max_drop", type=float, default=None)
    parser.add_argument("--guard_use_best", type=int, choices=[0, 1], default=None)

    parser.add_argument("--save_every_updates", type=int, default=None)
    parser.add_argument("--log_interval_steps", type=int, default=None)
    parser.add_argument("--pseudo_logits_source", type=str, choices=["final", "origin", "hd"], default=None)

    parser.add_argument("--zero_update_tau_prob_step", type=float, default=None)
    parser.add_argument("--zero_update_tau_margin_step", type=float, default=None)

    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra detector config keys if needed')

    args = parser.parse_args()

    if not Path(args.cfg_file).is_file():
        raise FileNotFoundError(f'cfg_file not found: {args.cfg_file}')
    if not Path(args.ckpt).is_file():
        raise FileNotFoundError(f'ckpt not found: {args.ckpt}')
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

    log_file = output_dir / ('online_cnn_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start online CNN logging**********************')
    logger.info(f'cfg_file={args.cfg_file}')
    logger.info(f'ckpt={args.ckpt}')
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


def _override_split(dataset_cfg, split_name: Optional[str]):
    cfg_local = copy.deepcopy(dataset_cfg)
    if split_name is None:
        return cfg_local
    cfg_local.DATA_SPLIT['test'] = split_name
    split_to_info = {
        'train': ['kitti_infos_train.pkl'],
        'val': ['kitti_infos_val.pkl'],
        'test': ['kitti_infos_test.pkl'],
    }
    if hasattr(cfg_local, 'INFO_PATH') and split_name in split_to_info:
        cfg_local.INFO_PATH['test'] = split_to_info[split_name]
    return cfg_local


def _load_online_cfg_file(path: str) -> Dict:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise RuntimeError(f'Invalid online cfg file: expected dict root, got {type(raw)}')
    online_cfg = raw.get('CNN_ONLINE', raw)
    if not isinstance(online_cfg, dict):
        raise RuntimeError(f'Invalid CNN_ONLINE section type: {type(online_cfg)}')
    return dict(online_cfg)


def _build_online_cfg_dict(args) -> Dict:
    keys = [
        'tau_prob', 'tau_margin', 'select_top_ratio', 'select_min_k', 'select_max_k',
        'teacher_momentum', 'update_every_n_steps', 'min_selected_anchors', 'min_confpass_anchors',
        'grad_clip', 'max_per_class_per_step', 'min_per_class_for_update',
        'class_balance_min_classes_to_update', 'class_balance_max_pending_steps',
        'eval_every_updates', 'guard_max_drop', 'save_every_updates', 'log_interval_steps',
        'pseudo_logits_source', 'zero_update_tau_prob_step', 'zero_update_tau_margin_step',
    ]
    out = {}
    for k in keys:
        v = getattr(args, k)
        if v is not None:
            out[k] = v

    if args.use_teacher is not None:
        out['use_teacher'] = bool(args.use_teacher)
    if args.class_balance_enable is not None:
        out['class_balance_enable'] = bool(args.class_balance_enable)
    if args.guard_use_best is not None:
        out['guard_use_best'] = bool(args.guard_use_best)

    # Optimizer fields are consumed by main_online_cnn.py, but keep them in the
    # resolved config for reproducibility.
    for k in ('optimizer', 'lr', 'weight_decay', 'momentum'):
        v = getattr(args, k)
        if v is not None:
            out[k] = v

    # main_online_cnn.py uses args.guard_metric for eval callback; mirror it into
    # runner config so snapshots and metrics use the same key.
    if args.guard_metric is not None:
        out['metric_key'] = args.guard_metric

    return out


def _save_run_artifacts(output_dir: Path, args, online_cfg: Dict, logger):
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_online_cfg_path = output_dir / 'online_cfg_resolved.yaml'
    with open(resolved_online_cfg_path, 'w') as f:
        yaml.safe_dump({'CNN_ONLINE': online_cfg}, f, sort_keys=True)
    logger.info(f'[ARTIFACT] saved resolved online cfg: {resolved_online_cfg_path}')

    run_meta = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cfg_file': args.cfg_file,
        'ckpt': args.ckpt,
        'online_cfg_arg': args.online_cfg,
        'stream_split': args.stream_split,
        'val_split': args.val_split,
        'max_steps': args.max_steps,
        'batch_size': args.batch_size,
        'workers': args.workers,
        'use_amp': bool(args.use_amp),
        'guard_metric': args.guard_metric,
        'guard_fast_recall_only': bool(args.guard_fast_recall_only),
        'final_full_eval': bool(args.final_full_eval),
        'save_to_file': bool(args.save_to_file),
        'extra_tag': args.extra_tag,
        'output_root': args.output_root,
    }
    run_meta_path = output_dir / 'run_meta.yaml'
    with open(run_meta_path, 'w') as f:
        yaml.safe_dump(run_meta, f, sort_keys=True)
    logger.info(f'[ARTIFACT] saved run meta: {run_meta_path}')


def _set_requires_grad(module, flag: bool):
    if module is None:
        return 0
    n = 0
    for p in module.parameters():
        p.requires_grad = bool(flag)
        n += p.numel()
    return n


def _freeze_everything_except_conv_cls_out(model, logger):
    total = 0
    trainable = 0
    for p in model.parameters():
        p.requires_grad = False
        total += p.numel()

    dense_head = getattr(model, 'dense_head', None)
    if dense_head is None:
        raise RuntimeError('Model has no dense_head; cannot set CNN online trainable params')

    n_pre = _set_requires_grad(getattr(dense_head, 'conv_cls_pre', None), False)
    n_out = _set_requires_grad(getattr(dense_head, 'conv_cls_out', None), True)
    logger.info(f'[TRAINABLE] dense_head.conv_cls_pre params={n_pre} frozen')
    logger.info(f'[TRAINABLE] dense_head.conv_cls_out params={n_out} trainable')

    for p in model.parameters():
        if p.requires_grad:
            trainable += p.numel()
    ratio = 100.0 * trainable / max(total, 1)
    logger.info(f'[TRAINABLE] total={total}, trainable={trainable} ({ratio:.4f}%)')


def _build_optimizer_for_conv_cls_out(model, online_cfg: Dict, logger):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError('No trainable parameters found for CNN online update')

    opt_name = str(online_cfg.get('optimizer', 'adamw')).lower()
    lr = float(online_cfg.get('lr', 1e-4))
    weight_decay = float(online_cfg.get('weight_decay', 0.0))
    momentum = float(online_cfg.get('momentum', 0.9))

    if opt_name == 'adam':
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise RuntimeError(f'Unsupported optimizer: {opt_name}')

    logger.info(f'[OPTIM] optimizer={opt_name} lr={lr} weight_decay={weight_decay} momentum={momentum}')
    return optimizer


def _build_eval_fn(args, logger, output_dir: Path, batch_size: int):
    eval_counter = {'n': 0}

    def _run_eval(eval_model, eval_loader, *, fast_recall_only: bool, eval_name: str):
        eval_counter['n'] += 1
        logger.info(
            f'[EVAL] start {eval_name} eval #{eval_counter["n"]} '
            f'(fast_recall_only={bool(fast_recall_only)})'
        )
        eval_args = SimpleNamespace(
            save_to_file=bool(args.save_to_file),
            infer_time=False,
            fast_recall_only=bool(fast_recall_only),
            batch_size=batch_size,
            time_log_every=0,
        )
        result_dir = output_dir / 'online_eval' / f'eval_{eval_counter["n"]:04d}'
        result_dir.mkdir(parents=True, exist_ok=True)
        return eval_utils.eval_one_epoch(
            cfg,
            eval_args,
            eval_model,
            eval_loader,
            epoch_id=f'online_cnn_{eval_counter["n"]:04d}',
            logger=logger,
            dist_test=False,
            result_dir=result_dir,
        )

    def eval_fn(eval_model, eval_loader):
        tb_dict = _run_eval(
            eval_model,
            eval_loader,
            fast_recall_only=bool(args.guard_fast_recall_only),
            eval_name='guard',
        )
        if isinstance(tb_dict, dict):
            if args.guard_metric in tb_dict:
                metric = float(tb_dict[args.guard_metric])
                logger.info(f'[EVAL] metric {args.guard_metric}={metric:.6f}')
                return metric
            logger.warning(
                f'[EVAL] guard metric {args.guard_metric} not found; keys={sorted(list(tb_dict.keys()))[:20]}'
            )
            for k in ('recall/rcnn_0.3', 'recall/rcnn_0.5', 'mAP', 'map'):
                if k in tb_dict:
                    metric = float(tb_dict[k])
                    logger.info(f'[EVAL] fallback metric {k}={metric:.6f}')
                    return metric
        logger.warning('[EVAL] no valid metric found in eval output, fallback=0.0')
        return 0.0

    return eval_fn, _run_eval


def _load_runner_best_state_into_model(runner: OnlineCNNRunner, logger):
    best_state = getattr(runner, 'best_state', None)
    if not isinstance(best_state, dict):
        logger.warning('[BEST] runner.best_state is not available; keep current model for final eval')
        return
    model_state = best_state.get('model_state', None)
    if not isinstance(model_state, dict):
        logger.warning('[BEST] runner.best_state has no valid model_state; keep current model for final eval')
        return
    runner.model.load_state_dict(model_state, strict=False)
    if hasattr(runner.model, 'dense_head') and hasattr(runner.model.dense_head, 'hd_mode'):
        runner.model.dense_head.hd_mode = 'baseline'
    logger.info('[BEST] loaded runner.best_state model into memory for final evaluation')


def main():
    args, _ = parse_config()
    output_dir, logger = _build_output_and_logger(args)

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    if args.batch_size <= 0:
        raise RuntimeError(f'Invalid batch_size={args.batch_size}, must be > 0')

    online_cfg = _load_online_cfg_file(args.online_cfg)
    logger.info(f'[ONLINE_CFG] loaded from file: {args.online_cfg}')
    logger.info(f'[ONLINE_CFG] file keys: {sorted(list(online_cfg.keys()))}')
    online_cfg.update(_build_online_cfg_dict(args))
    logger.info(f'[ONLINE_CFG] final keys: {sorted(list(online_cfg.keys()))}')
    logger.info(f'[ONLINE_CFG] final values:\n{yaml.safe_dump(online_cfg, sort_keys=True)}')
    _save_run_artifacts(output_dir, args, online_cfg, logger)

    stream_cfg = _override_split(cfg.DATA_CONFIG, args.stream_split)
    logger.info(f"[DATA] stream split={stream_cfg.DATA_SPLIT.get('test', 'N/A')}")
    stream_set, stream_loader, _ = build_dataloader(
        dataset_cfg=stream_cfg,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )
    logger.info(f'[DATA] stream dataset size={_safe_len(stream_set)} loader_len={_safe_len(stream_loader)}')
    if _safe_len(stream_set) <= 0:
        raise RuntimeError('stream dataset is empty')

    if args.val_split is not None:
        val_cfg = _override_split(cfg.DATA_CONFIG, args.val_split)
        logger.info(f"[DATA] val split={val_cfg.DATA_SPLIT.get('test', 'N/A')}")
        val_set, val_loader, _ = build_dataloader(
            dataset_cfg=val_cfg,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=False,
            workers=args.workers,
            logger=logger,
            training=False,
        )
        logger.info(f'[DATA] val dataset size={_safe_len(val_set)} loader_len={_safe_len(val_loader)}')
    else:
        val_loader = None
        logger.info('[DATA] val_split is None, guard evaluation disabled')

    logger.info('[MODEL] building detector...')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=stream_set)
    logger.info('[MODEL] loading checkpoint...')
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    if hasattr(model, 'dense_head') and hasattr(model.dense_head, 'hd_mode'):
        model.dense_head.hd_mode = 'baseline'
        logger.info('[MODEL] forced dense_head.hd_mode = baseline')

    _freeze_everything_except_conv_cls_out(model, logger)
    optimizer = _build_optimizer_for_conv_cls_out(model, online_cfg, logger)

    eval_fn, run_eval = _build_eval_fn(args, logger, output_dir, args.batch_size)

    runner = OnlineCNNRunner(
        model=model,
        optimizer=optimizer,
        stream_loader=stream_loader,
        logger=logger,
        cfg=online_cfg,
        val_loader=val_loader,
        eval_fn=eval_fn if val_loader is not None else None,
        output_dir=str(output_dir / 'online_state'),
        state_save_prefix='online_cnn',
        use_amp=bool(args.use_amp),
    )

    logger.info('[ONLINE] CNN runner start')
    runner.run(max_steps=args.max_steps)
    logger.info('[ONLINE] CNN runner finished')

    if val_loader is not None and bool(args.final_full_eval):
        _load_runner_best_state_into_model(runner, logger)
        logger.info('[EVAL] start final full official evaluation once')
        tb_dict = run_eval(model, val_loader, fast_recall_only=False, eval_name='final_full')
        if isinstance(tb_dict, dict):
            if args.guard_metric in tb_dict:
                logger.info(f'[EVAL][FINAL] {args.guard_metric}={float(tb_dict[args.guard_metric]):.6f}')
            for k in ('recall/rcnn_0.3', 'recall/rcnn_0.5', 'mAP', 'map'):
                if k in tb_dict:
                    logger.info(f'[EVAL][FINAL] {k}={float(tb_dict[k]):.6f}')
        logger.info('[EVAL] final full official evaluation finished')

    logger.info('**********************End online CNN**********************')


if __name__ == '__main__':
    main()
