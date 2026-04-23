import _init_path
import argparse
import copy
import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from eval_utils import eval_utils
from modules.online_pure_cnn_runner import OnlinePureCNNRunner
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
    parser = argparse.ArgumentParser(description="HyperRadar Pure CNN Unsupervised Online Adaptation")

    parser.add_argument("--cfg_file", type=str, required=True, help="Pure CNN detector config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Pure CNN detector checkpoint")
    parser.add_argument("--online_cfg", type=str, required=True,
                        help="Path to pure CNN online yaml config. Expected root key: PURE_CNN_ONLINE")

    parser.add_argument("--extra_tag", type=str, default="online_pure_cnn_unsupervised", help="Output tag")
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
    parser.add_argument("--freeze_vfe", type=int, choices=[0, 1], default=None)
    parser.add_argument("--freeze_map_to_bev", type=int, choices=[0, 1], default=None)
    parser.add_argument("--freeze_backbone_2d", type=int, choices=[0, 1], default=None)
    parser.add_argument("--freeze_box_head", type=int, choices=[0, 1], default=None)
    parser.add_argument("--freeze_dir_head", type=int, choices=[0, 1], default=None)

    # Pseudo-label / update overrides
    parser.add_argument("--tau_prob", type=float, default=None)
    parser.add_argument("--tau_margin", type=float, default=None)
    parser.add_argument("--min_tau_prob", type=float, default=None)
    parser.add_argument("--min_tau_margin", type=float, default=None)
    parser.add_argument("--select_top_ratio", type=float, default=None)
    parser.add_argument("--select_min_k", type=int, default=None)
    parser.add_argument("--select_max_k", type=int, default=None)
    parser.add_argument("--consistency_bonus", type=float, default=None)
    parser.add_argument("--inconsistency_penalty", type=float, default=None)
    parser.add_argument("--use_teacher", type=int, choices=[0, 1], default=None)
    parser.add_argument("--teacher_momentum", type=float, default=None)
    parser.add_argument("--update_every_n_steps", type=int, default=None)
    parser.add_argument("--min_selected_anchors", type=int, default=None)
    parser.add_argument("--min_confpass_anchors", type=int, default=None)
    parser.add_argument("--max_per_class_per_step", type=int, default=None)
    parser.add_argument("--min_per_class_for_update", type=int, default=None)
    parser.add_argument("--class_balance_enable", type=int, choices=[0, 1], default=None)
    parser.add_argument("--class_balance_min_classes_to_update", type=int, default=None)
    parser.add_argument("--class_balance_max_pending_steps", type=int, default=None)
    parser.add_argument("--pseudo_logits_source", type=str, choices=["final", "origin"], default=None)
    parser.add_argument("--zero_update_tau_prob_step", type=float, default=None)
    parser.add_argument("--zero_update_tau_margin_step", type=float, default=None)

    # Optimizer overrides
    parser.add_argument("--grad_clip", type=float, default=None)

    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"], default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)

    # Eval / save overrides
    parser.add_argument("--eval_every_updates", type=int, default=None)
    parser.add_argument("--metric_key", "--guard_metric", dest="metric_key", type=str, default=None)
    parser.add_argument("--fast_recall_only", type=int, choices=[0, 1], default=None)
    parser.add_argument("--final_full_eval", type=int, choices=[0, 1], default=None)
    parser.add_argument("--guard_max_drop", type=float, default=None)
    parser.add_argument("--guard_use_best", type=int, choices=[0, 1], default=None)
    parser.add_argument("--save_every_updates", type=int, default=None)
    parser.add_argument("--log_interval_steps", "--log_every_n_batches", dest="log_interval_steps", type=int, default=None)
    parser.add_argument("--save_best_model", type=int, choices=[0, 1], default=None)
    parser.add_argument("--save_last_model", type=int, choices=[0, 1], default=None)
    parser.add_argument("--experiment_note", type=str, default=None)
    parser.add_argument("--use_amp", action="store_true", default=False)

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

    log_file = output_dir / ('online_pure_cnn_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start pure CNN unsupervised online logging**********************')
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


def _override_split(dataset_cfg, split_name: Optional[str], info_paths: Optional[list] = None):
    cfg_local = copy.deepcopy(dataset_cfg)
    if split_name is None:
        return cfg_local
    cfg_local.DATA_SPLIT['test'] = split_name
    split_to_info = {
        'train': ['kitti_infos_train.pkl'],
        'val': ['kitti_infos_val.pkl'],
        'test': ['kitti_infos_test.pkl'],
    }
    if hasattr(cfg_local, 'INFO_PATH'):
        if info_paths is not None:
            cfg_local.INFO_PATH['test'] = list(info_paths)
        elif split_name in split_to_info:
            cfg_local.INFO_PATH['test'] = split_to_info[split_name]
        elif split_name in cfg_local.INFO_PATH:
            cfg_local.INFO_PATH['test'] = copy.deepcopy(cfg_local.INFO_PATH[split_name])
        else:
            raise RuntimeError(
                f'split "{split_name}" has no matching INFO_PATH entry. '
                f'Please add DATA_CONFIG.INFO_PATH["{split_name}"] to the detector yaml '
                f'or pass explicit info paths via stream_info_paths/eval_info_paths.'
            )
    return cfg_local


def _load_online_cfg_file(path: str) -> Dict:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise RuntimeError(f'Invalid online cfg file: expected dict root, got {type(raw)}')
    online_cfg = raw.get('PURE_CNN_ONLINE', raw)
    if not isinstance(online_cfg, dict):
        raise RuntimeError(f'Invalid PURE_CNN_ONLINE section type: {type(online_cfg)}')
    return dict(online_cfg)


def _build_online_cfg_dict(args) -> Dict:
    mapping = {
        'stream_split': args.stream_split,
        'eval_split': args.eval_split,
        'stream_ratio': args.stream_ratio,
        'max_stream_samples': args.max_stream_samples,
        'stream_info_paths': args.stream_info_paths,
        'eval_info_paths': args.eval_info_paths,
        'tau_prob': args.tau_prob,
        'tau_margin': args.tau_margin,
        'min_tau_prob': args.min_tau_prob,
        'min_tau_margin': args.min_tau_margin,
        'select_top_ratio': args.select_top_ratio,
        'select_min_k': args.select_min_k,
        'select_max_k': args.select_max_k,
        'consistency_bonus': args.consistency_bonus,
        'inconsistency_penalty': args.inconsistency_penalty,
        'teacher_momentum': args.teacher_momentum,
        'update_every_n_steps': args.update_every_n_steps,
        'min_selected_anchors': args.min_selected_anchors,
        'min_confpass_anchors': args.min_confpass_anchors,
        'max_per_class_per_step': args.max_per_class_per_step,
        'min_per_class_for_update': args.min_per_class_for_update,
        'class_balance_min_classes_to_update': args.class_balance_min_classes_to_update,
        'class_balance_max_pending_steps': args.class_balance_max_pending_steps,
        'pseudo_logits_source': args.pseudo_logits_source,
        'zero_update_tau_prob_step': args.zero_update_tau_prob_step,
        'zero_update_tau_margin_step': args.zero_update_tau_margin_step,
        'grad_clip': args.grad_clip,
        'optimizer': args.optimizer,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'eval_every_updates': args.eval_every_updates,
        'metric_key': args.metric_key,
        'guard_max_drop': args.guard_max_drop,
        'save_every_updates': args.save_every_updates,
        'log_interval_steps': args.log_interval_steps,
        'experiment_note': args.experiment_note,
    }

    bool_args = [
        'use_stream_prefix',
        'train_conv_cls_pre',
        'train_conv_cls_out',
        'freeze_vfe',
        'freeze_map_to_bev',
        'freeze_backbone_2d',
        'freeze_box_head',
        'freeze_dir_head',
        'use_teacher',
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
        yaml.safe_dump({'PURE_CNN_ONLINE': online_cfg}, f, sort_keys=False)
    logger.info(f'[ARTIFACT] saved run meta and resolved online cfg to {str(output_dir)}')


def _build_loader_for_split(split_name: str, batch_size: int, workers: int, logger, info_paths: Optional[list] = None):
    local_cfg = _override_split(cfg.DATA_CONFIG, split_name, info_paths=info_paths)
    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=local_cfg,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=False,
        workers=workers,
        logger=logger,
        training=False,
    )
    return dataset, dataloader


def _make_stream_subset_indices(dataset, online_cfg: Dict, logger):
    total = len(dataset)
    ratio = float(online_cfg.get('stream_ratio', 1.0))
    max_samples = int(online_cfg.get('max_stream_samples', 0))
    use_prefix = bool(online_cfg.get('use_stream_prefix', True))
    seed = int(online_cfg.get('stream_seed', 0))

    n = int(total * ratio)
    if ratio > 0 and n <= 0:
        n = 1
    n = min(total, n)
    if max_samples > 0:
        n = min(n, max_samples)
    if n <= 0:
        logger.warning('[DATA] stream subset size resolved to 0; no update samples will be used')
        return []

    if use_prefix:
        indices = list(range(n))
        logger.info(f'[DATA] stream subset uses prefix of size {n}/{total}')
        return indices

    g = torch.Generator(device='cpu')
    g.manual_seed(seed)
    perm = torch.randperm(total, generator=g).tolist()
    indices = perm[:n]
    logger.info(f'[DATA] stream subset uses random sample of size {n}/{total} with seed={seed}')
    return indices


def _build_subset_loader(base_dataset, indices, batch_size: int, workers: int, seed: int):
    subset = Subset(base_dataset, indices)
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=workers,
        shuffle=False,
        collate_fn=base_dataset.collate_batch,
        drop_last=False,
        sampler=None,
        timeout=0,
        worker_init_fn=partial(common_utils.worker_init_fn, seed=seed),
    )
    return subset, dataloader


def _build_eval_fn(args, logger, output_dir: Path, online_cfg: Dict, eval_batch_size: int):
    eval_counter = {'n': 0}

    def _run_eval(eval_model, eval_loader, *, fast_recall_only: bool, eval_name: str):
        eval_counter['n'] += 1
        logger.info(
            f'[EVAL] start {eval_name} eval #{eval_counter["n"]} '
            f'(fast_recall_only={bool(fast_recall_only)})'
        )
        eval_args = SimpleNamespace(
            save_to_file=bool(args.save_to_file or online_cfg.get('save_to_file', False)),
            infer_time=False,
            fast_recall_only=bool(fast_recall_only),
            batch_size=int(eval_batch_size),
            time_log_every=0,
        )
        result_dir = output_dir / 'online_eval' / f'eval_{eval_counter["n"]:04d}'
        result_dir.mkdir(parents=True, exist_ok=True)
        return eval_utils.eval_one_epoch(
            cfg,
            eval_args,
            eval_model,
            eval_loader,
            epoch_id=f'pure_cnn_online_{eval_counter["n"]:04d}',
            logger=logger,
            dist_test=False,
            result_dir=result_dir,
        )

    def eval_fn(eval_model, eval_loader):
        return _run_eval(
            eval_model,
            eval_loader,
            fast_recall_only=bool(online_cfg.get('fast_recall_only', True)),
            eval_name='guard',
        )

    return eval_fn, _run_eval


def _set_requires_grad(module, flag: bool):
    if module is None:
        return 0
    n = 0
    for p in module.parameters():
        p.requires_grad = bool(flag)
        n += p.numel()
    return n


def _log_trainable_params(model, logger):
    total = 0
    trainable = 0
    names = []
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            names.append(name)
    logger.info(f'[TRAINABLE] total={total}, trainable={trainable} ({100.0 * trainable / max(total, 1):.4f}%)')
    for name in names:
        logger.info(f'[TRAINABLE] {name}')


def _configure_trainable_scope(model, online_cfg: Dict, logger):
    for p in model.parameters():
        p.requires_grad = False

    if not bool(online_cfg.get('freeze_vfe', True)):
        n = _set_requires_grad(getattr(model, 'vfe', None), True)
        logger.info(f'[TRAINABLE] vfe params={n} trainable')
    else:
        logger.info('[TRAINABLE] vfe frozen')

    if not bool(online_cfg.get('freeze_map_to_bev', True)):
        n = _set_requires_grad(getattr(model, 'map_to_bev_module', None), True)
        logger.info(f'[TRAINABLE] map_to_bev_module params={n} trainable')
    else:
        logger.info('[TRAINABLE] map_to_bev_module frozen')

    if not bool(online_cfg.get('freeze_backbone_2d', True)):
        n = _set_requires_grad(getattr(model, 'backbone_2d', None), True)
        logger.info(f'[TRAINABLE] backbone_2d params={n} trainable')
    else:
        logger.info('[TRAINABLE] backbone_2d frozen')

    dense_head = getattr(model, 'dense_head', None)
    if dense_head is None:
        raise RuntimeError('Model has no dense_head; cannot configure pure CNN online trainable scope')

    train_pre = bool(online_cfg.get('train_conv_cls_pre', False))
    n_pre = _set_requires_grad(getattr(dense_head, 'conv_cls_pre', None), train_pre)
    logger.info(f"[TRAINABLE] dense_head.conv_cls_pre params={n_pre} {'trainable' if train_pre else 'frozen'}")

    train_out = bool(online_cfg.get('train_conv_cls_out', True))
    n_out = _set_requires_grad(getattr(dense_head, 'conv_cls_out', None), train_out)
    logger.info(f"[TRAINABLE] dense_head.conv_cls_out params={n_out} {'trainable' if train_out else 'frozen'}")

    if not bool(online_cfg.get('freeze_box_head', True)):
        n = _set_requires_grad(getattr(dense_head, 'conv_box', None), True)
        logger.info(f'[TRAINABLE] dense_head.conv_box params={n} trainable')
    else:
        logger.info('[TRAINABLE] dense_head.conv_box frozen')

    if not bool(online_cfg.get('freeze_dir_head', True)):
        n = _set_requires_grad(getattr(dense_head, 'conv_dir_cls', None), True)
        logger.info(f'[TRAINABLE] dense_head.conv_dir_cls params={n} trainable')
    else:
        logger.info('[TRAINABLE] dense_head.conv_dir_cls frozen')

    _log_trainable_params(model, logger)


def _build_optimizer(model, online_cfg: Dict, logger):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError('No trainable parameters found for pure CNN online update')

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


def _load_runner_best_state_into_model(runner: OnlinePureCNNRunner, logger):
    best_state = getattr(runner, 'best_state', None)
    if not isinstance(best_state, dict):
        logger.warning('[BEST] runner.best_state is not available; keep current model for final eval')
        return
    model_state = best_state.get('model_state', None)
    if not isinstance(model_state, dict):
        logger.warning('[BEST] runner.best_state has no valid model_state; keep current model for final eval')
        return
    runner.model.load_state_dict(model_state, strict=False)
    logger.info('[BEST] loaded runner.best_state model into memory for final evaluation')


def main():
    args, _cfg = parse_config()
    output_dir, logger = _build_output_and_logger(args)

    online_cfg = _load_online_cfg_file(args.online_cfg)
    logger.info(f'[ONLINE_CFG] loaded from file: {args.online_cfg}')
    logger.info(f'[ONLINE_CFG] file keys: {sorted(list(online_cfg.keys()))}')
    online_cfg.update(_build_online_cfg_dict(args))
    logger.info(f'[ONLINE_CFG] final keys: {sorted(list(online_cfg.keys()))}')
    logger.info(f'[ONLINE_CFG] final values:\n{yaml.safe_dump(online_cfg, sort_keys=True)}')
    _save_run_artifacts(output_dir, args, online_cfg, logger)

    stream_split = str(online_cfg.get('stream_split', 'val_stream_by_scene'))
    eval_split = str(online_cfg.get('eval_split', 'val_eval_by_scene'))
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

    logger.info('[MODEL] building pure CNN detector...')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=stream_set_full)
    logger.info('[MODEL] loading checkpoint...')
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    logger.info('[MODEL] model moved to CUDA and set to eval')

    dense_head = getattr(model, 'dense_head', None)
    if dense_head is None:
        raise RuntimeError('Cannot find dense_head in pure CNN model.')
    if hasattr(dense_head, 'hd_core') and getattr(dense_head, 'hd_core') is not None:
        raise RuntimeError(
            'Pure CNN online script received a model with dense_head.hd_core. '
            'Use cfgs/kitti_models/pointpillar_vod.yaml for NO-HD experiments.'
        )
    if hasattr(dense_head, 'hd_enabled') and bool(getattr(dense_head, 'hd_enabled')):
        raise RuntimeError(
            'Pure CNN online script received HD-enabled dense_head. '
            'Use cfgs/kitti_models/pointpillar_vod.yaml for NO-HD experiments.'
        )

    _configure_trainable_scope(model, online_cfg, logger)
    optimizer = _build_optimizer(model, online_cfg, logger)
    eval_fn, run_eval = _build_eval_fn(args, logger, output_dir, online_cfg, eval_bs)

    runner = OnlinePureCNNRunner(
        model=model,
        optimizer=optimizer,
        stream_loader=stream_loader,
        logger=logger,
        cfg=online_cfg,
        val_loader=eval_loader,
        eval_fn=eval_fn,
        output_dir=str(output_dir / 'online_state'),
        state_save_prefix='online_pure_cnn',
        use_amp=bool(args.use_amp),
    )

    logger.info('[ONLINE] pure CNN unsupervised runner start')
    runner.run(max_steps=args.max_steps)
    logger.info('[ONLINE] pure CNN unsupervised runner finished')

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

    logger.info('**********************End pure CNN unsupervised online**********************')


if __name__ == '__main__':
    main()
