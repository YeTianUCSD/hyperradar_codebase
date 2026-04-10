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
from modules.online_state_io import OnlineStateIO
from modules.supervised_online_hd_runner import SupervisedOnlineHDRunner
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
    parser = argparse.ArgumentParser(description="HyperRadar Supervised Online HD Adaptation")

    parser.add_argument("--cfg_file", type=str, required=True, help="Detector config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Detector checkpoint")
    parser.add_argument("--source_memory", type=str, default=None,
                        help="Optional HD memory/prototype file used to initialize the supervised online run")
    parser.add_argument("--online_cfg", type=str, required=True,
                        help="Path to supervised online yaml config. Expected root key: SUPERVISED_ONLINE")

    parser.add_argument("--extra_tag", type=str, default="online_hd_supervised", help="Output tag")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Optional output root dir. If set, outputs go to <output_root>/<extra_tag>")

    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for stream loader")
    parser.add_argument("--workers", type=int, default=1, help="Workers for stream loader")
    parser.add_argument("--eval_batch_size", type=int, default=None, help="Batch size for eval loader")
    parser.add_argument("--eval_workers", type=int, default=None, help="Workers for eval loader")

    parser.add_argument("--max_steps", type=int, default=-1, help="Max stream steps (-1 for all)")
    parser.add_argument("--save_to_file", action="store_true", default=False,
                        help="Save eval predictions to files")

    # Optional supervised-online overrides
    parser.add_argument("--stream_split", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default=None)
    parser.add_argument("--stream_ratio", type=float, default=None)
    parser.add_argument("--max_stream_samples", type=int, default=None)
    parser.add_argument("--use_stream_prefix", type=int, choices=[0, 1], default=None)
    parser.add_argument("--stream_seed", type=int, default=None)
    parser.add_argument("--stream_info_paths", type=str, nargs='+', default=None,
                        help="Optional explicit info pkl path(s) for the stream split")
    parser.add_argument("--eval_info_paths", type=str, nargs='+', default=None,
                        help="Optional explicit info pkl path(s) for the eval split")

    parser.add_argument("--feature_source", type=str, choices=["cls", "bev"], default=None)
    parser.add_argument("--update_mode", type=str, choices=["train", "retrain", "both"], default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--update_every_n_batches", type=int, default=None)
    parser.add_argument("--normalize_every_updates", type=int, default=None)
    parser.add_argument("--max_pos_per_class_per_batch", type=int, default=None)
    parser.add_argument("--min_pos_per_class_per_batch", type=int, default=None)
    parser.add_argument("--max_total_pos_per_batch", type=int, default=None)
    parser.add_argument("--eval_every_updates", type=int, default=None)
    parser.add_argument("--metric_key", type=str, default=None)
    parser.add_argument("--fast_recall_only", type=int, choices=[0, 1], default=None)
    parser.add_argument("--final_full_eval", type=int, choices=[0, 1], default=None)
    parser.add_argument("--save_every_updates", type=int, default=None)
    parser.add_argument("--log_every_n_batches", type=int, default=None)
    parser.add_argument("--save_best_memory", type=int, choices=[0, 1], default=None)
    parser.add_argument("--save_last_memory", type=int, choices=[0, 1], default=None)
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

    log_file = output_dir / ('online_supervised_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start supervised online HD logging**********************')
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
        logger.info('{:24} {}'.format(key, val))
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


def _load_supervised_online_cfg_file(path: str) -> Dict:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise RuntimeError(f'Invalid online cfg file: expected dict root, got {type(raw)}')
    online_cfg = raw.get('SUPERVISED_ONLINE', raw)
    if not isinstance(online_cfg, dict):
        raise RuntimeError(f'Invalid SUPERVISED_ONLINE section type: {type(online_cfg)}')
    return dict(online_cfg)


@torch.no_grad()
def _load_memory_payload_into_hd_core(hd_core, memory_path: str, logger):
    payload = torch.load(memory_path, map_location='cpu')

    if torch.is_tensor(payload):
        raise RuntimeError('Unsupported memory payload: bare tensor. Expected dict payload.')
    if not isinstance(payload, dict):
        raise RuntimeError(f'Unsupported memory payload type: {type(payload)}')

    mem = getattr(hd_core, 'memory', None)
    if mem is None:
        raise RuntimeError('hd_core.memory is missing; cannot load source_memory')

    loaded_any = False

    if 'memory_state' in payload and isinstance(payload['memory_state'], dict):
        mem.load_state_dict(payload['memory_state'], strict=False)
        loaded_any = True
        logger.info('[MEM] loaded hd_core.memory state_dict from payload[memory_state]')
    elif 'memory' in payload and isinstance(payload['memory'], dict):
        mem.load_state_dict(payload['memory'], strict=False)
        loaded_any = True
        logger.info('[MEM] loaded hd_core.memory state_dict from payload[memory]')

    if 'classify_weights' in payload and hasattr(mem, 'classify_weights'):
        mem.classify_weights.copy_(payload['classify_weights'].to(mem.classify_weights.device).to(mem.classify_weights.dtype))
        loaded_any = True
        logger.info('[MEM] loaded classify_weights from payload')
    if 'prototypes' in payload and payload['prototypes'] is not None and hasattr(mem, 'prototypes'):
        mem.prototypes.copy_(payload['prototypes'].to(mem.prototypes.device).to(mem.prototypes.dtype))
        loaded_any = True
        logger.info('[MEM] loaded prototypes from payload')
    if 'bg_weight' in payload and hasattr(mem, 'bg_weight'):
        mem.bg_weight.copy_(payload['bg_weight'].to(mem.bg_weight.device).to(mem.bg_weight.dtype))
        loaded_any = True
    if 'bg_prototype' in payload and payload['bg_prototype'] is not None and hasattr(mem, 'bg_prototype'):
        mem.bg_prototype.copy_(payload['bg_prototype'].to(mem.bg_prototype.device).to(mem.bg_prototype.dtype))
        loaded_any = True

    embedder = getattr(hd_core, 'embedder', None)
    if 'embedder' in payload and embedder is not None and isinstance(payload['embedder'], dict):
        try:
            embedder.load_state_dict(payload['embedder'], strict=False)
            logger.info('[MEM] loaded embedder state_dict from payload[embedder]')
        except Exception as e:
            logger.warning(f'[MEM] failed loading embedder from source_memory: {repr(e)}')

    if not loaded_any:
        raise RuntimeError(
            'No valid memory fields found in source_memory payload. '
            'Expected memory_state/memory or classify_weights/prototypes.'
        )

    mem.normalize_()
    logger.info('[MEM] source memory loaded and normalized')


def _build_supervised_online_cfg_dict(args) -> Dict:
    mapping = {
        'stream_split': args.stream_split,
        'eval_split': args.eval_split,
        'stream_ratio': args.stream_ratio,
        'max_stream_samples': args.max_stream_samples,
        'stream_info_paths': args.stream_info_paths,
        'eval_info_paths': args.eval_info_paths,
        'feature_source': args.feature_source,
        'update_mode': args.update_mode,
        'alpha': args.alpha,
        'update_every_n_batches': args.update_every_n_batches,
        'normalize_every_updates': args.normalize_every_updates,
        'max_pos_per_class_per_batch': args.max_pos_per_class_per_batch,
        'min_pos_per_class_per_batch': args.min_pos_per_class_per_batch,
        'max_total_pos_per_batch': args.max_total_pos_per_batch,
        'eval_every_updates': args.eval_every_updates,
        'metric_key': args.metric_key,
        'save_every_updates': args.save_every_updates,
        'log_every_n_batches': args.log_every_n_batches,
        'experiment_note': args.experiment_note,
    }
    if args.use_stream_prefix is not None:
        mapping['use_stream_prefix'] = bool(args.use_stream_prefix)
    if args.stream_seed is not None:
        mapping['stream_seed'] = int(args.stream_seed)
    if args.fast_recall_only is not None:
        mapping['fast_recall_only'] = bool(args.fast_recall_only)
    if args.final_full_eval is not None:
        mapping['final_full_eval'] = bool(args.final_full_eval)
    if args.save_best_memory is not None:
        mapping['save_best_memory'] = bool(args.save_best_memory)
    if args.save_last_memory is not None:
        mapping['save_last_memory'] = bool(args.save_last_memory)
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
        yaml.safe_dump({'SUPERVISED_ONLINE': online_cfg}, f, sort_keys=False)
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
            save_to_file=bool(args.save_to_file),
            infer_time=False,
            fast_recall_only=bool(fast_recall_only),
            batch_size=int(eval_batch_size),
            time_log_every=0,
        )
        result_dir = output_dir / 'online_eval' / f'eval_{eval_counter["n"]:04d}'
        result_dir.mkdir(parents=True, exist_ok=True)
        tb_dict = eval_utils.eval_one_epoch(
            cfg,
            eval_args,
            eval_model,
            eval_loader,
            epoch_id=f'sup_online_{eval_counter["n"]:04d}',
            logger=logger,
            dist_test=False,
            result_dir=result_dir,
        )
        return tb_dict

    def eval_fn(eval_model, eval_loader):
        tb_dict = _run_eval(
            eval_model,
            eval_loader,
            fast_recall_only=bool(online_cfg.get('fast_recall_only', True)),
            eval_name='guard'
        )
        return tb_dict

    return eval_fn, _run_eval


def _load_runner_best_state_into_model(runner: SupervisedOnlineHDRunner, logger):
    best_state = getattr(runner, 'best_state', None)
    if not isinstance(best_state, dict):
        logger.warning('[BEST] runner.best_state is not available; keep current memory for final eval')
        return
    mem_state = best_state.get('memory_state', None)
    if not isinstance(mem_state, dict):
        logger.warning('[BEST] runner.best_state has no valid memory_state; keep current memory for final eval')
        return
    runner.hd_core.memory.load_state_dict(mem_state, strict=False)
    runner.hd_core.memory.normalize_()
    logger.info('[BEST] loaded runner.best_state memory into model for final evaluation')


def main():
    args, _cfg = parse_config()
    output_dir, logger = _build_output_and_logger(args)

    online_cfg = _load_supervised_online_cfg_file(args.online_cfg)
    logger.info(f'[ONLINE_CFG] loaded from file: {args.online_cfg}')
    logger.info(f'[ONLINE_CFG] file keys: {sorted(list(online_cfg.keys()))}')
    online_cfg.update(_build_supervised_online_cfg_dict(args))
    logger.info(f'[ONLINE_CFG] final keys: {sorted(list(online_cfg.keys()))}')
    logger.info(f'[ONLINE_CFG] final values:\n{yaml.safe_dump(online_cfg, sort_keys=True)}')
    _save_run_artifacts(output_dir, args, online_cfg, logger)

    stream_split = str(online_cfg.get('stream_split', 'val_stream'))
    eval_split = str(online_cfg.get('eval_split', 'val_eval'))
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

    if args.source_memory is not None:
        hd_core = getattr(getattr(model, 'dense_head', None), 'hd_core', None)
        if hd_core is None:
            raise RuntimeError('Cannot find dense_head.hd_core in model.')
        logger.info(f'[MEM] loading source memory: {args.source_memory}')
        _load_memory_payload_into_hd_core(hd_core, args.source_memory, logger)

    eval_fn, run_eval = _build_eval_fn(args, logger, output_dir, online_cfg, eval_bs)

    runner = SupervisedOnlineHDRunner(
        model=model,
        stream_loader=stream_loader,
        logger=logger,
        cfg=online_cfg,
        eval_loader=eval_loader,
        eval_fn=eval_fn,
        output_dir=str(output_dir / 'online_state'),
        state_save_prefix='supervised_online_hd',
    )

    logger.info('[ONLINE] supervised runner start')
    runner.run(max_steps=args.max_steps)
    logger.info('[ONLINE] supervised runner finished')

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

    logger.info('**********************End supervised online HD**********************')


if __name__ == '__main__':
    main()
