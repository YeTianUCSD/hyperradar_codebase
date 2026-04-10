import _init_path
import argparse
import copy
import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import yaml

from eval_utils import eval_utils
from modules.online_hd_runner import OnlineHDRunner
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
    parser = argparse.ArgumentParser(description="HyperRadar Online HD Adaptation")

    parser.add_argument("--cfg_file", type=str, required=True, help="Config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Detector checkpoint")
    parser.add_argument("--extra_tag", type=str, default="online_hd", help="Output tag")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Optional output root dir. If set, outputs go to <output_root>/<extra_tag>")

    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--workers", type=int, default=1, help="Dataloader workers")

    parser.add_argument("--stream_split", type=str, default=None,
                        help="Split name used for stream (e.g., train/val/test). If None, keep cfg default test split")
    parser.add_argument("--val_split", type=str, default=None,
                        help="Split name used for guard evaluation (e.g., val/test). If None, disable guard eval")

    parser.add_argument("--source_memory", type=str, default=None,
                        help="Optional HD memory/prototype file to initialize online source anchor")
    parser.add_argument("--resume_online_state", type=str, default=None,
                        help="Resume online state saved by OnlineHDRunner")
    parser.add_argument("--online_cfg", type=str, default=None,
                        help="Path to online yaml config. Expected root key: ONLINE")

    parser.add_argument("--max_steps", type=int, default=-1, help="Max stream steps (-1 for all)")
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use mixed precision for forward")

    parser.add_argument("--guard_metric", type=str, default="recall/rcnn_0.3",
                        help="Metric key used from eval dict for guard")
    parser.add_argument("--fast_recall_only", action="store_true", default=False,
                        help="Use recall-only eval for faster guard checks")
    parser.add_argument("--guard_fast_recall_only", type=int, choices=[0, 1], default=1,
                        help="Use recall-only eval during online guard checks (recommended: 1)")
    parser.add_argument("--final_full_eval", type=int, choices=[0, 1], default=1,
                        help="Run one full official evaluation after online updates finish")
    parser.add_argument("--save_to_file", action="store_true", default=False,
                        help="Save eval predictions to files")

    # Online config overrides (map to OnlineHDConfig fields)
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

    parser.add_argument("--update_mode", type=str, choices=["train", "retrain", "both"], default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--source_pullback_lambda", type=float, default=None)
    parser.add_argument("--max_per_class_per_step", type=int, default=None)
    parser.add_argument("--min_per_class_for_update", type=int, default=None)
    parser.add_argument("--class_balance_enable", type=int, choices=[0, 1], default=None)
    parser.add_argument("--class_balance_min_classes_to_update", type=int, default=None)
    parser.add_argument("--class_balance_max_pending_steps", type=int, default=None)

    parser.add_argument("--replay_enable", type=int, choices=[0, 1], default=None)
    parser.add_argument("--replay_cap_per_class", type=int, default=None)
    parser.add_argument("--replay_per_class", type=int, default=None)
    parser.add_argument("--replay_alpha_scale", type=float, default=None)

    parser.add_argument("--eval_every_updates", type=int, default=None)
    parser.add_argument("--guard_max_drop", type=float, default=None)
    parser.add_argument("--guard_use_best", type=int, choices=[0, 1], default=None)

    parser.add_argument("--save_every_updates", type=int, default=None)
    parser.add_argument("--log_interval_steps", type=int, default=None)
    parser.add_argument("--feat_source", type=str, choices=["cls", "bev"], default=None)
    parser.add_argument("--pseudo_logits_source", type=str, choices=["final", "origin", "hd"], default=None)

    parser.add_argument("--zero_update_tau_prob_step", type=float, default=None)
    parser.add_argument("--zero_update_tau_margin_step", type=float, default=None)

    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()

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

    log_file = output_dir / ('online_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start online HD logging**********************')
    logger.info(f'cfg_file={args.cfg_file}')
    logger.info(f'ckpt={args.ckpt}')
    logger.info(f'source_memory={args.source_memory}')
    logger.info(f'resume_online_state={args.resume_online_state}')
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


def _find_hd_core(model):
    dense_head = getattr(model, 'dense_head', None)
    if dense_head is None:
        return None
    return getattr(dense_head, 'hd_core', None)


@torch.no_grad()
def _try_load_hd_from_model_state_dict(hd_core, state_dict: dict, logger) -> bool:
    if not isinstance(state_dict, dict):
        return False

    prefix_candidates = [
        'module.dense_head.hd_core.',
        'dense_head.hd_core.',
        'model.dense_head.hd_core.',
        'hd_core.',
    ]

    loaded = False
    for prefix in prefix_candidates:
        mem_sub = {}
        emb_sub = {}
        p_mem = prefix + 'memory.'
        p_emb = prefix + 'embedder.'
        for k, v in state_dict.items():
            if not torch.is_tensor(v):
                continue
            if k.startswith(p_mem):
                mem_sub[k[len(p_mem):]] = v
            elif k.startswith(p_emb):
                emb_sub[k[len(p_emb):]] = v

        if mem_sub:
            hd_core.memory.load_state_dict(mem_sub, strict=False)
            loaded = True
            logger.info(f'[MEM] loaded memory from model_state prefix={prefix} tensors={len(mem_sub)}')
        if emb_sub:
            try:
                hd_core.embedder.load_state_dict(emb_sub, strict=False)
                logger.info(f'[MEM] loaded embedder from model_state prefix={prefix} tensors={len(emb_sub)}')
            except Exception as e:
                logger.warning(f'[MEM] failed loading embedder from model_state prefix={prefix}: {repr(e)}')

        if loaded:
            break

    return loaded


def _log_hd_core_summary(hd_core, logger):
    mem = getattr(hd_core, 'memory', None)
    cfg_obj = getattr(hd_core, 'cfg', None)
    logger.info('[HD] summary begin')
    if cfg_obj is not None:
        for k in ('mode', 'lam', 'temperature', 'quantize', 'encode_chunk', 'logits_chunk',
                  'bg_enabled', 'bg_margin_scale', 'anchor_id_scale'):
            if hasattr(cfg_obj, k):
                logger.info(f'[HD][cfg] {k}={getattr(cfg_obj, k)}')
    if mem is not None:
        if hasattr(mem, 'classify_weights') and torch.is_tensor(mem.classify_weights):
            logger.info(f'[HD][mem] classify_weights={tuple(mem.classify_weights.shape)}')
        if hasattr(mem, 'prototypes') and torch.is_tensor(mem.prototypes):
            logger.info(f'[HD][mem] prototypes={tuple(mem.prototypes.shape)}')
        if hasattr(mem, 'bg_weight') and torch.is_tensor(mem.bg_weight):
            logger.info(f'[HD][mem] bg_weight={tuple(mem.bg_weight.shape)}')
        if hasattr(mem, 'bg_prototype') and torch.is_tensor(mem.bg_prototype):
            logger.info(f'[HD][mem] bg_prototype={tuple(mem.bg_prototype.shape)}')
    logger.info('[HD] summary end')


@torch.no_grad()
def _load_memory_payload_into_hd_core(hd_core, memory_path: str, logger):
    payload = torch.load(memory_path, map_location='cpu')

    if torch.is_tensor(payload):
        raise RuntimeError('Unsupported memory payload: bare tensor. Expected dict payload.')

    if not isinstance(payload, dict):
        raise RuntimeError(f'Unsupported memory payload type: {type(payload)}')

    loaded = False

    if 'memory' in payload:
        hd_core.memory.load_state_dict(payload['memory'], strict=False)
        loaded = True
        logger.info('[MEM] loaded hd_core.memory state_dict from payload[memory]')

    if 'embedder' in payload:
        try:
            hd_core.embedder.load_state_dict(payload['embedder'], strict=False)
            logger.info('[MEM] loaded hd_core.embedder state_dict from payload[embedder]')
        except Exception as e:
            logger.warning(f'[MEM] failed loading embedder: {repr(e)}')

    # Support simplified payload from memory builders
    if 'classify_weights' in payload and hasattr(hd_core.memory, 'classify_weights'):
        cw = payload['classify_weights']
        if not torch.is_tensor(cw):
            cw = torch.tensor(cw)
        hd_core.memory.classify_weights.copy_(cw.to(hd_core.memory.classify_weights.device).to(hd_core.memory.classify_weights.dtype))
        loaded = True
        logger.info('[MEM] loaded classify_weights from payload')

    if 'prototypes' in payload and payload['prototypes'] is not None and hasattr(hd_core.memory, 'prototypes'):
        p = payload['prototypes']
        if not torch.is_tensor(p):
            p = torch.tensor(p)
        hd_core.memory.prototypes.copy_(p.to(hd_core.memory.prototypes.device).to(hd_core.memory.prototypes.dtype))
        loaded = True
        logger.info('[MEM] loaded prototypes from payload')

    if 'bg_weight' in payload and hasattr(hd_core.memory, 'bg_weight'):
        bgw = payload['bg_weight']
        if not torch.is_tensor(bgw):
            bgw = torch.tensor(bgw)
        hd_core.memory.bg_weight.copy_(bgw.to(hd_core.memory.bg_weight.device).to(hd_core.memory.bg_weight.dtype))
        loaded = True
        logger.info('[MEM] loaded bg_weight from payload')

    if 'bg_prototype' in payload and payload['bg_prototype'] is not None and hasattr(hd_core.memory, 'bg_prototype'):
        bgp = payload['bg_prototype']
        if not torch.is_tensor(bgp):
            bgp = torch.tensor(bgp)
        hd_core.memory.bg_prototype.copy_(bgp.to(hd_core.memory.bg_prototype.device).to(hd_core.memory.bg_prototype.dtype))
        loaded = True
        logger.info('[MEM] loaded bg_prototype from payload')

    # Support normal detector checkpoints: payload['model_state'|'state_dict'...]
    if not loaded:
        for key in ('model_state', 'state_dict', 'model_state_dict', 'model'):
            if key in payload and isinstance(payload[key], dict):
                if _try_load_hd_from_model_state_dict(hd_core, payload[key], logger):
                    loaded = True
                    break

    # Support direct state_dict payload
    if not loaded and _try_load_hd_from_model_state_dict(hd_core, payload, logger):
        loaded = True

    if not loaded:
        raise RuntimeError(
            'No recognized memory fields found in payload. '
            'Expected one of: memory/embedder, classify_weights/prototypes, or model_state with dense_head.hd_core.*'
        )

    hd_core.memory.normalize_()



def _build_online_cfg_dict(args):
    keys = [
        'tau_prob', 'tau_margin', 'select_top_ratio', 'select_min_k', 'select_max_k',
        'teacher_momentum', 'update_every_n_steps', 'min_selected_anchors', 'min_confpass_anchors',
        'update_mode', 'alpha', 'source_pullback_lambda', 'max_per_class_per_step', 'min_per_class_for_update',
        'class_balance_min_classes_to_update', 'class_balance_max_pending_steps',
        'replay_cap_per_class', 'replay_per_class', 'replay_alpha_scale', 'eval_every_updates',
        'guard_max_drop', 'save_every_updates', 'log_interval_steps', 'feat_source', 'pseudo_logits_source',
        'zero_update_tau_prob_step', 'zero_update_tau_margin_step'
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
    if args.replay_enable is not None:
        out['replay_enable'] = bool(args.replay_enable)
    if args.guard_use_best is not None:
        out['guard_use_best'] = bool(args.guard_use_best)

    return out


def _load_online_cfg_file(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f'Online cfg not found: {path}')
    with open(cfg_path, 'r') as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RuntimeError(f'Online cfg must be a dict yaml, got {type(raw)}')
    if 'ONLINE' in raw:
        node = raw['ONLINE']
        if node is None:
            return {}
        if not isinstance(node, dict):
            raise RuntimeError(f'ONLINE section must be dict, got {type(node)}')
        return node
    return raw


def _save_run_artifacts(output_dir: Path, args, online_cfg: dict, logger):
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_online_cfg_path = output_dir / 'online_cfg_resolved.yaml'
    with open(resolved_online_cfg_path, 'w') as f:
        yaml.safe_dump(online_cfg, f, sort_keys=True)
    logger.info(f'[ARTIFACT] saved resolved online cfg: {resolved_online_cfg_path}')

    run_meta = {
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cfg_file': args.cfg_file,
        'ckpt': args.ckpt,
        'online_cfg_arg': args.online_cfg,
        'source_memory': args.source_memory,
        'resume_online_state': args.resume_online_state,
        'stream_split': args.stream_split,
        'val_split': args.val_split,
        'max_steps': args.max_steps,
        'batch_size': args.batch_size,
        'workers': args.workers,
        'use_amp': bool(args.use_amp),
        'guard_metric': args.guard_metric,
        'fast_recall_only': bool(args.fast_recall_only),
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



def main():
    args, _ = parse_config()
    output_dir, logger = _build_output_and_logger(args)

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    if args.batch_size <= 0:
        raise RuntimeError(f'Invalid batch_size={args.batch_size}, must be > 0')

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {args.ckpt}')
    if args.source_memory is not None and (not Path(args.source_memory).is_file()):
        raise FileNotFoundError(f'source_memory file not found: {args.source_memory}')
    if args.resume_online_state is not None and (not Path(args.resume_online_state).is_file()):
        raise FileNotFoundError(f'resume_online_state file not found: {args.resume_online_state}')

    # Stream loader
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

    # Val loader (optional)
    if args.val_split is not None:
        val_cfg = _override_split(cfg.DATA_CONFIG, args.val_split)
        logger.info(f"[DATA] val split={val_cfg.DATA_SPLIT.get('test', 'N/A')}")
        _val_set, val_loader, _ = build_dataloader(
            dataset_cfg=val_cfg,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=False,
            workers=args.workers,
            logger=logger,
            training=False,
        )
        logger.info(f'[DATA] val dataset size={_safe_len(_val_set)} loader_len={_safe_len(val_loader)}')
    else:
        val_loader = None
        logger.info('[DATA] val_split is None, guard evaluation disabled')

    # Build model
    logger.info('[MODEL] building detector...')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=stream_set)
    logger.info('[MODEL] loading checkpoint...')
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    logger.info('[MODEL] model moved to CUDA and set to eval')

    hd_core = _find_hd_core(model)
    if hd_core is None:
        raise RuntimeError('Cannot find dense_head.hd_core in model.')
    _log_hd_core_summary(hd_core, logger)

    if args.source_memory is not None:
        logger.info(f'[MEM] loading source memory: {args.source_memory}')
        _load_memory_payload_into_hd_core(hd_core, args.source_memory, logger)
        logger.info('[MEM] source memory loaded and normalized')

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
            batch_size=args.batch_size,
            time_log_every=0,
        )
        result_dir = output_dir / 'online_eval' / f'eval_{eval_counter["n"]:04d}'
        result_dir.mkdir(parents=True, exist_ok=True)
        try:
            tb_dict = eval_utils.eval_one_epoch(
                cfg,
                eval_args,
                eval_model,
                eval_loader,
                epoch_id=f'online_{eval_counter["n"]:04d}',
                logger=logger,
                dist_test=False,
                result_dir=result_dir,
            )
        except Exception as e:
            logger.exception(f'[EVAL] failed at {eval_name} eval #{eval_counter["n"]}: {repr(e)}')
            raise
        return tb_dict

    def eval_fn(eval_model, eval_loader):
        fast_guard = bool(args.guard_fast_recall_only)
        tb_dict = _run_eval(eval_model, eval_loader, fast_recall_only=fast_guard, eval_name='guard')
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

    # online cfg merge rule: file defaults -> CLI overrides
    online_cfg = {}
    if args.online_cfg is not None:
        online_cfg = _load_online_cfg_file(args.online_cfg)
        logger.info(f'[ONLINE_CFG] loaded from file: {args.online_cfg}')
        logger.info(f'[ONLINE_CFG] file keys: {sorted(list(online_cfg.keys()))}')

    cli_online_cfg = _build_online_cfg_dict(args)
    online_cfg.update(cli_online_cfg)
    logger.info(f'[ONLINE_CFG] final keys: {sorted(list(online_cfg.keys()))}')
    logger.info(f'[ONLINE_CFG] final values:\n{yaml.safe_dump(online_cfg, sort_keys=True)}')
    _save_run_artifacts(output_dir, args, online_cfg, logger)

    runner = OnlineHDRunner(
        model=model,
        stream_loader=stream_loader,
        logger=logger,
        cfg=online_cfg,
        val_loader=val_loader,
        eval_fn=eval_fn if val_loader is not None else None,
        output_dir=str(output_dir / 'online_state'),
        state_save_prefix='online_hd',
        use_amp=bool(args.use_amp),
    )

    if args.resume_online_state is not None:
        logger.info(f'[STATE] resume from: {args.resume_online_state}')
        runner.load_state(args.resume_online_state)
        logger.info('[STATE] resume done')

    logger.info('[ONLINE] runner start')
    runner.run(max_steps=args.max_steps)
    logger.info('[ONLINE] runner finished')

    if val_loader is not None and bool(args.final_full_eval):
        logger.info('[EVAL] start final full official evaluation once')
        tb_dict = _run_eval(model, val_loader, fast_recall_only=False, eval_name='final_full')
        if isinstance(tb_dict, dict):
            if args.guard_metric in tb_dict:
                logger.info(f'[EVAL][FINAL] {args.guard_metric}={float(tb_dict[args.guard_metric]):.6f}')
            for k in ('recall/rcnn_0.3', 'recall/rcnn_0.5', 'mAP', 'map'):
                if k in tb_dict:
                    logger.info(f'[EVAL][FINAL] {k}={float(tb_dict[k]):.6f}')
        logger.info('[EVAL] final full official evaluation finished')

    logger.info('**********************End online HD**********************')


if __name__ == '__main__':
    main()
