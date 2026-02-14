import _init_path
import argparse
import datetime
import logging
import os
import sys
from pathlib import Path
from test import repeat_eval_ckpt

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from eval_utils import eval_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model, train_one_epoch, checkpoint_state, save_checkpoint, disable_augmentation_hook


def parse_config():
    parser = argparse.ArgumentParser(description='HD finetune training (two-stage)')
    parser.add_argument('--cfg_file', type=str, required=True, help='config for retrain')

    parser.add_argument('--batch_size', type=int, default=None, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs to train')
    parser.add_argument('--workers', type=int, default=8, help='num dataloader workers')
    parser.add_argument('--extra_tag', type=str, default='retrain_hd', help='output tag')

    parser.add_argument('--base_ckpt', type=str, default=None,
                        help='baseline checkpoint to initialize weights (loads model params only)')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='resume checkpoint for retrain run (loads model+optimizer)')

    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distributed training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='use sync BN')
    parser.add_argument('--fix_random_seed', action='store_true', default=False)
    parser.add_argument('--ckpt_save_interval', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--max_ckpt_save_num', type=int, default=30)
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epochs_to_eval', type=int, default=0)
    parser.add_argument('--save_to_file', action='store_true', default=False)
    parser.add_argument('--skip_eval', action='store_true', default=False,
                        help='skip post-train repeat_eval_ckpt')

    parser.add_argument('--auto_eval_each_epoch', action='store_true', default=False,
                        help='evaluate after each epoch and keep all checkpoints/results')
    parser.add_argument('--best_metric_key', type=str, default='recall/rcnn_0.3',
                        help='metric key to print in auto-eval logs')
    parser.add_argument('--eval_workers', type=int, default=1, help='workers for eval dataloader in auto-eval mode')
    parser.add_argument('--eval_batch_size', type=int, default=None, help='batch size for eval dataloader in auto-eval mode')
    parser.add_argument('--eval_hd_mode', type=str, default='hd_only', choices=['baseline', 'hd_only', 'fused'],
                        help='override HD mode during auto-eval only')
    parser.add_argument('--save_latest_ckpt', action='store_true', default=False,
                        help='if set, also save rolling latest_model.pth during epoch')

    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False)
    parser.add_argument('--logger_iter_interval', type=int, default=50)
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300)
    parser.add_argument('--wo_gpu_stat', action='store_true')
    parser.add_argument('--use_amp', action='store_true', help='use mixed precision training')

    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='freeze vfe/map_to_bev/backbone_2d')
    parser.add_argument('--no_freeze_backbone', action='store_false', dest='freeze_backbone',
                        help='disable backbone freezing')
    parser.add_argument('--freeze_modules', type=str, default='vfe,map_to_bev_module,backbone_2d',
                        help='comma-separated module names to freeze')
    parser.add_argument('--unfreeze_cls_pre', action='store_true', default=True,
                        help='unfreeze dense_head.conv_cls_pre for adaptation')
    parser.add_argument('--freeze_cls_pre', action='store_false', dest='unfreeze_cls_pre',
                        help='freeze dense_head.conv_cls_pre')
    parser.add_argument('--unfreeze_cls_out', action='store_true', default=True,
                        help='unfreeze dense_head.conv_cls_out for adaptation/teacher branch')
    parser.add_argument('--freeze_cls_out', action='store_false', dest='unfreeze_cls_out',
                        help='freeze dense_head.conv_cls_out')
    parser.add_argument('--train_hd_embedder', action='store_true', default=True,
                        help='set hd_core.embedder params requires_grad=True')
    parser.add_argument('--freeze_hd_embedder', action='store_false', dest='train_hd_embedder',
                        help='freeze hd_core.embedder')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])

    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def _set_requires_grad(module, flag: bool):
    if module is None:
        return 0
    n = 0
    for p in module.parameters():
        p.requires_grad = bool(flag)
        n += p.numel()
    return n


def _resolve_model(m):
    return m.module if hasattr(m, 'module') else m


def _freeze_for_hd_retrain(model, args, logger):
    net = _resolve_model(model)

    if not args.freeze_backbone:
        logger.info('[FREEZE] freeze_backbone=False: keep backbone trainable')
    else:
        for name in [x.strip() for x in args.freeze_modules.split(',') if x.strip()]:
            m = getattr(net, name, None)
            if m is None:
                logger.info(f'[FREEZE] skip missing module: {name}')
                continue
            n = _set_requires_grad(m, False)
            logger.info(f'[FREEZE] {name}: frozen params={n}')

    dh = getattr(net, 'dense_head', None)
    if dh is None:
        logger.warning('[FREEZE] model has no dense_head; skip HD-specific unfreeze')
        return

    if args.unfreeze_cls_pre and hasattr(dh, 'conv_cls_pre'):
        n = _set_requires_grad(dh.conv_cls_pre, True)
        logger.info(f'[UNFREEZE] dense_head.conv_cls_pre params={n}')

    if args.unfreeze_cls_out and hasattr(dh, 'conv_cls_out'):
        n = _set_requires_grad(dh.conv_cls_out, True)
        logger.info(f'[UNFREEZE] dense_head.conv_cls_out params={n}')

    hd_core = getattr(dh, 'hd_core', None)
    if hd_core is not None and hasattr(hd_core, 'embedder'):
        n = _set_requires_grad(hd_core.embedder, bool(args.train_hd_embedder))
        logger.info(f'[HD] embedder trainable={bool(args.train_hd_embedder)} params={n}')


def _log_trainable_params(model, logger):
    total = 0
    trainable = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    ratio = 100.0 * trainable / max(total, 1)
    logger.info(f'[PARAM] total={total}, trainable={trainable} ({ratio:.2f}%)')


def _auto_train_eval_loop(
    model, optimizer, train_loader, train_sampler, lr_scheduler, lr_warmup_scheduler,
    args, cfg, logger, output_dir, ckpt_dir, start_epoch, it, tb_log
):
    eval_bs = args.eval_batch_size if args.eval_batch_size is not None else args.batch_size
    _test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=eval_bs,
        dist=False,
        workers=args.eval_workers,
        logger=logger,
        training=False
    )
    logger.info(f'[AUTO-EVAL] enabled: metric(log only)={args.best_metric_key}, eval_bs={eval_bs}, eval_workers={args.eval_workers}')

    dataloader_iter = iter(train_loader)
    total_it_each_epoch = len(train_loader)
    hook_config = cfg.get('HOOK', None)
    augment_disable_flag = False
    accumulated_iter = it

    class _TbarStub:
        def __init__(self):
            self.format_dict = {'elapsed': 0}

        def format_interval(self, _sec):
            return '00:00'

    tbar_stub = _TbarStub()
    epoch_ckpt_time_interval = args.ckpt_save_time_interval if args.save_latest_ckpt else 10 ** 12

    for cur_epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(cur_epoch)

        if lr_warmup_scheduler is not None and cur_epoch < cfg.OPTIMIZATION.WARMUP_EPOCH:
            cur_scheduler = lr_warmup_scheduler
        else:
            cur_scheduler = lr_scheduler

        augment_disable_flag = disable_augmentation_hook(
            hook_config, dataloader_iter, args.epochs, cur_epoch, cfg, augment_disable_flag, logger
        )

        accumulated_iter = train_one_epoch(
            model, optimizer, train_loader, model_fn_decorator(),
            lr_scheduler=cur_scheduler,
            accumulated_iter=accumulated_iter, optim_cfg=cfg.OPTIMIZATION,
            rank=cfg.LOCAL_RANK, tbar=tbar_stub,
            tb_log=tb_log, leave_pbar=(cur_epoch + 1 == args.epochs),
            total_it_each_epoch=total_it_each_epoch, dataloader_iter=dataloader_iter,
            cur_epoch=cur_epoch, total_epochs=args.epochs,
            use_logger_to_record=not args.use_tqdm_to_record,
            logger=logger, logger_iter_interval=args.logger_iter_interval,
            ckpt_save_dir=ckpt_dir, ckpt_save_time_interval=epoch_ckpt_time_interval,
            show_gpu_stat=not args.wo_gpu_stat,
            use_amp=args.use_amp
        )

        trained_epoch = cur_epoch + 1
        if (trained_epoch % args.ckpt_save_interval) == 0 and cfg.LOCAL_RANK == 0:
            ckpt_name = ckpt_dir / f'checkpoint_epoch_{trained_epoch}'
            save_checkpoint(checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name)

        if cfg.LOCAL_RANK != 0:
            continue

        eval_epoch_dir = output_dir / 'eval' / f'epoch_{trained_epoch}' / cfg.DATA_CONFIG.DATA_SPLIT['test'] / 'default'
        eval_epoch_dir.mkdir(parents=True, exist_ok=True)
        eval_log_file = eval_epoch_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        eval_file_handler = logging.FileHandler(filename=eval_log_file)
        eval_file_handler.setLevel(logging.INFO if cfg.LOCAL_RANK == 0 else logging.ERROR)
        eval_file_handler.setFormatter(logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s'))
        logger.addHandler(eval_file_handler)

        eval_model = model.module if hasattr(model, 'module') else model
        dh = getattr(eval_model, 'dense_head', None)
        old_mode = None
        if dh is not None and hasattr(dh, 'hd_mode'):
            old_mode = dh.hd_mode
            dh.hd_mode = str(args.eval_hd_mode)
        try:
            tb_dict = eval_utils.eval_one_epoch(
                cfg, args, eval_model, test_loader, trained_epoch, logger, dist_test=False, result_dir=eval_epoch_dir
            )
        finally:
            if dh is not None and old_mode is not None:
                dh.hd_mode = old_mode
            logger.removeHandler(eval_file_handler)
            eval_file_handler.close()
        metric_val = tb_dict.get(args.best_metric_key, None)
        if metric_val is None:
            logger.warning(f"[AUTO-EVAL] metric key '{args.best_metric_key}' not found. Available keys: {list(tb_dict.keys())[:30]}")
        else:
            logger.info(f"[AUTO-EVAL] epoch={trained_epoch}, eval_hd_mode={args.eval_hd_mode}, {args.best_metric_key}={metric_val:.6f}")


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    # Mirror logs to terminal for interactive runs.
    if cfg.LOCAL_RANK == 0:
        has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        if not has_stream:
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.INFO)
            logger.addHandler(sh)
        logger.propagate = False

    logger.info('**********************Start HD retrain logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')

    for key, val in vars(args).items():
        logger.info('{:20} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    logger.info('----------- Create dataloader & network -----------')
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train,
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    start_epoch = it = 0
    last_epoch = -1

    if args.base_ckpt is not None:
        logger.info(f'[CKPT] initialize model from base_ckpt: {args.base_ckpt}')
        model.load_params_from_file(filename=args.base_ckpt, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        logger.info(f'[CKPT] resume retrain run from ckpt: {args.ckpt}')

    model.train()
    _freeze_for_hd_retrain(model, args, logger)
    _log_trainable_params(model, logger)

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1

    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    logger.info('**********************Start HD retrain %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    auto_eval_enabled = bool(args.auto_eval_each_epoch)
    if dist_train and auto_eval_enabled:
        logger.warning('[AUTO-EVAL] disabled because distributed training is enabled.')
        auto_eval_enabled = False

    if auto_eval_enabled:
        _auto_train_eval_loop(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            train_sampler=train_sampler,
            lr_scheduler=lr_scheduler,
            lr_warmup_scheduler=lr_warmup_scheduler,
            args=args,
            cfg=cfg,
            logger=logger,
            output_dir=output_dir,
            ckpt_dir=ckpt_dir,
            start_epoch=start_epoch,
            it=it,
            tb_log=tb_log
        )
    else:
        rolling_ckpt_time_interval = args.ckpt_save_time_interval if args.save_latest_ckpt else 10 ** 12
        train_model(
            model,
            optimizer,
            train_loader,
            model_func=model_fn_decorator(),
            lr_scheduler=lr_scheduler,
            optim_cfg=cfg.OPTIMIZATION,
            start_epoch=start_epoch,
            total_epochs=args.epochs,
            start_iter=it,
            rank=cfg.LOCAL_RANK,
            tb_log=tb_log,
            ckpt_save_dir=ckpt_dir,
            train_sampler=train_sampler,
            lr_warmup_scheduler=lr_warmup_scheduler,
            ckpt_save_interval=args.ckpt_save_interval,
            max_ckpt_save_num=args.max_ckpt_save_num,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            logger=logger,
            logger_iter_interval=args.logger_iter_interval,
            ckpt_save_time_interval=rolling_ckpt_time_interval,
            use_logger_to_record=not args.use_tqdm_to_record,
            show_gpu_stat=not args.wo_gpu_stat,
            use_amp=args.use_amp,
            cfg=cfg
        )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End HD retrain %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    if args.skip_eval or auto_eval_enabled:
        reason = 'auto_eval_each_epoch enabled' if auto_eval_enabled else 'skip_eval=True'
        logger.info(f'[EVAL] skip post-train repeat_eval_ckpt ({reason}).')
        return

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    _test_set, test_loader, _sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs - args.num_epochs_to_eval, 0)

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
