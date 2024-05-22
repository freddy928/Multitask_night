import argparse
import os, sys
import math
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import matplotlib.pyplot as plt
import pprint
import time
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter
from lib.utils import DataLoaderX, torch_distributed_zero_first

from lib.dataset.zurich_pair_dataset import zurich_pair_DataSet
import lib.dataset as dataset
from lib.config import cfg
from lib.config import cfgdan
from lib.config import update_config
from lib.core.loss import get_loss
from lib.core.loss_dan import L_TV, L_exp_z, SSIM, L_pair
from lib.core.function import train
from lib.core.function import validate
from lib.core.general import fitness
# from lib.models import get_net
from lib.models.MultiGate import get_net
from lib.models.discriminator import multiDiscriminator
from lib.models.relighting import relight
from lib.utils import is_parallel
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger, select_device
from lib.utils import run_anchor


def weightedMSE(D_out, D_label):
    return torch.mean((D_out - D_label).abs() ** 2)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Multitask network')
    # general
    # parser.add_argument('--cfg',
    #                     help='experiment configure file name',
    #                     required=True,
    #                     type=str)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='runs/')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    args = parser.parse_args()

    return args


def main():
    # set all the configurations
    args = parse_args()
    update_config(cfg, args)

    # Set DDP variables
    world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    rank = global_rank
    # print(rank)
    # TODO: handle distributed training logger
    # set the logger, tb_log_dir means tensorboard logdir

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, cfg.LOG_DIR, 'train', rank=rank)

    if rank in [-1, 0]:
        logger.info(pprint.pformat(args))
        logger.info(cfg)

        writer_dict = {
            'writer': SummaryWriter(log_dir=tb_log_dir),
            'train_global_steps': 0,
            'valid_global_steps': 0,
        }
    else:
        writer_dict = None

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # bulid up model
    # start_time = time.time()
    print("begin to bulid up model...")
    # DP mode
    device = select_device(logger, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS)) if not cfg.DEBUG \
        else select_device(logger, 'cpu')

    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend

    print("load model to device")
    discr1= multiDiscriminator().to(device)
    discr2= multiDiscriminator().to(device)
    lightnet= relight().to(device)
    model = get_net(cfg).to(device)

    # load model pth
    checkpoint = torch.load('weights/modelpth.pth')
    model.load_state_dict(checkpoint['state_dict'])

    # define loss function (criterion) and optimizer
    optimizer = get_optimizer(cfg, model)
    optimizer_D1 = optim.Adam(discr1.parameters(), lr=cfgdan.LRDISCR, betas=(0.9, 0.99))
    optimizer_D2 = optim.Adam(discr2.parameters(), lr=cfgdan.LRDISCR, betas=(0.9, 0.99))

    lf = lambda x: ((1 + math.cos(x * math.pi / cfg.TRAIN.END_EPOCH)) / 2) * \
                   (1 - cfg.TRAIN.LRF) + cfg.TRAIN.LRF  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(optimizer_D1, lr_lambda=lf)
    lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(optimizer_D2, lr_lambda=lf)

    criterion = get_loss(cfg, device=device)
    loss_pair = L_pair(cfg, device=device)
    loss_exp_z = L_exp_z(16).to(device)
    loss_TV = L_TV().to(device)
    loss_SSIM = SSIM().to(device)

    # assign model params
    model.gr = 1.0
    model.nc = 1
    # print('bulid model finished')

    print("begin to load data")

    '''data loading'''
    # 1.BDD100k multitask dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg=cfg,
        is_train=True,
        inputsize=cfg.MODEL.IMAGE_SIZE,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if rank != -1 else None

    train_loader = DataLoaderX(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=(cfg.TRAIN.SHUFFLE & rank == -1),
        num_workers=cfg.WORKERS,
        sampler=train_sampler,
        pin_memory=cfg.PIN_MEMORY,
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    num_batch = len(train_loader)

    if rank in [-1, 0]:
        valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
            cfg=cfg,
            is_train=False,
            inputsize=cfg.MODEL.IMAGE_SIZE,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        valid_loader = DataLoaderX(
            valid_dataset,
            batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
            shuffle=False,
            num_workers=cfg.WORKERS,
            pin_memory=cfg.PIN_MEMORY,
            collate_fn=dataset.AutoDriveDataset.collate_fn
        )
    # 2. daytime-night pair
    # cfgdan.NUM_STEPS = 40 // (cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS))
    cfgdan.NUM_STEPS = 70000 // (cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS))
    target_loader = data.DataLoader(zurich_pair_DataSet(cfgdan, cfgdan.DATA_DIRECTORY_TARGET, cfgdan.DATA_LIST_PATH_TARGET,
                                                       max_iters=cfgdan.NUM_STEPS * cfgdan.BATCH_SIZE, set='train'),
                                    batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS), shuffle=False,num_workers=cfg.WORKERS,
                                    pin_memory=cfg.PIN_MEMORY)
    print('load data finished')
    num_target= len(target_loader)

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH

    # load state dict
    if rank in [-1, 0]:
        checkpoint_file = os.path.join(
            os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET), 'checkpoint.pth'
        )
        if os.path.exists(cfg.MODEL.PRETRAINED):
            logger.info("=> loading model '{}'".format(cfg.MODEL.PRETRAINED))
            checkpoint = torch.load(cfg.MODEL.PRETRAINED)
            begin_epoch = checkpoint['epoch']
            # best_perf = checkpoint['perf']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            lightnet.load_state_dict(checkpoint['light_dict'])
            discr1.load_state_dict(checkpoint['discr1_dict'])
            discr2.load_state_dict(checkpoint['discr2_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer_D1.load_state_dict(checkpoint['optimizer_D1'])
            optimizer_D2.load_state_dict(checkpoint['optimizer_D2'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                cfg.MODEL.PRETRAINED, checkpoint['epoch']))
            # cfg.NEED_AUTOANCHOR = False     #disable autoanchor

        if os.path.exists(cfg.MODEL.PRETRAINED_DET):
            logger.info("=> loading model weight in det branch from '{}'".format(cfg.MODEL.PRETRAINED))
            det_idx_range = [str(i) for i in range(0, 25)]
            model_dict = model.state_dict()
            checkpoint_file = cfg.MODEL.PRETRAINED_DET
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            last_epoch = checkpoint['epoch']
            checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
            model_dict.update(checkpoint_dict)
            model.load_state_dict(model_dict)
            lightnet.load_state_dict(checkpoint['light_dict'])
            discr1.load_state_dict(checkpoint['discr1_dict'])
            discr2.load_state_dict(checkpoint['discr2_dict'])
            logger.info("=> loaded det branch checkpoint '{}' ".format(checkpoint_file))

        if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
            logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            begin_epoch = checkpoint['epoch']
            # best_perf = checkpoint['perf']
            last_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            lightnet.load_state_dict(checkpoint['light_dict'])
            discr1.load_state_dict(checkpoint['discr1_dict'])
            discr2.load_state_dict(checkpoint['discr2_dict'])
            # optimizer = get_optimizer(cfg, model)
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer_D1.load_state_dict(checkpoint['optimizer_D1'])
            optimizer_D2.load_state_dict(checkpoint['optimizer_D2'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                checkpoint_file, checkpoint['epoch']))
            # cfg.NEED_AUTOANCHOR = False     #disable autoanchor

    # training
    num_warmup = max(round(cfg.TRAIN.WARMUP_EPOCHS * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')

    print('=> start training...')
    source_label = 0
    target_label = 1

    for epoch in range(begin_epoch + 1, cfg.TRAIN.END_EPOCH + 1):
        if rank != -1:
            train_loader.sampler.set_epoch(epoch)
        # train for one epoch
        train(cfg, train_loader, enumerate(target_loader), model, lightnet, discr1, discr2, criterion, loss_exp_z, loss_TV, loss_SSIM, loss_pair,
              optimizer, optimizer_D1, optimizer_D2, scaler, epoch, num_batch, num_warmup, writer_dict, logger, device, rank)

        # optimizer step
        lr_scheduler.step()
        lr_scheduler_D1.step()
        lr_scheduler_D2.step()

        # evaluate on validation set
        if (epoch % cfg.TRAIN.VAL_FREQ == 0 or epoch == cfg.TRAIN.END_EPOCH) and rank in [-1, 0]:
            # print('validate')
            da_segment_results, ll_segment_results, detect_results, total_loss, maps, times = validate(
                epoch, cfg, valid_loader, valid_dataset, model, lightnet, criterion,
                final_output_dir, tb_log_dir, writer_dict, logger, device, rank)
            fi = fitness(np.array(detect_results).reshape(1, -1))

            msg = 'Epoch: [{0}]    Loss({loss:.3f})\n' \
                  'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                  'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
                  'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n' \
                  'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                epoch, loss=total_loss, da_seg_acc=da_segment_results[0], da_seg_iou=da_segment_results[1],
                da_seg_miou=da_segment_results[2],
                ll_seg_acc=ll_segment_results[0], ll_seg_iou=ll_segment_results[1], ll_seg_miou=ll_segment_results[2],
                p=detect_results[0], r=detect_results[1], map50=detect_results[2], map=detect_results[3],
                t_inf=times[0], t_nms=times[1])
            logger.info(msg)

        # save checkpoint model and best model
        if rank in [-1, 0]:
            savepath = os.path.join(final_output_dir, f'epoch-{epoch}.pth')
            logger.info('=> saving checkpoint to {}'.format(savepath))
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                light=lightnet,
                discr1=discr1,
                discr2=discr2,
                # 'best_state_dict': model.module.state_dict(),
                # 'perf': perf_indicator,
                optimizer=optimizer,
                optimizer_D1=optimizer_D1,
                optimizer_D2=optimizer_D2,
                output_dir=final_output_dir,
                filename=f'epoch-{epoch}.pth'
            )
            save_checkpoint(
                epoch=epoch,
                name=cfg.MODEL.NAME,
                model=model,
                light=lightnet,
                discr1=discr1,
                discr2=discr2,
                # 'best_state_dict': model.module.state_dict(),
                # 'perf': perf_indicator,
                optimizer=optimizer,
                optimizer_D1=optimizer_D1,
                optimizer_D2=optimizer_D2,
                output_dir=os.path.join(cfg.LOG_DIR, cfg.DATASET.DATASET),
                filename='checkpoint.pth'
            )

if __name__ == '__main__':
    main()
