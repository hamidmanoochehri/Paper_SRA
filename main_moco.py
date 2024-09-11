#!/usr/bin/env python
# original code obtained from https://github.com/facebookresearch/moco-v3
# Modified by Authors of SRA (paper link) 
# ------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# ------------------------------------------------------------------------------
# Modified by Authors of SRA (paper link)

from torchvision.transforms import ToTensor
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
from torch.utils.tensorboard import SummaryWriter

import moco.builder as mbuilder
import moco.loader
import moco.optimizer

import vits

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import defaultdict
import re

import importlib
importlib.reload(mbuilder)
import inspect
#print(inspect.getsource(mbuilder.MoCo_ResNet))

import numpy as np
import os
import torch
import pandas as pd

from sra_v4 import rgb_he_wrgb
from code_snapshotting import copy_files_with_timestamp
from datetime import datetime
from torchvision.datasets import ImageFolder
import csv
from torch.utils.data import Subset

t0                     = time.perf_counter()
runs_root              = './runs/'
adaptive_matrix_file   = './tcgakirc_adaptive_params.txt' # options: tcgakirc_adaptive_params.txt, ukidney_adaptive_params.txt
code_snapshot_files    = ['main_moco.py', 'train_mocov3.sh', 'vits.py', 'sra_*.py', 'code_snapshotting.py', 'moco/*.py']
training_tile_size     = 256

start_time             = datetime.now()
run_timestamp          = start_time.strftime("%Y%m%d_%H%M%S") # unique run number for convenient saving of results and code snapshots

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(torchvision_models.__dict__[name]))
model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')
parser.add_argument('--mode', metavar='MODE',
                    help='run mode: train, test')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('--rgb-he-wrgb-dist-Hmax', default='uniform', type=str,
                    choices=['uniform', 'binomial', 'trinomial', 'normal', 'beta'],
                    help='the random dist for pathology based augmentation H image')
parser.add_argument('--rgb-he-wrgb-params-Hmax', nargs='+', default=[0.0, 1.0, 0.0, 0.0], type=float,
                    metavar='my_aug_pa1', help='parameters for pathology augmentation')
parser.add_argument('--rgb-he-wrgb-dist-Emax', default='uniform', type=str,
                    choices=['uniform', 'binomial', 'trinomial', 'normal', 'beta'],
                    help='the random dist for pathology based augmentation E image')
parser.add_argument('--rgb-he-wrgb-params-Emax', nargs='+', default=[0.0, 1.0, 0.0, 0.0], type=float,
                    metavar='my_aug_pa2', help='parameters for pathology augmentation E image')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--rgb-he-wrgb-randomapply-p', nargs=2, default=[0.4, 6.0], type=float,
                    metavar='my_aug_a', help='probabilities for RandomApply of the "rgb_he_wrg" augmentation to representation 1 (list[0]) and representation 2 (list[1]) of the image. 0: never apply, 1: apply to all dataset', dest='rgb_he_wrgb_randomapply_p')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                    metavar='W', help='weight decay (default: 1e-6)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=256, type=int,
                    help='feature dimension (default: 256)')
parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                    help='hidden dimension in MLPs (default: 4096)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating momentum encoder (default: 0.99)')
parser.add_argument('--moco-m-cos', action='store_true',
                    help='gradually increase moco momentum to 1 with a '
                         'half-cycle cosine schedule')
parser.add_argument('--moco-t', default=1.0, type=float,
                    help='softmax temperature (default: 1.0)')

# vit specific configs:
parser.add_argument('--stop-grad-conv1', action='store_true',
                    help='stop-grad after first conv, or patch embedding')

# other upgrades
parser.add_argument('--optimizer', default='lars', type=str,
                    choices=['lars', 'adamw'],
                    help='optimizer used (default: lars)')
parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--crop-min', default=0.08, type=float,
                    help='minimum scale for random cropping (default: 0.08)')

# added
parser.add_argument('--batches-per-epoch', default=None, type=int, metavar='N', help='number of batches per epoch (default: None, which means use all batches)')

def main():
    global run_timestamp
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

# ------------------------- Subset for Subset Training -------------------------
def get_subset(dataset, num_batches, batch_size):
    # Calculate the total number of samples to use
    num_samples = num_batches * batch_size
    # Randomly select indices for the subset
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    return Subset(dataset, indices)

# ------------------- aux fn to get WSI name from tile name --------------------
def get_WSI_name_from_tile_name(tile_path):
    base_name = os.path.basename(tile_path)  # Extracts the base name from the path
    name_split = base_name.split('_')

    # Check if 'IMG' is part of the second split element to decide the naming convention
    if 'IMG' in name_split[1]:
        wsi_name_parts = name_split[2:4]  # Get the third and fourth parts for the WSI name
    else:
        wsi_name_parts = name_split[1:3]  # Get the second and third parts for the WSI name

    # Join the necessary parts and return the WSI name
    # If any part contains 'polygon' or a similar suffix, exclude it
    WSI_name = '_'.join(part for part in wsi_name_parts if 'polygon' not in part and not part.isdigit())
    return WSI_name

# ---------------- to load adaptive matrix params from txt file ----------------
def load_adaptive_parameters(file_path):
    adaptive_params = {}
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=';')
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) < 51:
                continue
            wsi_name = row[3]
            RGBabsorp2HERes_Matrix = np.array([[float(row[36]), float(row[37]), float(row[38])],
                                                 [float(row[39]), float(row[40]), float(row[41])],
                                                 [float(row[42]), float(row[43]), float(row[44])]])
            HERes2RGBabsorp_Matrix = np.linalg.inv(RGBabsorp2HERes_Matrix)
            adaptive_params[wsi_name] = {
                'background_RGB': np.array([float(row[4]), float(row[5]), float(row[6])]),
                'RGB_absorption_to_H': np.array([float(row[36]), float(row[37]), float(row[38])]),
                'RGB_absorption_to_E': np.array([float(row[39]), float(row[40]), float(row[41])]),
                'RGB_absorption_to_Res': np.array([float(row[42]), float(row[43]), float(row[44])]),
                'Mat_HERes2RGBabsorp': HERes2RGBabsorp_Matrix,
                'maxH': float(row[45]),
                'maxE': float(row[46])
            }
    return adaptive_params

# ------------------------- Custom ImageFolder class ---------------------------
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, adaptive_params=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.adaptive_params = adaptive_params

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        # Extract WSI name from path
        wsi_name = get_WSI_name_from_tile_name(path)
        current_adaptive_params = self.adaptive_params.get(wsi_name, None)
        #print(f"CustomImageFolder __getitem__: adaptive_params for {wsi_name}: {current_adaptive_params}")

        # Apply the transforms
        if self.transform:
            sample = self.transform(sample, current_adaptive_params)

        return sample, target
# --------------------- (end) Custom ImageFolder subclass ----------------------

def main_worker(gpu, ngpus_per_node, args):
    global run_timestamp

    args.gpu = gpu
    #main_mocoo
    # suppress printing if not first GPU on each node
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # Code snapshotting only on rank 0
    if args.mode == 'train' and (args.rank == 0 or not args.distributed):
        print(f'RUN TIMESTAMP is: {run_timestamp}')
        copy_files_with_timestamp(code_snapshot_files, save_root=os.path.join(runs_root, run_timestamp), run_timestamp=run_timestamp)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('vit'):
        model = mbuilder.MoCo_ViT(
            partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
            args.moco_dim, args.moco_mlp_dim, args.moco_t)
    else:
        model = mbuilder.MoCo_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True), 
            args.moco_dim, args.moco_mlp_dim, args.moco_t)

    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size / 256

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#, find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)#, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    #print(model) # print model after SyncBatchNorm

    if args.optimizer == 'lars':
        optimizer = moco.optimizer.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
        
    scaler = torch.cuda.amp.GradScaler()
    summary_writer = SummaryWriter(os.path.join(runs_root, run_timestamp)) if args.rank == 0 else None

    # optionally resume from a checkpoint
    if args.resume and args.mode=='train':
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scaler.load_state_dict(checkpoint['scaler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    # ---------------------- Defining Data Augmentations -----------------------
    # follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
    # Added Stain Reconstruction Augmentation (SRA): ToDo: SRA paper link
    # augmentation1 is for the first representation of the image and augmentation2 is for the seceond representation of the image

    # normalization transform
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                     std  = [0.229, 0.224, 0.225])
    
    #These are moco v3's orig augs with PSA added as the first try of PSA. These got great results on UKidney but Bad results on TCGA
    augmentation1 = moco.loader.CustomCompose([
        moco.loader.TransformWrapper(transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.))),
        moco.loader.TransformWrapper(rgb_he_wrgb(Hmax_dist_type=args.rgb_he_wrgb_dist_Hmax, Hmax_dist_params=args.rgb_he_wrgb_params_Hmax, Emax_dist_type=args.rgb_he_wrgb_dist_Emax, Emax_dist_params=args.rgb_he_wrgb_params_Emax), use_adaptive_params=True),
        moco.loader.TransformWrapper(transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8)), 
        moco.loader.TransformWrapper(transforms.RandomGrayscale(p=0.2)),
        moco.loader.TransformWrapper(transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=1.0)),
        moco.loader.TransformWrapper(transforms.RandomHorizontalFlip()),
        moco.loader.TransformWrapper(transforms.ToTensor()),
        moco.loader.TransformWrapper(normalize)
    ])

    augmentation2 = moco.loader.CustomCompose([
        moco.loader.TransformWrapper(transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.))),
        moco.loader.TransformWrapper(rgb_he_wrgb(Hmax_dist_type=args.rgb_he_wrgb_dist_Hmax, Hmax_dist_params=args.rgb_he_wrgb_params_Hmax, Emax_dist_type=args.rgb_he_wrgb_dist_Emax, Emax_dist_params=args.rgb_he_wrgb_params_Emax), use_adaptive_params=True),
        moco.loader.TransformWrapper(transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8)),
        moco.loader.TransformWrapper(transforms.RandomGrayscale(p=0.2)),
        moco.loader.TransformWrapper(transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.1)),
        moco.loader.TransformWrapper(transforms.RandomApply([moco.loader.Solarize()], p=0.2)),
        moco.loader.TransformWrapper(transforms.RandomHorizontalFlip()),
        moco.loader.TransformWrapper(transforms.ToTensor()),
        moco.loader.TransformWrapper(normalize)
    ])

    # ------------------------------ Data loading ------------------------------
    # Load adaptive matrix parameters (load txt file only once)
    adaptive_params = load_adaptive_parameters(adaptive_matrix_file)
    traindir = os.path.join(args.data, 'train')

    t1 = time.perf_counter()
    print(f'Time from start until loading the dataset: {t1-t0} s')

    # Data loading with custom augmentation
    train_dataset = CustomImageFolder(
        traindir,
        moco.loader.TwoCropsTransform(
            augmentation1, 
            augmentation2
            ),
        adaptive_params=adaptive_params
        )
    
    t2 = time.perf_counter()
    print(f'Time for loading the dataset: {t2-t1} s')

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        print(f'*** train sampler is set to DistributedSampler since args.distributed is {args.distributed}')

    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
                                               train_dataset,
                                               batch_size  = args.batch_size,
                                               shuffle     = (train_sampler is None),
                                               num_workers = args.workers,
                                               pin_memory  = True,
                                               sampler     = train_sampler,
                                               drop_last   = True
                                              )

    # ---------------------------------------------------- Val Prep ----------------------------------------------------added
    valdir = os.path.join(args.data, 'val')  # Path to validation data
    val_dataset = CustomImageFolder(
                                     valdir,
                                     transform=moco.loader.TwoCropsTransform(
                                         augmentation1,
                                         augmentation2
                                         ),
                                     adaptive_params=adaptive_params
                                    )
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False) if args.distributed else None

    val_loader = torch.utils.data.DataLoader(
                                             val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=(val_sampler is None),
                                             num_workers=args.workers,
                                             pin_memory=True,
                                             sampler=val_sampler
                                            )

    # --------------------------------------------------- TRAIN LOOP ---------------------------------------------------
    if args.mode=='train':
        for epoch in range(args.start_epoch, args.epochs):
            t4 = time.perf_counter()

            if args.distributed:
                train_sampler.set_epoch(epoch)

            if args.batches_per_epoch: # for having fixed iterations per epoch
                subset_dataset = get_subset(train_dataset, args.batches_per_epoch, args.batch_size)
                subset_loader = DataLoader(subset_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
                train(subset_loader, model, optimizer, scaler, summary_writer, epoch, args, batches_per_epoch=args.batches_per_epoch)
            else:
                train(train_loader, model, optimizer, scaler, summary_writer, epoch, args, batches_per_epoch=args.batches_per_epoch)\
            
            # Perform validation at the end of each epoch
            print('Validating')
            val_loss = validate(val_loader, model, args)
            if args.rank == 0:
                print(f"  Epoch: {epoch}, Validation Loss: {val_loss}")
                # Optionally log to TensorBoard
                summary_writer.add_scalar("val_loss", val_loss, epoch)

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                    and args.rank == 0): # only the first GPU saves checkpoint
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                }, is_best=False, args=args, filename=os.path.join(runs_root, run_timestamp, 'saved_checkpoints', f'checkpoint_{epoch:04d}.pth.tar'))

            # Ensure all processes complete the epoch, especially in DDP
            if args.distributed:
                dist.barrier()

            t5 = time.perf_counter()
            print(f'TRAINING LOOP: Time for running train and val: {t5-t4} s')

    if args.rank == 0:
        summary_writer.close()

def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args, batches_per_epoch=None):
    tr_t0 = time.perf_counter()
    
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    learning_rates = AverageMeter('LR', ':.4e')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    
    tr_t1 = time.perf_counter()
    print(f'   ITERATIONS LOOP: Time for getting to the start of the iterations loop inside train fn: {tr_t0-tr_t0} s')    
    
    train_total_loss = 0
    for i, (images, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(True):
            loss = model(images[0], images[1], moco_m)

        losses.update(loss.item(), images[0].size(0))
        train_total_loss += loss.item()
        if args.rank == 0:
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    train_total_loss /= len(train_loader)
    if args.rank == 0:
        summary_writer.add_scalar("train loss", train_total_loss, epoch)

# ------------------------------------------------------- val fn ------------------------------------------------------- added
def validate(val_loader, model, args):
    model.eval() 
    total_loss = 0.0
    with torch.no_grad(): 
        for i, (images, _) in enumerate(val_loader):
            if args.gpu is not None:
                image1 = images[0].cuda(args.gpu, non_blocking=True)
                image2 = images[1].cuda(args.gpu, non_blocking=True)
            
            # MoCo momentum update parameter 'm' can be fixed during validation
            m = args.moco_m
            
            # Compute loss for a given pair of images
            loss = model(image1, image2, m)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    global run_timestamp
    if args.rank==0:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    print(f'RUN TIMESTAMP IS: {run_timestamp}')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(runs_root, run_timestamp, 'saved_checkpoints', 'model_best.pth.tar'))
# ----------------

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m

if __name__ == '__main__':
    main()