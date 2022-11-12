# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import os
import numpy as np
import math
import sys
from typing import Iterable, Optional

import torch
import torch.distributed as dist

from timm.data import Mixup
from timm.utils import accuracy

import utils.misc as misc
import utils.lr_sched as lr_sched
from torchvision import transforms
import cv2
from PIL import Image
import torch.nn as nn


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    num_frames = args.num_frames
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    if args.data_set == 'SSV2' or args.data_set == 'Kinetics':
        print_freq = 1000
    else:
        print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, targets = batch[0], batch[1]
        batch_size = samples.shape[0]
        in_channels = samples.shape[1]
        frame_size = samples.shape[3]
        block_width = 4
        num_temporal_frame = int(num_frames / (block_width ** 2))
        spatial_stride = int(num_frames / 8)
        resize = transforms.Resize((frame_size, frame_size))
        samples_t = samples
        samples_s = samples[:, :, 0::spatial_stride, :, :]

        samples_t = samples_t.reshape(batch_size, in_channels, num_temporal_frame, int(block_width ** 2), frame_size, frame_size)
        samples_t = samples_t.reshape(batch_size, in_channels, num_temporal_frame, block_width, block_width, frame_size, frame_size)
        samples_t = samples_t.permute(0, 1, 2, 3, 5, 4, 6)
        samples_t = samples_t.reshape(batch_size * in_channels * num_temporal_frame, block_width * frame_size, block_width * frame_size)
        samples_t = resize(samples_t).reshape(batch_size, in_channels, num_temporal_frame, frame_size, frame_size)

        samples = torch.cat([samples_t, samples_s], dim=2)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)


        if samples.shape[2] != 1:
            samples = samples.permute(0, 2, 1, 3, 4)    # B x Ts+Tt x C x W x H
            samples = samples.reshape(batch_size * (8 + num_temporal_frame), in_channels, frame_size, frame_size)

        elif samples.shape[2] == 1:
            samples = samples.squeeze()


        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            if hasattr(model, 'module'):
                output = model.module.encode_image(samples)
            else:
                output = model.encode_image(samples)
            output = output.squeeze()
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 100, header):
        images = batch[0]
        target = batch[1]  # TODO: check why default use -1
        images = images.squeeze()
        batch_size, in_channels, num_frames, frame_size, _ = images.shape
        block_width = 4
        num_temporal_frame = int(num_frames / (block_width ** 2))
        resize = transforms.Resize((frame_size, frame_size))

        spatial_stride = int(num_frames / 8)

        samples_t = images
        samples_s = images[:, :, 0::spatial_stride, :, :]

        samples_t = samples_t.reshape(batch_size, in_channels, num_temporal_frame, int(block_width ** 2), frame_size, frame_size)
        samples_t = samples_t.reshape(batch_size, in_channels, num_temporal_frame, block_width, block_width, frame_size, frame_size)
        samples_t = samples_t.permute(0, 1, 2, 3, 5, 4, 6)
        samples_t = samples_t.reshape(batch_size * in_channels * num_temporal_frame, block_width * frame_size, block_width * frame_size)
        samples_t = resize(samples_t).reshape(batch_size, in_channels, num_temporal_frame, frame_size, frame_size)

        images = torch.cat([samples_t, samples_s], dim=2)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if num_frames != 1:
            images = images.permute(0, 2, 1, 3, 4)
            images = images.reshape(batch_size * (8 + num_temporal_frame), in_channels, frame_size, frame_size)

        elif num_frames == 1:
            images = images.squeeze()


        # compute output
        with torch.cuda.amp.autocast():
            if hasattr(model, 'module'):
                output = model.module.encode_image(images)
            else:
                output = model.encode_image(images)
            output = output.squeeze()
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def final_test(data_loader, model, device, file, args):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Final_Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []

    for batch in metric_logger.log_every(data_loader, 100, header):
        images = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]

        batch_size = images.shape[0]
        num_frames = images.shape[2]
        image_channel = images.shape[1]
        image_size = images.shape[3]

        block_width = 4
        num_temporal_frame = int(num_frames / (block_width ** 2))
        resize = transforms.Resize((image_size, image_size))

        spatial_stride = int(num_frames / 8)

        samples_t = images
        samples_s = images[:, :, 0::spatial_stride, :, :]

        samples_t = samples_t.reshape(batch_size, image_channel, num_temporal_frame, int(block_width ** 2), image_size, image_size)
        samples_t = samples_t.reshape(batch_size, image_channel, num_temporal_frame, block_width, block_width, image_size, image_size)
        samples_t = samples_t.permute(0, 1, 2, 3, 5, 4, 6)
        samples_t = samples_t.reshape(batch_size * image_channel * num_temporal_frame, block_width * image_size, block_width * image_size)
        samples_t = resize(samples_t).reshape(batch_size, image_channel, num_temporal_frame, image_size, image_size)

        images = torch.cat([samples_t, samples_s], dim=2)

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if num_frames != 1:
            images = images.permute(0, 2, 1, 3, 4)
            images = images.reshape(batch_size * (8 + num_temporal_frame), image_channel, image_size, image_size)

        elif num_frames == 1:
            images = images.squeeze()

        # compute output
        with torch.cuda.amp.autocast():
            if hasattr(model, 'module'):
                output = model.module.encode_image(images)
            else:
                output = model.encode_image(images)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(
                ids[i], str(output.data[i].cpu().numpy().tolist()), str(int(target[i].cpu().numpy())),
                str(int(chunk_nb[i].cpu().numpy())), str(int(split_nb[i].cpu().numpy()))
            )
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks, is_hmdb=False):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video_hmdb if is_hmdb else compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100


def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]


def compute_video_hmdb(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    try:
        pred = np.argmax(feat)
        top1 = (int(pred) == int(label)) * 1.0
        top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    except:
        pred = 0
        top1 = 1.0
        top5 = 1.0
        label = 0
    return [pred, top1, top5, int(label)]
