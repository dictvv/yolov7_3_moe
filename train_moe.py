# train_moe.py
"""
MoE-YOLOv7 訓練腳本

基於原版 train.py，修改：
1. 使用 MoEYOLOv7 模型
2. 使用 ComputeLossMoE 計算 loss
3. 添加 MoE 特有參數
4. 每 N 個 epoch 驗證 mAP
"""

import argparse
import logging
import math
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.moe_yolo import MoEYOLOv7
from models.wbf import weighted_boxes_fusion
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr, xywh2xyxy, box_iou, \
    non_max_suppression, scale_coords, xyxy2xywh, clip_coords
from utils.google_utils import attempt_download
from utils.loss_moe import ComputeLossMoE
from utils.metrics import ap_per_class
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

logger = logging.getLogger(__name__)


def validate_moe(model, dataloader, device, nc, iou_thres=0.6, conf_thres=0.001, max_det=300):
    """
    驗證 MoE 模型的 mAP

    Args:
        model: MoEYOLOv7 模型
        dataloader: 驗證資料集
        device: 計算裝置
        nc: 類別數量
        iou_thres: NMS IoU 閾值
        conf_thres: 信心閾值
        max_det: 最大偵測數量

    Returns:
        mp: mean precision
        mr: mean recall
        map50: mAP@0.5
        map: mAP@0.5:0.95
    """
    model.eval()

    seen = 0
    stats = []

    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc='Validating')):
        imgs = imgs.to(device, non_blocking=True).float() / 255.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape

        with torch.no_grad():
            # MoE forward
            moe_output = model(imgs)

            # 合併專家輸出
            preds = merge_expert_outputs(moe_output, conf_thres)

        # 對每張圖片計算統計
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # 裁剪到圖片範圍
            clip_coords(pred, (height, width))

            # 轉換為 xyxy 格式（如果需要）
            predn = pred.clone()

            # 評估
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                correct = process_batch(predn, labelsn, iou_thres)
            else:
                correct = torch.zeros(pred.shape[0], 10, dtype=torch.bool)

            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # 計算 mAP
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0

    model.train()
    return mp, mr, map50, map


def merge_expert_outputs(moe_output, conf_thres=0.001):
    """
    合併多專家輸出為標準預測格式

    Args:
        moe_output: MoEYOLOv7 的輸出
        conf_thres: 信心閾值

    Returns:
        list of predictions, 每個元素是 [N, 6] (x1, y1, x2, y2, conf, cls)
    """
    expert_outputs = moe_output['expert_outputs']
    top_indices = moe_output['top_indices']
    top_weights = moe_output['top_weights']

    batch_size = top_indices.shape[0]
    results = []

    for b in range(batch_size):
        all_preds = []

        for k in range(top_indices.shape[1]):
            expert_idx = top_indices[b, k].item()
            weight = top_weights[b, k].item()

            if expert_idx in expert_outputs:
                pred = expert_outputs[expert_idx]

                # 處理 Detect 層輸出格式
                if isinstance(pred, tuple):
                    pred = pred[0]  # 推論輸出

                # pred: [B, num_anchors, 5+nc]
                pred_b = pred[b]  # [num_anchors, 5+nc]

                # 過濾低信心
                conf = pred_b[:, 4]
                mask = conf > conf_thres
                pred_b = pred_b[mask]

                if len(pred_b) > 0:
                    # 調整信心分數
                    pred_b[:, 4] *= weight

                    # 轉換格式：xywh -> xyxy
                    box = pred_b[:, :4].clone()
                    box_xyxy = torch.zeros_like(box)
                    box_xyxy[:, 0] = box[:, 0] - box[:, 2] / 2  # x1
                    box_xyxy[:, 1] = box[:, 1] - box[:, 3] / 2  # y1
                    box_xyxy[:, 2] = box[:, 0] + box[:, 2] / 2  # x2
                    box_xyxy[:, 3] = box[:, 1] + box[:, 3] / 2  # y2

                    # 取得類別
                    cls_conf, cls_pred = pred_b[:, 5:].max(1)
                    conf = pred_b[:, 4] * cls_conf

                    # 組合結果 [x1, y1, x2, y2, conf, cls]
                    det = torch.cat([box_xyxy, conf.unsqueeze(1), cls_pred.float().unsqueeze(1)], 1)
                    all_preds.append(det)

        if all_preds:
            all_preds = torch.cat(all_preds, 0)
            # NMS
            all_preds = simple_nms(all_preds, iou_thres=0.6)
            results.append(all_preds)
        else:
            results.append(torch.zeros(0, 6, device=top_indices.device))

    return results


def simple_nms(dets, iou_thres=0.6):
    """簡單的 NMS"""
    if len(dets) == 0:
        return dets

    # 按信心排序
    scores = dets[:, 4]
    order = scores.argsort(descending=True)
    dets = dets[order]

    keep = []
    while len(dets) > 0:
        keep.append(dets[0])
        if len(dets) == 1:
            break

        # 計算 IoU
        ious = box_iou_batch(dets[0:1, :4], dets[1:, :4]).squeeze(0)

        # 保留 IoU 小於閾值的
        mask = ious < iou_thres
        dets = dets[1:][mask]

    return torch.stack(keep) if keep else torch.zeros(0, 6, device=dets.device)


def box_iou_batch(box1, box2):
    """計算 IoU"""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    inter_x1 = torch.max(box1[:, 0].unsqueeze(1), box2[:, 0].unsqueeze(0))
    inter_y1 = torch.max(box1[:, 1].unsqueeze(1), box2[:, 1].unsqueeze(0))
    inter_x2 = torch.min(box1[:, 2].unsqueeze(1), box2[:, 2].unsqueeze(0))
    inter_y2 = torch.min(box1[:, 3].unsqueeze(1), box2[:, 3].unsqueeze(0))

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area

    return inter_area / (union_area + 1e-6)


def process_batch(detections, labels, iou_thres=0.5):
    """
    計算正確偵測

    Args:
        detections: [N, 6] (x1, y1, x2, y2, conf, cls)
        labels: [M, 5] (cls, x1, y1, x2, y2)
        iou_thres: IoU 閾值

    Returns:
        correct: [N, 10] 布林矩陣（10 個 IoU 閾值）
    """
    iou_thresholds = torch.linspace(0.5, 0.95, 10)
    correct = torch.zeros(detections.shape[0], 10, dtype=torch.bool, device=detections.device)

    if labels.shape[0] == 0:
        return correct

    iou = box_iou_batch(labels[:, 1:], detections[:, :4])

    for i, iou_t in enumerate(iou_thresholds):
        matches = (iou >= iou_t) & (labels[:, 0:1] == detections[:, 5])
        if matches.any():
            # 找到最佳匹配
            matches_sum = matches.sum(0)
            correct[:, i] = matches_sum > 0

    return correct


def train(hyp, opt, device, tb_writer=None):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs, batch_size, total_batch_size, weights, rank = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.global_rank

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    plots = not opt.evolve
    cuda = device.type != 'cpu'
    init_seeds(2 + rank)
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)
    is_coco = opt.data.endswith('coco.yaml')

    # Logging
    loggers = {'wandb': None}
    if rank in [-1, 0]:
        opt.hyp = hyp
        run_id = torch.load(weights, map_location=device, weights_only=False).get('wandb_id') if weights.endswith('.pt') and os.path.isfile(weights) else None
        wandb_logger = WandbLogger(opt, Path(opt.save_dir).stem, run_id, data_dict)
        loggers['wandb'] = wandb_logger.wandb
        data_dict = wandb_logger.data_dict
        if wandb_logger.wandb:
            weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

    nc = 1 if opt.single_cls else int(data_dict['nc'])
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)

    # ========== MoE Model ==========
    logger.info(colorstr('MoE Config: ') + f'num_experts={opt.num_experts}, top_k={opt.top_k}')

    model = MoEYOLOv7(
        cfg=opt.cfg,
        num_experts=opt.num_experts,
        top_k=opt.top_k,
        nc=nc,
        ch=3
    ).to(device)

    # 載入預訓練權重
    if weights.endswith('.pt') and os.path.isfile(weights):
        model.load_pretrained(weights, verbose=True)
        logger.info(f'Loaded pretrained weights from {weights}')

    with torch_distributed_zero_first(rank):
        check_dataset(data_dict)
    train_path = data_dict['train']
    test_path = data_dict['val']

    # Optimizer
    nbs = 64
    accumulate = max(round(nbs / total_batch_size), 1)
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)

    if opt.adam:
        optimizer = optim.Adam(pg0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # Scheduler
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    ema = ModelEMA(model) if rank in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0.0

    # Image sizes
    gs = max(int(model.stride.max()), 32)
    nl = 3
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]

    # DP mode
    if cuda and rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    if opt.sync_bn and cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logger.info('Using SyncBatchNorm()')

    # Trainloader
    dataloader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()
    nb = len(dataloader)
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Process 0
    if rank in [-1, 0]:
        testloader = create_dataloader(test_path, imgsz_test, batch_size * 2, gs, opt,
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])
            if plots:
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)
            model.half().float()

    # DDP mode
    if cuda and rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank,
                    find_unused_parameters=True)

    # Model parameters
    hyp['box'] *= 3. / nl
    hyp['cls'] *= nc / 80. * 3. / nl
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc
    model.hyp = hyp
    model.gr = 1.0
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    # ========== MoE Loss ==========
    compute_loss = ComputeLossMoE(model, aux_loss_weight=opt.aux_loss_weight)

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = start_epoch - 1
    scaler = amp.GradScaler(enabled=cuda)

    # 驗證間隔
    val_interval = opt.val_interval

    logger.info(f'Image sizes {imgsz} train, {imgsz_test} test\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Validating every {val_interval} epochs\n'
                f'Starting MoE training for {epochs} epochs...')

    # 記錄表頭
    with open(results_file, 'a') as f:
        f.write('%10s' * 12 % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total',
                               'labels', 'img_sz', 'P', 'R', 'mAP50', 'mAP') + '\n')

    for epoch in range(start_epoch, epochs):
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            if rank in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)
            if rank != -1:
                indices = (torch.tensor(dataset.indices) if rank == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if rank != 0:
                    dataset.indices = indices.cpu().numpy()

        mloss = torch.zeros(4, device=device)
        if rank != -1:
            dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size'))
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=nb)
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / total_batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward (MoE 返回 dict)
            with amp.autocast(enabled=cuda):
                moe_output = model(imgs)
                loss, loss_items = compute_loss(moe_output, targets.to(device))
                if rank != -1:
                    loss *= opt.world_size
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            # Print
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
                s = ('%10s' * 2 + '%10.4g' * 6) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1])
                pbar.set_description(s)

                # Plot
                if plots and ni < 3:
                    f = save_dir / f'train_batch{ni}.jpg'
                    Thread(target=plot_images, args=(imgs, targets, paths, f), daemon=True).start()

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]
        scheduler.step()

        # DDP process 0 or single-GPU
        if rank in [-1, 0]:
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs

            # ========== 每 N 個 epoch 驗證 mAP ==========
            mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0
            if (epoch + 1) % val_interval == 0 or final_epoch:
                logger.info('\nValidating...')
                mp, mr, map50, map = validate_moe(
                    ema.ema, testloader, device, nc,
                    iou_thres=0.6, conf_thres=0.001
                )
                logger.info(f'Validation: P={mp:.3f}, R={mr:.3f}, mAP@0.5={map50:.3f}, mAP@0.5:0.95={map:.3f}')

            # Write results
            with open(results_file, 'a') as f:
                f.write(('%10s' * 2 + '%10.4g' * 10) % (
                    '%g/%g' % (epoch, epochs - 1), mem, *mloss, targets.shape[0], imgs.shape[-1],
                    mp, mr, map50, map) + '\n')

            # Log
            tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'train/total_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95',
                    'x/lr0', 'x/lr1', 'x/lr2']
            for x, tag in zip(list(mloss) + [mp, mr, map50, map] + lr, tags):
                if tb_writer:
                    tb_writer.add_scalar(tag, x, epoch)
                if wandb_logger.wandb:
                    wandb_logger.log({tag: x})

            # Update best (使用 mAP@0.5 作為 fitness)
            fi = map50
            if fi > best_fitness:
                best_fitness = fi
            wandb_logger.end_epoch(best_result=best_fitness == fi)

            # Save model
            if (not opt.nosave) or final_epoch:
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(model.module if is_parallel(model) else model).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'moe_config': {
                            'num_experts': opt.num_experts,
                            'top_k': opt.top_k,
                            'aux_loss_weight': opt.aux_loss_weight
                        }}

                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if ((epoch + 1) % 25) == 0:
                    torch.save(ckpt, wdir / 'epoch_{:03d}.pt'.format(epoch))
                del ckpt

    # End training
    if rank in [-1, 0]:
        if plots:
            plot_results(save_dir=save_dir)
        logger.info('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))

    wandb_logger.finish_run()
    torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 原版參數
    parser.add_argument('--weights', type=str, default='yolov7-tiny.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='cfg/training/yolov7-tiny.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.tiny.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320, 320], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='moe_exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')

    # ========== MoE 專用參數 ==========
    parser.add_argument('--num-experts', type=int, default=4, help='number of experts')
    parser.add_argument('--top-k', type=int, default=2, help='number of experts to select per image')
    parser.add_argument('--aux-loss-weight', type=float, default=0.01, help='load balancing loss weight')
    parser.add_argument('--val-interval', type=int, default=3, help='validation interval (epochs)')

    opt = parser.parse_args()

    # Set DDP variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(opt.global_rank)

    # Check files
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)

    # DDP mode
    opt.total_batch_size = opt.batch_size
    device = select_device(opt.device, batch_size=opt.batch_size)
    if opt.local_rank != -1:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    # Train
    logger.info(opt)
    tb_writer = None
    if opt.global_rank in [-1, 0]:
        prefix = colorstr('tensorboard: ')
        logger.info(f"{prefix}Start with 'tensorboard --logdir {opt.project}', view at http://localhost:6006/")
        tb_writer = SummaryWriter(opt.save_dir)

    train(hyp, opt, device, tb_writer)
