"""
MoE-YOLOv7 Training Script

基於原始 train.py 修改，用於訓練 MoE-YOLOv7 模型

主要修改:
1. 使用 MoEYOLOv7 模型
2. 使用 ComputeLossMoE 計算 loss
3. 追蹤專家使用統計
"""

import argparse
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from torch.cuda import amp
from tqdm import tqdm

from models.moe_yolo import MoEYOLOv7
from utils.datasets import create_dataloader
from utils.general import (labels_to_class_weights, increment_path, init_seeds,
                           fitness, check_dataset, check_file, check_img_size,
                           set_logging, one_cycle, colorstr)
from utils.loss_moe import ComputeLossMoE, ComputeLossMoESimple
from utils.torch_utils import ModelEMA, select_device

logger = logging.getLogger(__name__)


def train_moe(hyp, opt, device):
    """
    MoE-YOLOv7 訓練主函數
    """
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    save_dir = Path(opt.save_dir)
    epochs = opt.epochs
    batch_size = opt.batch_size
    weights = opt.weights

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure
    cuda = device.type != 'cpu'
    init_seeds(1)

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)

    nc = 1 if opt.single_cls else int(data_dict['nc'])
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset'

    # ============ MoE Model ============
    logger.info(f"Creating MoE-YOLOv7 with {opt.num_experts} experts, top-{opt.top_k}")

    model = MoEYOLOv7(
        cfg=opt.cfg,
        num_experts=opt.num_experts,
        top_k=opt.top_k,
        nc=nc,
        ch=3
    ).to(device)

    # Load pretrained weights
    if weights and os.path.exists(weights):
        logger.info(f"Loading pretrained weights from {weights}")
        model.load_pretrained(weights, verbose=True)

    # Check dataset
    check_dataset(data_dict)
    train_path = data_dict['train']
    test_path = data_dict['val']

    # ============ Freeze Backbone (optional) ============
    if opt.freeze_backbone:
        logger.info("Freezing backbone parameters")
        for param in model.backbone.parameters():
            param.requires_grad = False

    # ============ Optimizer ============
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    hyp['weight_decay'] *= batch_size * accumulate / nbs

    # Separate parameter groups
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
    logger.info(f'Optimizer groups: {len(pg2)} .bias, {len(pg1)} conv.weight, {len(pg0)} other')
    del pg0, pg1, pg2

    # ============ Scheduler ============
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # ============ EMA ============
    ema = ModelEMA(model)

    # ============ DataLoader ============
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.img_size, gs)

    dataloader, dataset = create_dataloader(
        train_path, imgsz, batch_size, gs, opt,
        hyp=hyp, augment=True, cache=opt.cache_images,
        rect=opt.rect, rank=-1, world_size=1,
        workers=opt.workers, prefix=colorstr('train: ')
    )
    nb = len(dataloader)

    # TestLoader
    testloader = create_dataloader(
        test_path, imgsz, batch_size * 2, gs, opt,
        hyp=hyp, cache=opt.cache_images,
        rect=True, rank=-1, world_size=1,
        workers=opt.workers, pad=0.5, prefix=colorstr('val: ')
    )[0]

    # ============ Model Parameters ============
    nl = 3  # number of detection layers (YOLOv7-tiny)
    hyp['box'] *= 3. / nl
    hyp['cls'] *= nc / 80. * 3. / nl
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl
    hyp['label_smoothing'] = opt.label_smoothing

    model.nc = nc
    model.hyp = hyp
    model.gr = 1.0
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    # ============ Loss Function ============
    compute_loss = ComputeLossMoESimple(model, aux_loss_weight=opt.aux_loss_weight)
    compute_loss.set_hyp(hyp)

    # ============ Training ============
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    best_fitness = 0.0
    results = (0, 0, 0, 0, 0, 0, 0)
    scheduler.last_epoch = -1
    scaler = amp.GradScaler(enabled=cuda)

    logger.info(f'Image size: {imgsz}\n'
                f'Using {dataloader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')

    # 寫入 results 文件標題
    with open(results_file, 'w') as f:
        f.write('epoch\tbox\tobj\tcls\ttotal\tP\tR\tmAP@0.5\tmAP@0.5:0.95\n')

    # Expert usage tracking
    expert_usage_history = []

    for epoch in range(epochs):
        model.train()

        mloss = torch.zeros(4, device=device)
        expert_counts = torch.zeros(opt.num_experts, device=device)

        pbar = enumerate(dataloader)
        logger.info(('\n' + '%10s' * 9) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'labels', 'img_size', 'experts'))

        pbar = tqdm(pbar, total=nb)
        optimizer.zero_grad()

        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs
                sf = sz / max(imgs.shape[2:])
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                moe_output = model(imgs)
                loss, loss_items = compute_loss(moe_output, targets.to(device))

            # Track expert usage
            top_indices = moe_output['top_indices']
            for idx in top_indices.flatten():
                expert_counts[idx] += 1

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)

            # Expert usage string
            expert_str = ','.join([f'{int(c)}' for c in expert_counts[:4]])

            s = ('%10s' * 2 + '%10.4g' * 5 + '%10s' * 2) % (
                f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1], expert_str)
            pbar.set_description(s)

        # End epoch

        # Scheduler
        scheduler.step()

        # Log expert usage
        total_count = expert_counts.sum()
        if total_count > 0:
            expert_probs = (expert_counts / total_count).cpu().numpy()
            expert_usage_history.append(expert_probs)
            logger.info(f"Expert usage (epoch {epoch}): {expert_probs}")

        # ============ Validation ============
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])

        # 完整 mAP 驗證
        mp, mr, map50, map_val, val_loss = validate_moe(
            model, testloader, compute_loss, device, nc, names
        )
        logger.info(f"Validation: P={mp:.3f}, R={mr:.3f}, mAP@0.5={map50:.3f}, mAP@0.5:0.95={map_val:.3f}")

        # 使用 mAP@0.5 作為 fitness
        fi = map50

        # Save
        if fi > best_fitness:
            best_fitness = fi

        ckpt = {
            'epoch': epoch,
            'best_fitness': best_fitness,
            'model': model.state_dict(),
            'ema': ema.ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'expert_usage_history': expert_usage_history,
            'hyp': hyp,
            'opt': vars(opt),
            'results': (mp, mr, map50, map_val),  # 保存驗證結果
        }

        # Save last and best
        torch.save(ckpt, last)
        if best_fitness == fi:
            torch.save(ckpt, best)

        # Log
        with open(results_file, 'a') as f:
            f.write(f'{epoch}\t{mloss[0]:.4f}\t{mloss[1]:.4f}\t{mloss[2]:.4f}\t{mloss[3]:.4f}\t'
                    f'{mp:.4f}\t{mr:.4f}\t{map50:.4f}\t{map_val:.4f}\n')

        del ckpt

    # End training
    logger.info(f'{epochs} epochs completed in {(time.time() - t0) / 3600:.3f} hours.\n')

    # Final model info
    model.info()

    return results


def validate_moe(model, dataloader, compute_loss, device, nc, names, conf_thres=0.001, iou_thres=0.6):
    """
    MoE 模型驗證函數，計算 mAP

    Args:
        model: MoEYOLOv7 模型
        dataloader: 驗證資料載入器
        compute_loss: loss 計算器
        device: 計算裝置
        nc: 類別數量
        names: 類別名稱
        conf_thres: 置信度閾值
        iou_thres: NMS IoU 閾值

    Returns:
        results: (mp, mr, map50, map, val_loss)
    """
    from utils.metrics import ap_per_class
    from utils.general import box_iou, xywh2xyxy, scale_coords
    from models.wbf import apply_wbf_to_moe_output

    model.eval()

    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # IoU 向量 for mAP@0.5:0.95
    niou = iouv.numel()

    seen = 0
    stats = []
    total_loss = torch.zeros(3, device=device)
    n_batches = 0

    pbar = tqdm(dataloader, desc='Validating')

    with torch.no_grad():
        for batch_i, (imgs, targets, paths, shapes) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)
            nb, _, height, width = imgs.shape

            # Forward (訓練模式以獲取 loss)
            model.train()
            moe_output_train = model(imgs)
            loss, loss_items = compute_loss(moe_output_train, targets)
            total_loss += loss_items[:3]
            n_batches += 1

            # Forward (推論模式以獲取預測)
            model.eval()
            moe_output = model(imgs)

            # WBF 合併專家輸出
            img_size = (height, width)
            out = apply_wbf_to_moe_output(
                moe_output,
                img_size=img_size,
                iou_thr=0.55,
                conf_thr=conf_thres
            )

            # 將 targets 轉換為像素座標
            targets[:, 2:] *= torch.tensor([width, height, width, height], device=device)

            # 統計每張圖片
            for si, pred in enumerate(out):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool),
                                     torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # 預測結果
                pred = pred.to(device)
                predn = pred.clone()

                # 縮放到原始圖片尺寸
                if shapes is not None:
                    scale_coords(imgs[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])

                # 評估
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
                if nl:
                    detected = []
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    if shapes is not None:
                        scale_coords(imgs[si].shape[1:], tbox, shapes[si][0], shapes[si][1])

                    # 對每個類別
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)

                        if pi.shape[0]:
                            ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)

                            detected_set = []
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[i[j]]
                                if d.item() not in detected_set:
                                    detected_set.append(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv
                                    if len(detected) == nl:
                                        break

                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # 計算指標
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
    else:
        mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0

    # 平均 loss
    val_loss = (total_loss / max(n_batches, 1)).cpu().numpy()

    model.train()
    return mp, mr, map50, map, val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--cfg', type=str, default='cfg/training/yolov7-tiny.yaml', help='model.yaml path')
    parser.add_argument('--weights', type=str, default='', help='pretrained weights path')

    # MoE settings
    parser.add_argument('--num-experts', type=int, default=4, help='number of experts')
    parser.add_argument('--top-k', type=int, default=2, help='top-k experts to use')
    parser.add_argument('--aux-loss-weight', type=float, default=0.01, help='load balancing loss weight')
    parser.add_argument('--freeze-backbone', action='store_true', help='freeze backbone')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='image size')

    # Optimizer
    parser.add_argument('--adam', action='store_true', help='use Adam optimizer')
    parser.add_argument('--linear-lr', action='store_true', help='linear learning rate')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='label smoothing')

    # Other
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='moe_exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class')
    parser.add_argument('--cache-images', action='store_true', help='cache images')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--multi-scale', action='store_true', help='multi-scale training')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.tiny.yaml', help='hyperparameters path')

    opt = parser.parse_args()

    # Setup
    set_logging()
    device = select_device(opt.device, batch_size=opt.batch_size)

    # Save directory
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    # Train
    logger.info(f"Training MoE-YOLOv7")
    logger.info(f"  Experts: {opt.num_experts}")
    logger.info(f"  Top-K: {opt.top_k}")
    logger.info(f"  Aux Loss Weight: {opt.aux_loss_weight}")

    train_moe(hyp, opt, device)
