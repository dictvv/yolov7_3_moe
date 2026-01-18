"""
MoE-YOLOv7 Detection Script

推論腳本，使用 WBF 合併多個專家的輸出

Usage:
    python detect_moe.py --weights runs/train/moe_exp/weights/best.pt --source data/images
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.moe_yolo import MoEYOLOv7
from models.wbf import apply_wbf_to_moe_output
from utils.datasets import LoadImages, LoadStreams
from utils.general import (check_img_size, non_max_suppression, scale_coords,
                           set_logging, increment_path, colorstr)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


def detect_moe(opt):
    """
    MoE-YOLOv7 推論主函數
    """
    source = opt.source
    weights = opt.weights
    imgsz = opt.img_size
    conf_thres = opt.conf_thres
    iou_thres = opt.iou_thres
    save_dir = Path(opt.save_dir)
    save_txt = opt.save_txt
    save_img = not opt.nosave

    # Directories
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # ============ Load Model ============
    print(f"Loading MoE-YOLOv7 from {weights}")

    # Load checkpoint
    ckpt = torch.load(weights, map_location=device, weights_only=False)

    # Get model config from checkpoint
    model_opt = ckpt.get('opt', {})
    num_experts = model_opt.get('num_experts', 4)
    top_k = model_opt.get('top_k', 2)

    # Determine nc and cfg
    hyp = ckpt.get('hyp', {})

    # Create model
    model = MoEYOLOv7(
        cfg=opt.cfg,
        num_experts=num_experts,
        top_k=top_k,
        nc=opt.nc,
        ch=3
    ).to(device)

    # Load weights
    model.load_state_dict(ckpt['model'])
    model.eval()

    if half:
        model.half()

    # Get names and colors
    names = model.names if hasattr(model, 'names') else [str(i) for i in range(opt.nc)]
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    # Grid size
    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(imgsz, s=gs)

    # ============ Dataloader ============
    if source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://')):
        # Stream
        dataset = LoadStreams(source, img_size=imgsz, stride=gs)
    else:
        # Images/Videos
        dataset = LoadImages(source, img_size=imgsz, stride=gs)

    # ============ Run Inference ============
    if device.type != 'cpu':
        # Warmup
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():
            moe_output = model(img)
        t2 = time_synchronized()

        # ============ WBF Merge ============
        # 獲取圖片尺寸
        if isinstance(im0s, list):
            im0 = im0s[0]
        else:
            im0 = im0s
        img_size = (im0.shape[0], im0.shape[1])  # (height, width)

        # 應用 WBF 合併專家輸出
        detections = apply_wbf_to_moe_output(
            moe_output,
            img_size=img_size,
            iou_thr=opt.wbf_iou_thres,
            conf_thr=conf_thres
        )

        # 處理每張圖的檢測結果
        for i, det in enumerate(detections):
            if isinstance(im0s, list):
                p, s, im0 = path[i], f'{i}: ', im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s.copy()

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem)

            s += f'{img.shape[2]}x{img.shape[3]} '

            # Expert info
            top_indices = moe_output['top_indices'][i].cpu().numpy()
            top_weights = moe_output['top_weights'][i].cpu().numpy()
            expert_info = ', '.join([f'E{idx}:{w:.2f}' for idx, w in zip(top_indices, top_weights)])
            s += f'[{expert_info}] '

            if len(det):
                # det is [M, 6]: (x1, y1, x2, y2, conf, cls)
                det = det.numpy() if isinstance(det, torch.Tensor) else det

                s += f'{len(det)} detections, '

                # Write results
                for *xyxy, conf, cls in det:
                    cls = int(cls)

                    if save_txt:
                        # Save to txt file
                        (save_dir / 'labels').mkdir(parents=True, exist_ok=True)
                        with open(txt_path + '.txt', 'a') as f:
                            # YOLO format: class x_center y_center width height
                            xywh = xyxy_to_xywh(xyxy, im0.shape[1], im0.shape[0])
                            f.write(f'{cls} {" ".join(map(str, xywh))} {conf:.4f}\n')

                    if save_img:
                        # Draw box
                        label = f'{names[cls]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[cls], line_thickness=2)

            # Print time
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Save image
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    # Video
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    save_path = str(Path(save_path).with_suffix('.mp4'))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    print(f'Done. ({time.time() - t0:.3f}s)')
    print(f'Results saved to {save_dir}')


def xyxy_to_xywh(xyxy, img_width, img_height):
    """
    Convert xyxy format to normalized xywh format
    """
    x1, y1, x2, y2 = xyxy
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return [x_center, y_center, width, height]


def visualize_expert_selection(moe_output, img, names, save_path=None):
    """
    視覺化專家選擇結果
    """
    import matplotlib.pyplot as plt

    batch_size = moe_output['top_indices'].shape[0]
    num_experts = len(moe_output['expert_outputs'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Expert selection for this batch
    top_indices = moe_output['top_indices'].cpu().numpy()
    top_weights = moe_output['top_weights'].cpu().numpy()

    # Count expert usage
    expert_counts = np.zeros(num_experts)
    for indices in top_indices:
        for idx in indices:
            expert_counts[idx] += 1

    axes[0].bar(range(num_experts), expert_counts)
    axes[0].set_xlabel('Expert Index')
    axes[0].set_ylabel('Selection Count')
    axes[0].set_title('Expert Selection Distribution')
    axes[0].set_xticks(range(num_experts))

    # Average weights
    avg_weights = np.zeros(num_experts)
    weight_counts = np.zeros(num_experts)
    for indices, weights in zip(top_indices, top_weights):
        for idx, w in zip(indices, weights):
            avg_weights[idx] += w
            weight_counts[idx] += 1

    avg_weights = np.divide(avg_weights, weight_counts, where=weight_counts > 0)

    axes[1].bar(range(num_experts), avg_weights)
    axes[1].set_xlabel('Expert Index')
    axes[1].set_ylabel('Average Weight')
    axes[1].set_title('Expert Average Weights')
    axes[1].set_xticks(range(num_experts))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f'Expert visualization saved to {save_path}')
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--weights', type=str, default='runs/train/moe_exp/weights/best.pt', help='model weights path')
    parser.add_argument('--cfg', type=str, default='cfg/training/yolov7-tiny.yaml', help='model.yaml path')
    parser.add_argument('--nc', type=int, default=80, help='number of classes')

    # Source
    parser.add_argument('--source', type=str, default='data/images', help='source')

    # Inference
    parser.add_argument('--img-size', type=int, default=640, help='inference size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--wbf-iou-thres', type=float, default=0.55, help='WBF IoU threshold')

    # Output
    parser.add_argument('--device', default='', help='cuda device')
    parser.add_argument('--project', default='runs/detect', help='save to project/name')
    parser.add_argument('--name', default='moe_exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project ok')
    parser.add_argument('--save-txt', action='store_true', help='save results to txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images')

    opt = parser.parse_args()

    # Save directory
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    print(f"Detecting with MoE-YOLOv7")
    print(f"  Weights: {opt.weights}")
    print(f"  Source: {opt.source}")
    print(f"  WBF IoU threshold: {opt.wbf_iou_thres}")

    with torch.no_grad():
        detect_moe(opt)
