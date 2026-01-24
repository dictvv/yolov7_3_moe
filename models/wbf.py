# models/wbf.py
"""
Weighted Boxes Fusion (WBF) - 合併多專家的偵測結果

用於推論階段，將多個專家的輸出合併成單一結果
"""

import torch
import torch.nn.functional as F


def weighted_boxes_fusion(
    expert_outputs,
    top_indices,
    top_weights,
    iou_threshold=0.55,
    conf_threshold=0.001
):
    """
    合併多專家的偵測結果

    Args:
        expert_outputs: dict {expert_idx: prediction}
            每個 prediction 的格式為 [B, num_boxes, 5+nc]
            其中 5+nc = [x, y, w, h, conf, class_scores...]
        top_indices: [B, top_k] 被選中的專家索引
        top_weights: [B, top_k] 對應的權重
        iou_threshold: WBF 合併的 IoU 閾值
        conf_threshold: 過濾低信心預測的閾值

    Returns:
        merged: [B, num_boxes, 5+nc] 合併後的預測
    """
    batch_size = top_indices.shape[0]
    top_k = top_indices.shape[1]

    # 處理 Detect 層的輸出格式
    # 推論時 Detect 返回 (inference_out, training_out)
    # 我們只需要 inference_out
    processed_outputs = {}
    for expert_idx, pred in expert_outputs.items():
        if isinstance(pred, tuple):
            # 推論模式：取第一個元素
            pred = pred[0]
        processed_outputs[expert_idx] = pred

    # 取得輸出維度
    sample_pred = list(processed_outputs.values())[0]
    device = sample_pred.device
    num_outputs = sample_pred.shape[-1]  # 5 + nc

    results = []

    for b in range(batch_size):
        # 收集這個 batch sample 被選中的專家輸出
        batch_preds = []
        batch_weights = []

        for k in range(top_k):
            expert_idx = top_indices[b, k].item()
            weight = top_weights[b, k].item()

            if expert_idx in processed_outputs:
                pred = processed_outputs[expert_idx][b]  # [num_boxes, 5+nc]
                batch_preds.append(pred)
                batch_weights.append(weight)

        if not batch_preds:
            # 沒有預測，返回空結果
            results.append(torch.zeros(0, num_outputs, device=device))
            continue

        # 簡化版 WBF：加權平均
        # 完整版 WBF 需要更複雜的 box clustering
        merged = _simple_weighted_fusion(
            batch_preds, batch_weights, iou_threshold, conf_threshold
        )
        results.append(merged)

    # 統一輸出大小
    max_boxes = max(r.shape[0] for r in results) if results else 0
    if max_boxes == 0:
        return torch.zeros(batch_size, 0, num_outputs, device=device)

    # Padding 到相同大小
    padded_results = []
    for r in results:
        if r.shape[0] < max_boxes:
            padding = torch.zeros(max_boxes - r.shape[0], num_outputs, device=device)
            r = torch.cat([r, padding], dim=0)
        padded_results.append(r)

    return torch.stack(padded_results, dim=0)


def _simple_weighted_fusion(preds, weights, iou_threshold, conf_threshold):
    """
    簡化版加權融合

    對於每個專家的預測，根據權重調整信心分數後合併
    然後使用 NMS 去除重複框

    Args:
        preds: list of [num_boxes, 5+nc]
        weights: list of float
        iou_threshold: NMS 的 IoU 閾值
        conf_threshold: 信心閾值

    Returns:
        merged: [num_boxes, 5+nc]
    """
    if not preds:
        return torch.zeros(0, preds[0].shape[-1] if preds else 6)

    device = preds[0].device
    num_outputs = preds[0].shape[-1]

    # 合併所有預測
    all_preds = []
    for pred, weight in zip(preds, weights):
        # 調整信心分數
        adjusted = pred.clone()
        adjusted[:, 4] = adjusted[:, 4] * weight  # conf *= weight
        all_preds.append(adjusted)

    merged = torch.cat(all_preds, dim=0)  # [total_boxes, 5+nc]

    if merged.shape[0] == 0:
        return merged

    # 過濾低信心預測
    mask = merged[:, 4] > conf_threshold
    merged = merged[mask]

    if merged.shape[0] == 0:
        return torch.zeros(0, num_outputs, device=device)

    # 應用 NMS
    # 轉換為 xyxy 格式
    boxes_xywh = merged[:, :4]
    boxes_xyxy = xywh2xyxy(boxes_xywh)
    scores = merged[:, 4]

    # 使用 torchvision NMS
    try:
        from torchvision.ops import nms
        keep = nms(boxes_xyxy, scores, iou_threshold)
    except ImportError:
        # 如果沒有 torchvision，使用簡單的 NMS
        keep = _simple_nms(boxes_xyxy, scores, iou_threshold)

    return merged[keep]


def xywh2xyxy(x):
    """
    將 [x, y, w, h] 轉換為 [x1, y1, x2, y2]
    """
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


def _simple_nms(boxes, scores, iou_threshold):
    """
    簡單的 NMS 實現（備用）

    Args:
        boxes: [N, 4] xyxy 格式
        scores: [N]
        iou_threshold: IoU 閾值

    Returns:
        keep: 保留的索引
    """
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break

        i = order[0].item()
        keep.append(i)

        # 計算 IoU
        ious = box_iou_single(boxes[i], boxes[order[1:]])

        # 保留 IoU 小於閾值的框
        mask = ious <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, device=boxes.device)


def box_iou_single(box1, boxes):
    """
    計算單個框與多個框的 IoU

    Args:
        box1: [4] xyxy 格式
        boxes: [N, 4] xyxy 格式

    Returns:
        ious: [N]
    """
    # 計算交集
    inter_x1 = torch.max(box1[0], boxes[:, 0])
    inter_y1 = torch.max(box1[1], boxes[:, 1])
    inter_x2 = torch.min(box1[2], boxes[:, 2])
    inter_y2 = torch.min(box1[3], boxes[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # 計算並集
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box1_area + boxes_area - inter_area

    return inter_area / (union_area + 1e-6)
