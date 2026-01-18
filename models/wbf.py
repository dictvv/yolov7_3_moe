"""
Weighted Boxes Fusion (WBF) for MoE-YOLOv7

WBF 合併多個專家的偵測結果
參考: https://github.com/ZFTurbo/Weighted-Boxes-Fusion

與 NMS 不同，WBF 會融合重疊的框而非直接丟棄
這對於 MoE 架構特別有用，因為不同專家可能對同一物體有稍微不同的預測
"""

import torch
import numpy as np
from collections import defaultdict


def weighted_boxes_fusion(
    boxes_list,
    scores_list,
    labels_list,
    weights=None,
    iou_thr=0.55,
    skip_box_thr=0.0,
    conf_type='avg',
    allows_overflow=False
):
    """
    Weighted Boxes Fusion

    Args:
        boxes_list: list of arrays, 每個 array shape [N, 4] (x1, y1, x2, y2) 歸一化座標
        scores_list: list of arrays, 每個 array shape [N]
        labels_list: list of arrays, 每個 array shape [N]
        weights: list of floats, 每個模型的權重
        iou_thr: IoU 閾值，超過此值的框會被融合
        skip_box_thr: 低於此置信度的框會被跳過
        conf_type: 置信度計算方式 ('avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg')
        allows_overflow: 是否允許座標超出 [0, 1]

    Returns:
        boxes: array [M, 4]
        scores: array [M]
        labels: array [M]
    """
    if weights is None:
        weights = [1.0] * len(boxes_list)

    # 確保權重總和為 1
    weights = np.array(weights)
    weights = weights / weights.sum()

    # 按類別分組處理
    all_boxes = defaultdict(list)
    all_scores = defaultdict(list)
    all_weights = defaultdict(list)

    for model_idx, (boxes, scores, labels) in enumerate(zip(boxes_list, scores_list, labels_list)):
        if len(boxes) == 0:
            continue

        for box, score, label in zip(boxes, scores, labels):
            if score < skip_box_thr:
                continue

            label = int(label)
            all_boxes[label].append(box)
            all_scores[label].append(score)
            all_weights[label].append(weights[model_idx])

    # 對每個類別進行 WBF
    final_boxes = []
    final_scores = []
    final_labels = []

    for label in all_boxes:
        boxes = np.array(all_boxes[label])
        scores = np.array(all_scores[label])
        box_weights = np.array(all_weights[label])

        # 執行 WBF
        fused_boxes, fused_scores = _wbf_single_class(
            boxes, scores, box_weights, iou_thr, conf_type, allows_overflow
        )

        for box, score in zip(fused_boxes, fused_scores):
            final_boxes.append(box)
            final_scores.append(score)
            final_labels.append(label)

    if len(final_boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0)

    final_boxes = np.array(final_boxes)
    final_scores = np.array(final_scores)
    final_labels = np.array(final_labels)

    # 按置信度排序
    sorted_indices = np.argsort(-final_scores)
    return final_boxes[sorted_indices], final_scores[sorted_indices], final_labels[sorted_indices]


def _wbf_single_class(boxes, scores, weights, iou_thr, conf_type, allows_overflow):
    """
    單一類別的 WBF

    Args:
        boxes: [N, 4] 框座標
        scores: [N] 置信度
        weights: [N] 每個框的權重 (來自模型權重)
        iou_thr: IoU 閾值
        conf_type: 置信度計算方式
        allows_overflow: 是否允許座標超出 [0, 1]

    Returns:
        fused_boxes: [M, 4]
        fused_scores: [M]
    """
    if len(boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0)

    # 按置信度排序
    order = np.argsort(-scores)
    boxes = boxes[order]
    scores = scores[order]
    weights = weights[order]

    # 融合後的框
    fused_boxes = []
    fused_scores = []

    # 記錄每個框是否已被融合
    used = np.zeros(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        if used[i]:
            continue

        # 找到所有與當前框重疊的框
        cluster_boxes = [boxes[i]]
        cluster_scores = [scores[i]]
        cluster_weights = [weights[i]]
        used[i] = True

        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue

            iou = _compute_iou(boxes[i], boxes[j])
            if iou > iou_thr:
                cluster_boxes.append(boxes[j])
                cluster_scores.append(scores[j])
                cluster_weights.append(weights[j])
                used[j] = True

        # 計算融合後的框
        cluster_boxes = np.array(cluster_boxes)
        cluster_scores = np.array(cluster_scores)
        cluster_weights = np.array(cluster_weights)

        # 加權平均計算融合框座標
        weighted_scores = cluster_scores * cluster_weights
        total_weight = weighted_scores.sum()

        if total_weight > 0:
            fused_box = (cluster_boxes * weighted_scores[:, np.newaxis]).sum(axis=0) / total_weight
        else:
            fused_box = cluster_boxes[0]

        # 計算融合後的置信度
        if conf_type == 'avg':
            fused_score = cluster_scores.mean()
        elif conf_type == 'max':
            fused_score = cluster_scores.max()
        elif conf_type == 'box_and_model_avg':
            fused_score = (cluster_scores * cluster_weights).sum() / cluster_weights.sum()
        else:
            fused_score = cluster_scores.mean()

        # 提升融合了多個框的置信度
        # 如果多個模型都偵測到同一物體，置信度應該更高
        n_models = len(cluster_boxes)
        if n_models > 1:
            fused_score = fused_score * (1 + 0.1 * (n_models - 1))
            fused_score = min(fused_score, 1.0)

        # 限制座標範圍
        if not allows_overflow:
            fused_box = np.clip(fused_box, 0, 1)

        fused_boxes.append(fused_box)
        fused_scores.append(fused_score)

    return np.array(fused_boxes), np.array(fused_scores)


def _compute_iou(box1, box2):
    """
    計算兩個框的 IoU

    Args:
        box1, box2: [4] (x1, y1, x2, y2)

    Returns:
        iou: float
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    if union_area <= 0:
        return 0

    return inter_area / union_area


class WBFMerger:
    """
    WBF 合併器，用於 MoE-YOLOv7 推論時合併多個專家的輸出

    Usage:
        merger = WBFMerger(iou_thr=0.55, conf_thr=0.001)
        merged_output = merger.merge(expert_outputs, expert_weights, img_size)
    """

    def __init__(self, iou_thr=0.55, conf_thr=0.001, conf_type='avg'):
        """
        Args:
            iou_thr: IoU 閾值，超過此值的框會被融合
            conf_thr: 置信度閾值，低於此值的框會被跳過
            conf_type: 置信度計算方式
        """
        self.iou_thr = iou_thr
        self.conf_thr = conf_thr
        self.conf_type = conf_type

    def merge(self, expert_outputs, expert_weights, img_size):
        """
        合併多個專家的輸出

        Args:
            expert_outputs: dict {expert_idx: tensor [B, N, 5+nc]}
                           每個 tensor 包含 [x, y, w, h, obj_conf, cls_conf...]
            expert_weights: dict {expert_idx: weight}
            img_size: (height, width) 圖片尺寸

        Returns:
            merged: list of tensors, 每個 tensor 是一張圖的合併結果
                   shape [M, 6] (x1, y1, x2, y2, conf, cls)
        """
        # 取得 batch size
        first_output = next(iter(expert_outputs.values()))
        batch_size = first_output.shape[0]

        merged_results = []

        for b in range(batch_size):
            # 收集這張圖的所有專家輸出
            boxes_list = []
            scores_list = []
            labels_list = []
            weights_list = []

            for expert_idx, output in expert_outputs.items():
                # output shape: [B, N, 5+nc]
                pred = output[b]  # [N, 5+nc]

                # 解析預測
                boxes, scores, labels = self._parse_predictions(pred, img_size)

                if len(boxes) > 0:
                    boxes_list.append(boxes)
                    scores_list.append(scores)
                    labels_list.append(labels)
                    weights_list.append(expert_weights.get(expert_idx, 1.0))

            # WBF 合併
            if len(boxes_list) > 0:
                fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
                    boxes_list, scores_list, labels_list,
                    weights=weights_list,
                    iou_thr=self.iou_thr,
                    skip_box_thr=self.conf_thr,
                    conf_type=self.conf_type
                )

                # 轉換回絕對座標
                h, w = img_size
                fused_boxes[:, [0, 2]] *= w
                fused_boxes[:, [1, 3]] *= h

                # 組合結果 [M, 6]
                result = np.column_stack([
                    fused_boxes,
                    fused_scores,
                    fused_labels
                ])
                merged_results.append(torch.from_numpy(result).float())
            else:
                merged_results.append(torch.zeros((0, 6)))

        return merged_results

    def _parse_predictions(self, pred, img_size):
        """
        解析 YOLO 格式的預測

        Args:
            pred: tensor [N, 5+nc] (x, y, w, h, obj_conf, cls_conf...)
            img_size: (height, width)

        Returns:
            boxes: [M, 4] 歸一化座標 (x1, y1, x2, y2)
            scores: [M]
            labels: [M]
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        h, w = img_size

        # 過濾低置信度
        obj_conf = pred[:, 4]
        mask = obj_conf > self.conf_thr
        pred = pred[mask]

        if len(pred) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0)

        # 轉換 xywh -> xyxy
        boxes = np.zeros((len(pred), 4))
        boxes[:, 0] = pred[:, 0] - pred[:, 2] / 2  # x1
        boxes[:, 1] = pred[:, 1] - pred[:, 3] / 2  # y1
        boxes[:, 2] = pred[:, 0] + pred[:, 2] / 2  # x2
        boxes[:, 3] = pred[:, 1] + pred[:, 3] / 2  # y2

        # 歸一化到 [0, 1]
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h
        boxes = np.clip(boxes, 0, 1)

        # 計算置信度和類別
        cls_conf = pred[:, 5:]
        labels = cls_conf.argmax(axis=1)
        scores = pred[:, 4] * cls_conf.max(axis=1)  # obj_conf * cls_conf

        return boxes, scores, labels


def apply_wbf_to_moe_output(moe_output, img_size, iou_thr=0.55, conf_thr=0.001):
    """
    便捷函數：對 MoE 模型輸出應用 WBF

    Args:
        moe_output: dict from MoEYOLOv7.forward()
            - expert_outputs: {expert_idx: prediction}
            - top_indices: [B, top_k]
            - top_weights: [B, top_k]
        img_size: (height, width)
        iou_thr: IoU 閾值
        conf_thr: 置信度閾值

    Returns:
        list of tensors: 每張圖的合併結果 [M, 6]
    """
    expert_outputs = moe_output['expert_outputs']
    top_indices = moe_output['top_indices']
    top_weights = moe_output['top_weights']

    batch_size = top_indices.shape[0]
    merger = WBFMerger(iou_thr=iou_thr, conf_thr=conf_thr)

    all_results = []

    for b in range(batch_size):
        # 取得這張圖選中的專家和權重
        indices = top_indices[b].cpu().numpy()
        weights = top_weights[b].cpu().numpy()

        # 收集選中專家的輸出
        selected_outputs = {}
        selected_weights = {}

        for i, (expert_idx, weight) in enumerate(zip(indices, weights)):
            expert_idx = int(expert_idx)
            if expert_idx in expert_outputs:
                # 只取這張圖的預測
                selected_outputs[expert_idx] = expert_outputs[expert_idx][b:b+1]
                selected_weights[expert_idx] = float(weight)

        # WBF 合併
        if selected_outputs:
            merged = merger.merge(selected_outputs, selected_weights, img_size)
            all_results.append(merged[0])
        else:
            all_results.append(torch.zeros((0, 6)))

    return all_results
