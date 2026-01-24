# utils/loss_moe.py
"""
MoE-YOLOv7 Loss 計算

設計原則：
- 100% 使用原版 ComputeLoss 計算每個專家的 loss
- 根據 Router 權重加權合併
- 加入 Load Balancing Loss
"""

import torch
import torch.nn as nn
from utils.loss import ComputeLoss


class ExpertModelWrapper:
    """
    包裝專家模型，讓它符合 ComputeLoss 需要的介面

    ComputeLoss 需要從 model 獲取：
    - model.hyp: 超參數
    - model.gr: giou loss ratio
    - model.model[-1]: Detect 層 (用於獲取 na, nc, nl, anchors)
    """

    def __init__(self, expert_layers, hyp, gr):
        self.hyp = hyp
        self.gr = gr
        # 建立一個假的 model 結構，讓 model[-1] 指向 Detect 層
        self.model = expert_layers

    def parameters(self):
        """讓 ComputeLoss 可以獲取 device"""
        for m in self.model:
            for p in m.parameters():
                yield p
                return


class ComputeLossMoE:
    """
    MoE Loss 計算

    計算流程：
    1. 為每個被選中的專家計算 loss（使用原版 ComputeLoss）
    2. 根據 Router 權重加權合併
    3. 加入 Load Balancing Loss
    """

    def __init__(self, model, autobalance=False, aux_loss_weight=0.01):
        """
        Args:
            model: MoEYOLOv7 模型
            autobalance: 是否自動平衡 objectness loss
            aux_loss_weight: Load Balancing Loss 權重
        """
        self.aux_loss_weight = aux_loss_weight
        self.num_experts = model.num_experts

        # 為每個專家建立 ComputeLoss
        self.loss_fns = []
        for i, expert in enumerate(model.experts):
            # 包裝專家讓它符合 ComputeLoss 介面
            wrapper = ExpertModelWrapper(expert, model.hyp, model.gr)
            loss_fn = ComputeLoss(wrapper, autobalance)
            self.loss_fns.append(loss_fn)

    def __call__(self, moe_output, targets):
        """
        計算 MoE Loss

        Args:
            moe_output: MoEYOLOv7.forward() 的輸出
                - expert_outputs: {expert_idx: prediction}
                - top_indices: [B, top_k]
                - top_weights: [B, top_k]
                - aux_loss: scalar (訓練時)
            targets: 標註資料 [N, 6] (image_idx, class, x, y, w, h)

        Returns:
            loss: 總 loss (scalar)
            loss_items: [lbox, lobj, lcls, total_loss] (用於 logging)
        """
        device = targets.device
        expert_outputs = moe_output['expert_outputs']
        top_indices = moe_output['top_indices']  # [B, top_k]
        top_weights = moe_output['top_weights']  # [B, top_k]

        batch_size = top_indices.shape[0]
        top_k = top_indices.shape[1]

        # 1. 計算每個被選中專家的 loss
        expert_losses = {}  # {expert_idx: (loss, loss_items)}

        for expert_idx, pred in expert_outputs.items():
            loss, loss_items = self.loss_fns[expert_idx](pred, targets)
            expert_losses[expert_idx] = (loss, loss_items)

        # 2. 根據 Router 權重加權合併
        # 簡化版本：使用所有被選中專家的平均權重
        # （因為同一個 batch 內不同圖片可能選擇不同專家）

        total_loss = torch.zeros(1, device=device)
        total_loss_items = torch.zeros(4, device=device)

        # 計算每個專家的平均權重
        expert_avg_weights = {}
        for expert_idx in expert_outputs.keys():
            # 找出選擇這個專家的 batch indices 和對應權重
            mask = (top_indices == expert_idx)  # [B, top_k]
            if mask.any():
                avg_weight = top_weights[mask].mean()
                expert_avg_weights[expert_idx] = avg_weight

        # 歸一化權重
        weight_sum = sum(expert_avg_weights.values())

        for expert_idx, (loss, loss_items) in expert_losses.items():
            if expert_idx in expert_avg_weights:
                normalized_weight = expert_avg_weights[expert_idx] / weight_sum
                total_loss += normalized_weight * loss
                total_loss_items += normalized_weight * loss_items

        # 3. 加入 Load Balancing Loss
        if 'aux_loss' in moe_output and moe_output['aux_loss'] is not None:
            aux_loss = self.aux_loss_weight * moe_output['aux_loss']
            total_loss += aux_loss
            # aux_loss 加到 total 項
            total_loss_items[3] += aux_loss.detach()

        return total_loss, total_loss_items.detach()
