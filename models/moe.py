# models/moe.py
"""
MoE Router - 根據 Backbone P5 輸出選擇 Top-K 專家
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """
    Top-K 路由器，帶 Load Balancing Loss

    Args:
        in_channels: P5 輸出通道數 (YOLOv7-tiny: 512)
        num_experts: 專家數量
        top_k: 每張圖片選擇的專家數量
    """

    def __init__(self, in_channels=512, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 網路層
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_experts)

        # 儲存 aux_loss 供訓練使用
        self.aux_loss = None

    def forward(self, x):
        """
        Args:
            x: P5 特徵 [B, C, H, W]

        Returns:
            top_indices: 選中的專家索引 [B, top_k]
            top_weights: 歸一化後的權重 [B, top_k]
        """
        # 1. Global Average Pooling: [B, C, H, W] -> [B, C]
        pooled = self.pool(x).flatten(1)

        # 2. FC 計算每個專家的分數: [B, C] -> [B, num_experts]
        scores = self.fc(pooled)

        # 3. Softmax 轉換為概率
        probs = F.softmax(scores, dim=-1)

        # 4. 選擇 Top-K 專家
        top_weights, top_indices = probs.topk(self.top_k, dim=-1)

        # 5. 歸一化權重 (讓 Top-K 權重總和為 1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # 6. 計算 Load Balancing Loss
        self._compute_aux_loss(probs, top_indices)

        return top_indices, top_weights

    def _compute_aux_loss(self, probs, top_indices):
        """
        計算 Load Balancing Loss (參考 Switch Transformer)

        aux_loss = N * sum(f_i * P_i)

        f_i: 專家 i 被選中的比例 (實際分配)
        P_i: 專家 i 的平均路由概率
        """
        # 統計每個專家被選中的次數
        # top_indices: [B, top_k]
        expert_mask = F.one_hot(top_indices, self.num_experts)  # [B, top_k, num_experts]
        expert_mask = expert_mask.sum(dim=1)  # [B, num_experts]

        # f: 每個專家被選中的比例
        f = expert_mask.float().mean(dim=0)  # [num_experts]

        # P: 每個專家的平均概率
        P = probs.mean(dim=0)  # [num_experts]

        # Load Balancing Loss
        self.aux_loss = self.num_experts * (f * P).sum()
