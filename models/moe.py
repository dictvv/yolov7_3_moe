"""
MoE (Mixture of Experts) Router Module
用於 MoE-YOLOv7 的路由器

功能:
- 根據 Backbone P5 輸出選擇 Top-K 專家
- 計算 Load Balancing Loss 防止專家偏好
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """
    帶 Load Balancing 的 Top-K 路由器

    根據 Backbone P5 輸出選擇最適合的專家

    Args:
        in_channels: Backbone P5 輸出通道數 (YOLOv7-tiny: 512)
        num_experts: 專家數量
        top_k: 每張圖片選擇的專家數量
    """
    def __init__(self, in_channels=512, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 特徵提取
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_experts)

        # 儲存最近一次的 aux_loss 供訓練使用
        self.aux_loss = None

        # 儲存選擇統計供 logging
        self.last_probs = None

    def forward(self, x):
        """
        Args:
            x: Backbone P5 輸出 [B, C, H, W]

        Returns:
            top_indices: 選中的專家索引 [B, top_k]
            top_weights: 歸一化後的權重 [B, top_k]
        """
        batch_size = x.shape[0]

        # Global Average Pooling
        pooled = self.pool(x).flatten(1)  # [B, C]

        # 計算每個專家的分數
        scores = self.fc(pooled)  # [B, num_experts]

        # Softmax 轉換為概率
        probs = F.softmax(scores, dim=-1)  # [B, num_experts]
        self.last_probs = probs.detach()

        # 選擇 Top-K 專家
        top_weights, top_indices = probs.topk(self.top_k, dim=-1)

        # 歸一化權重 (讓 Top-K 權重總和為 1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # ============ Load Balancing Loss ============
        # 參考: Switch Transformer (Google, 2021)
        #
        # f_i: 每個專家被選中的比例 (實際分配)
        # P_i: 每個專家的平均路由概率
        # aux_loss = N * sum(f_i * P_i)
        #
        # 這個 loss 會懲罰不平衡的專家使用
        # =============================================

        # 統計每個專家被選中的次數
        expert_mask = F.one_hot(top_indices, self.num_experts)  # [B, top_k, num_experts]
        expert_mask = expert_mask.sum(dim=1)  # [B, num_experts]

        # f: 每個專家被選的比例
        f = expert_mask.float().mean(dim=0)  # [num_experts]

        # P: 每個專家的平均概率
        P = probs.mean(dim=0)  # [num_experts]

        # Load Balancing Loss
        self.aux_loss = self.num_experts * (f * P).sum()

        return top_indices, top_weights

    def get_expert_usage(self, top_indices):
        """
        統計專家使用分布

        Args:
            top_indices: [B, top_k]

        Returns:
            usage: [num_experts] 每個專家被選中的次數
        """
        flat_indices = top_indices.flatten()
        usage = torch.bincount(flat_indices, minlength=self.num_experts)
        return usage
