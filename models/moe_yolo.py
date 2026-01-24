# models/moe_yolo.py
"""
MoE-YOLOv7: 共享 Backbone + Router + 多專家 (Neck+Head)

架構:
- Backbone (Layer 0-28): 共享
- Router: 根據 P5 選擇 Top-K 專家
- Experts (Layer 29-77): 4 個獨立的 Neck+Head

Forward 邏輯完全複製原版 YOLOv7 的 forward_once
"""

import torch
import torch.nn as nn
from copy import deepcopy
from pathlib import Path

from models.yolo import Model
from models.moe import Router


class MoEYOLOv7(nn.Module):
    """
    MoE-YOLOv7 模型

    Args:
        cfg: 模型配置檔路徑 (yolov7-tiny.yaml)
        num_experts: 專家數量
        top_k: 每張圖選擇的專家數量
        nc: 類別數量
        ch: 輸入通道數
    """

    def __init__(self, cfg='cfg/training/yolov7-tiny.yaml', num_experts=4, top_k=2, nc=80, ch=3):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.nc = nc

        # 載入原版 Model 取得完整結構
        base_model = Model(cfg, ch=ch, nc=nc)

        # 拆分 Backbone (Layer 0-28)
        self.backbone = nn.ModuleList(list(base_model.model[:29]))

        # 建立 Router (P5 通道數: 512 for YOLOv7-tiny)
        self.router = Router(in_channels=512, num_experts=num_experts, top_k=top_k)

        # 複製 4 份 Expert (Layer 29-77)
        expert_layers = list(base_model.model[29:])
        self.experts = nn.ModuleList([
            nn.ModuleList(deepcopy(expert_layers)) for _ in range(num_experts)
        ])

        # 保存必要屬性（與原版相同）
        self.save = base_model.save
        self.stride = base_model.stride
        self.names = base_model.names if hasattr(base_model, 'names') else None
        self.yaml = base_model.yaml if hasattr(base_model, 'yaml') else None

        # 訓練相關
        self.gr = 1.0  # giou loss ratio
        self.hyp = None  # 訓練時設定

    def forward(self, x, augment=False, profile=False):
        """
        Forward pass

        Args:
            x: 輸入圖片 [B, 3, H, W]

        Returns:
            dict:
                - expert_outputs: {expert_idx: prediction}
                - top_indices: [B, top_k]
                - top_weights: [B, top_k]
                - aux_loss: scalar (訓練時)
        """
        # 1. Backbone forward (完全複製原版邏輯)
        p5, backbone_y = self._forward_backbone(x)

        # 2. Router 選擇專家
        top_indices, top_weights = self.router(p5)

        # 3. 執行被選中的專家
        expert_outputs = {}
        unique_experts = top_indices.unique().tolist()

        for expert_idx in unique_experts:
            expert_out = self._forward_expert(expert_idx, backbone_y)
            expert_outputs[expert_idx] = expert_out

        # 4. 返回結果
        result = {
            'expert_outputs': expert_outputs,
            'top_indices': top_indices,
            'top_weights': top_weights,
        }

        if self.training:
            result['aux_loss'] = self.router.aux_loss

        return result

    def _forward_backbone(self, x):
        """
        Backbone forward (Layer 0-28)

        完全複製原版 forward_once 的邏輯
        """
        y = []

        for m in self.backbone:
            # 處理 skip connection（與原版完全相同）
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            # Forward
            x = m(x)

            # 保存輸出（與原版完全相同）
            y.append(x if m.i in self.save else None)

        # x 是 P5 (Layer 28 的輸出)
        return x, y

    def _forward_expert(self, expert_idx, backbone_y):
        """
        單個專家的 forward (Layer 29-77)

        繼承 Backbone 的 y list，用原版邏輯繼續執行
        """
        expert = self.experts[expert_idx]

        # 複製 backbone 的 y（避免不同專家互相影響）
        y = backbone_y.copy()

        # 從 P5 開始（y 的最後一個非 None 元素）
        x = y[28]  # Layer 28 = P5

        for m in expert:
            # 處理 skip connection（與原版完全相同）
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]

            # Forward
            x = m(x)

            # 保存輸出（與原版完全相同）
            y.append(x if m.i in self.save else None)

        return x

    def load_pretrained(self, weights_path, verbose=True):
        """
        載入預訓練權重

        - Backbone: 載入對應權重
        - Experts: 全部用相同的預訓練權重初始化
        """
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)

        # 取得 state_dict
        if isinstance(ckpt, dict) and 'model' in ckpt:
            state_dict = ckpt['model'].float().state_dict()
        else:
            state_dict = ckpt.float().state_dict() if hasattr(ckpt, 'state_dict') else ckpt

        # 載入 Backbone (Layer 0-28)
        backbone_loaded = 0
        for key, value in state_dict.items():
            if key.startswith('model.'):
                parts = key.split('.')
                layer_idx = int(parts[1])

                if layer_idx < 29:
                    # 建立新的 key
                    new_key = key.replace(f'model.{layer_idx}.', '')
                    try:
                        self.backbone[layer_idx].load_state_dict({new_key: value}, strict=False)
                        backbone_loaded += 1
                    except:
                        pass

        # 載入 Experts (Layer 29-77)，所有專家用相同權重
        expert_state = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                parts = key.split('.')
                layer_idx = int(parts[1])

                if layer_idx >= 29:
                    new_idx = layer_idx - 29
                    new_key = key.replace(f'model.{layer_idx}.', '')
                    expert_state[(new_idx, new_key)] = value

        expert_loaded = 0
        for i, expert in enumerate(self.experts):
            for (layer_idx, param_key), value in expert_state.items():
                try:
                    expert[layer_idx].load_state_dict({param_key: value}, strict=False)
                    expert_loaded += 1
                except:
                    pass

        if verbose:
            print(f"Loaded pretrained weights from {weights_path}")
            print(f"  Backbone layers: {backbone_loaded}")
            print(f"  Expert layers: {expert_loaded // self.num_experts} (per expert)")

    def info(self, verbose=False):
        """印出模型資訊"""
        n_params = sum(p.numel() for p in self.parameters())
        n_backbone = sum(p.numel() for p in self.backbone.parameters())
        n_router = sum(p.numel() for p in self.router.parameters())
        n_expert = sum(p.numel() for p in self.experts[0].parameters())

        print(f"MoE-YOLOv7 Summary:")
        print(f"  Total params: {n_params:,}")
        print(f"  Backbone: {n_backbone:,}")
        print(f"  Router: {n_router:,}")
        print(f"  Expert (x{self.num_experts}): {n_expert:,} each")
        print(f"  Top-K: {self.top_k}")
        print(f"  Stride: {self.stride.tolist()}")
