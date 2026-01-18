"""
MoE-YOLOv7: Mixture of Experts 版本的 YOLOv7

架構:
- 共享 Backbone (Layer 0-28)
- Router 選擇 Top-K 專家
- N 個專家 (Neck + Head)
- WBF 合併輸出 (推論時)

訓練: 各專家分別計算 loss，加權總和
推論: 各專家分別 NMS，用 WBF 合併
"""

import torch
import torch.nn as nn
from copy import deepcopy
from pathlib import Path

from models.yolo import Model
from models.moe import Router


class MoEYOLOv7(nn.Module):
    """
    MoE-YOLOv7 使用 WBF 合併

    Args:
        cfg: 模型配置檔路徑
        num_experts: 專家數量
        top_k: 每張圖選幾個專家
        nc: 類別數量
        ch: 輸入通道數
    """
    def __init__(self, cfg='cfg/training/yolov7-tiny.yaml',
                 num_experts=4, top_k=2, nc=80, ch=3):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.nc = nc

        # 載入基礎模型取得結構
        base_model = Model(cfg, ch=ch, nc=nc)

        # ============ Backbone (共享) ============
        # YOLOv7-tiny: Layer 0-28
        backbone_layers = list(base_model.model)[:29]
        self.backbone = nn.ModuleList(backbone_layers)

        # 需要保存輸出的層索引 (用於 skip connection)
        self.backbone_save = [i for i in base_model.save if i < 29]

        # ============ Router ============
        # P5 輸出通道數: 512 for YOLOv7-tiny
        # 從 cfg 讀取或使用預設值
        p5_channels = 512
        self.router = Router(
            in_channels=p5_channels,
            num_experts=num_experts,
            top_k=top_k
        )

        # ============ Experts (Neck + Head) ============
        # YOLOv7-tiny: Layer 29-77
        expert_layers = list(base_model.model)[29:]
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            self.experts.append(nn.ModuleList(deepcopy(expert_layers)))

        # 專家需要保存輸出的層索引
        self.expert_save = [i for i in base_model.save if i >= 29]

        # ============ 保存必要資訊 ============
        self.save = base_model.save
        self.stride = base_model.stride
        self.names = base_model.names
        self.yaml = base_model.yaml

        # 用於 loss 計算
        self.gr = 1.0  # giou loss ratio
        self.hyp = None  # 會在訓練時設定

    def forward(self, x, augment=False, profile=False):
        """
        Forward pass

        訓練時: 返回 dict，包含各專家的預測和權重
        推論時: 返回 dict，包含各專家的預測 (後續用 WBF 合併)

        Args:
            x: 輸入圖片 [B, 3, H, W]

        Returns:
            dict:
                - expert_outputs: {expert_idx: prediction}
                - top_indices: [B, top_k]
                - top_weights: [B, top_k]
                - aux_loss: scalar (僅訓練時)
        """
        # 1. Backbone forward
        backbone_cache = self._backbone_forward(x)

        # 2. Router 選擇專家
        p5 = backbone_cache[28]  # P5 特徵
        top_indices, top_weights = self.router(p5)

        # 3. 執行被選中的專家
        expert_outputs = {}
        batch_size = x.shape[0]

        # 找出所有被選中的專家 (去重)
        unique_experts = top_indices.unique().tolist()

        for expert_idx in unique_experts:
            # 執行這個專家 (對整個 batch)
            expert_out = self._expert_forward(
                self.experts[expert_idx],
                backbone_cache
            )
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

    def _backbone_forward(self, x):
        """
        Backbone forward pass

        Args:
            x: 輸入 [B, 3, H, W]

        Returns:
            cache: dict，包含關鍵層的輸出 {layer_idx: tensor}
        """
        cache = {}
        y = []

        for i, m in enumerate(self.backbone):
            # 處理 skip connection
            if m.f != -1:
                if isinstance(m.f, int):
                    x = y[m.f]
                else:
                    x = [x if j == -1 else y[j] for j in m.f]

            # Forward
            x = m(x)

            # 保存輸出
            if i in self.backbone_save:
                y.append(x)
            else:
                y.append(None)

            # 保存關鍵層 (用於 Neck 的 skip connection)
            if i in [14, 21, 28]:
                cache[i] = x

        return cache

    def _expert_forward(self, expert_layers, backbone_cache):
        """
        單個專家的 forward pass

        Args:
            expert_layers: nn.ModuleList，專家的層
            backbone_cache: dict，Backbone 輸出

        Returns:
            專家的預測輸出
        """
        # 初始化 cache，包含 backbone 的關鍵輸出
        # 使用 dict 存儲，key 是絕對層索引
        y = {14: backbone_cache[14], 21: backbone_cache[21], 28: backbone_cache[28]}
        x = backbone_cache[28]  # 從 P5 開始

        for i, m in enumerate(expert_layers):
            layer_idx = i + 29  # 實際層索引 (絕對索引)

            # 處理 skip connection
            if m.f != -1:
                if isinstance(m.f, int):
                    # 單一來源
                    if m.f < 0:
                        # 負數 = 相對索引，轉換為絕對索引
                        # 例如: layer_idx=36, m.f=-7 → 36-7=29
                        abs_idx = layer_idx + m.f
                    else:
                        # 正數 = 絕對索引
                        abs_idx = m.f
                    x = y[abs_idx]
                else:
                    # 多個來源 (Concat)
                    inputs = []
                    for j in m.f:
                        if j == -1:
                            # -1 表示前一層的輸出 (當前的 x)
                            inputs.append(x)
                        elif j < 0:
                            # 負數相對索引
                            abs_idx = layer_idx + j
                            inputs.append(y[abs_idx])
                        else:
                            # 正數絕對索引
                            inputs.append(y[j])
                    x = inputs

            # Forward
            x = m(x)

            # 保存所有層的輸出 (skip connection 需要)
            y[layer_idx] = x

        return x

    def load_pretrained(self, weights_path, verbose=True):
        """
        載入預訓練權重

        Args:
            weights_path: 權重檔案路徑
            verbose: 是否印出訊息
        """
        if not Path(weights_path).exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)

        # 處理不同格式的 checkpoint
        if 'model' in ckpt:
            state_dict = ckpt['model'].float().state_dict()
        else:
            state_dict = ckpt

        # ============ 載入 Backbone ============
        backbone_loaded = 0
        for k, v in state_dict.items():
            if k.startswith('model.'):
                parts = k.split('.')
                layer_idx = int(parts[1])

                if layer_idx < 29:
                    # Backbone 層
                    new_key = '.'.join(parts[1:])
                    try:
                        self.backbone[layer_idx].load_state_dict(
                            {'.'.join(parts[2:]): v}, strict=False
                        )
                        backbone_loaded += 1
                    except:
                        pass

        # ============ 載入 Experts ============
        # 所有專家用相同的預訓練權重初始化
        expert_state = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                parts = k.split('.')
                layer_idx = int(parts[1])

                if layer_idx >= 29:
                    new_idx = layer_idx - 29
                    new_key = '.'.join([str(new_idx)] + parts[2:])
                    expert_state[new_key] = v

        experts_loaded = 0
        for i, expert in enumerate(self.experts):
            for k, v in expert_state.items():
                parts = k.split('.')
                layer_idx = int(parts[0])
                try:
                    expert[layer_idx].load_state_dict(
                        {'.'.join(parts[1:]): v}, strict=False
                    )
                    experts_loaded += 1
                except:
                    pass

        if verbose:
            print(f"Loaded pretrained weights from {weights_path}")
            print(f"  Backbone: {backbone_loaded} parameters")
            print(f"  Experts: {experts_loaded // self.num_experts} parameters each")

    def get_expert_usage_stats(self):
        """
        取得專家使用統計

        Returns:
            dict: 專家使用相關統計
        """
        if self.router.last_probs is not None:
            probs = self.router.last_probs.mean(dim=0)
            return {
                'mean_probs': probs.cpu().numpy(),
                'aux_loss': self.router.aux_loss.item() if self.router.aux_loss else 0
            }
        return None

    def fuse(self):
        """Fuse Conv2d + BatchNorm2d layers"""
        print('Fusing layers... ')
        for module in [self.backbone] + list(self.experts):
            for m in module:
                if hasattr(m, 'fuse'):
                    m.fuse()
        return self

    def info(self, verbose=False):
        """Print model information"""
        from utils.torch_utils import model_info
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
