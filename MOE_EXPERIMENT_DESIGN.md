# MoE-YOLOv7-tiny 實驗設計

## 實驗目標

在 YOLOv7-tiny 上實現 Mixture of Experts (MoE) 架構，透過 Router 動態選擇最適合的專家來處理不同的輸入圖片，目標 mAP@0.5 達到 **45%**（Baseline OTA: 41.4%）。

---

## 架構設計

### 整體架構圖

```
                        ┌─────────────────┐
                        │   Input Image   │
                        │   (320×320)     │
                        └────────┬────────┘
                                 │
                                 ▼
                ┌────────────────────────────────┐
                │                                │
                │      Backbone (共享)           │
                │      Layers 0-28               │
                │                                │
                │   輸出: P3(14), P4(21), P5(28) │
                └────────────────┬───────────────┘
                                 │
                                 │ [P3, P4, P5] 特徵
                                 │
                ┌────────────────┼────────────────┐
                │                ▼                │
                │   ┌────────────────────────┐    │
                │   │   Router (新增模組)    │    │
                │   │                        │    │
                │   │   Input: P5 (layer 28) │    │
                │   │   - Global AvgPool     │    │
                │   │   - FC → 4 scores      │    │
                │   │   - Softmax            │    │
                │   │   - Top-2 Selection    │    │
                │   │   - Load Balance Loss  │    │
                │   └───────────┬────────────┘    │
                │               │                 │
                └───────────────┼─────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│   Expert 0    │       │   Expert 1    │       │   Expert 2    │   Expert 3
│               │       │               │       │               │
│ ┌───────────┐ │       │ ┌───────────┐ │       │ ┌───────────┐ │
│ │   Neck    │ │       │ │   Neck    │ │       │ │   Neck    │ │
│ │  (29-73)  │ │       │ │  (29-73)  │ │       │ │  (29-73)  │ │
│ └─────┬─────┘ │       │ └─────┬─────┘ │       │ └─────┬─────┘ │
│       ▼       │       │       ▼       │       │       ▼       │
│ ┌───────────┐ │       │ ┌───────────┐ │       │ ┌───────────┐ │
│ │   Head    │ │       │ │   Head    │ │       │ │   Head    │ │
│ │  (74-77)  │ │       │ │  (74-77)  │ │       │ │  (74-77)  │ │
│ └─────┬─────┘ │       │ └─────┬─────┘ │       │ └─────┬─────┘ │
└───────┼───────┘       └───────┼───────┘       └───────┼───────┘
        │                       │                       │
        │ Pred_0 (w0)           │ Pred_1 (w1)           X (未選中)
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────────────────┐
        │   訓練: 加權 Loss                  │
        │   L = w0×L0 + w1×L1 + α×aux_loss  │
        ├───────────────────────────────────┤
        │   推論: WBF 合併                   │
        │   Weighted Boxes Fusion           │
        └───────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │    Final Detection    │
        │    (80 classes)       │
        └───────────────────────┘
```

---

## YOLOv7-tiny 層數分布

| 部分 | 層數範圍 | 層數 | MoE 版本 |
|------|----------|------|----------|
| **Backbone** | 0-28 | 29 層 | 共享 1 份 |
| **Neck** | 29-73 | 45 層 | × 4 專家 |
| **Head** | 74-77 | 4 層 | × 4 專家 |
| **Router** | - | - | 新增 |

### Backbone 關鍵輸出層

| Layer | 名稱 | Stride | 用途 |
|-------|------|--------|------|
| 14 | P3 | 8 | Neck skip connection |
| 21 | P4 | 16 | Neck skip connection |
| 28 | P5 | 32 | Neck 輸入 + Router 輸入 |

---

## 檔案結構

```
yolov7_3_moe/
├── MOE_EXPERIMENT_DESIGN.md   # 本文件
├── CLAUDE.md                   # Claude Code 指引
├── progress.md                 # 進度追蹤
│
├── models/
│   ├── yolo.py                # 原始模型 (不修改)
│   ├── common.py              # 原始模組 (不修改)
│   ├── moe.py                 # Router 類別 (新增) ★
│   ├── moe_yolo.py            # MoE-YOLOv7 類別 (新增) ★
│   └── wbf.py                 # WBF 合併工具 (新增) ★
│
├── utils/
│   ├── loss.py                # 原始 Loss (不修改)
│   └── loss_moe.py            # MoE Loss 計算 (新增) ★
│
├── train_moe.py               # MoE 訓練腳本 (新增) ★
├── detect_moe.py              # MoE 推論腳本 (新增) ★
│
├── cfg/
│   └── training/
│       └── yolov7-tiny.yaml   # 原始配置
│
└── runs/
    └── train/
        └── moe_exp1/          # 實驗結果
```

---

## 核心模組實作

### 1. Router (`models/moe.py`)

Router 根據 Backbone P5 輸出選擇 Top-K 專家，並計算 Load Balancing Loss 防止專家偏好。

```python
class Router(nn.Module):
    """
    帶 Load Balancing 的 Top-K 路由器

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

        # 儲存 aux_loss 和統計
        self.aux_loss = None
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
        # =============================================

        expert_mask = F.one_hot(top_indices, self.num_experts)  # [B, top_k, N]
        expert_mask = expert_mask.sum(dim=1)  # [B, N]

        f = expert_mask.float().mean(dim=0)  # 每個專家被選的比例
        P = probs.mean(dim=0)                 # 每個專家的平均概率

        self.aux_loss = self.num_experts * (f * P).sum()

        return top_indices, top_weights
```

### Router 運作示意

```
Backbone P5 輸出
      │
      │  [B, 512, 10, 10]  (320×320 輸入時)
      ▼
┌─────────────────────────┐
│  Global Average Pool    │
│  [B, 512, 10, 10] → [B, 512]
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  FC Layer (可學習參數)   │
│  [B, 512] → [B, 4]      │
└───────────┬─────────────┘
            │
            │  raw scores: [0.8, 1.2, 0.3, 0.5]
            ▼
┌─────────────────────────┐
│  Softmax                │
│  → [0.25, 0.37, 0.15, 0.23]
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Top-2 Selection        │
│                         │
│  選中: Expert 1 (0.37)  │
│        Expert 0 (0.25)  │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Normalize              │
│                         │
│  w0 = 0.25/(0.37+0.25) = 0.40  │
│  w1 = 0.37/(0.37+0.25) = 0.60  │
└─────────────────────────┘
```

---

### 2. MoE-YOLOv7 主模型 (`models/moe_yolo.py`)

```python
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
        self.router = Router(
            in_channels=512,
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

    def forward(self, x, augment=False, profile=False):
        """
        Forward pass

        訓練時: 返回 dict，包含各專家的預測和權重
        推論時: 返回 dict，包含各專家的預測 (後續用 WBF 合併)

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

        # 找出所有被選中的專家 (去重)
        unique_experts = top_indices.unique().tolist()

        for expert_idx in unique_experts:
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

    def load_pretrained(self, weights_path, verbose=True):
        """
        載入預訓練權重

        所有專家用相同的預訓練權重初始化
        """
        # ... (載入 backbone 和 experts 權重)
```

---

### 3. WBF 合併工具 (`models/wbf.py`)

WBF (Weighted Boxes Fusion) 與 NMS 不同，會融合重疊的框而非直接丟棄，特別適合 MoE 架構。

```python
def weighted_boxes_fusion(
    boxes_list,      # 每個專家的 boxes [N, 4]
    scores_list,     # 每個專家的 scores [N]
    labels_list,     # 每個專家的 labels [N]
    weights=None,    # 每個專家的權重
    iou_thr=0.55,    # IoU 閾值
    skip_box_thr=0.0,
    conf_type='avg'
):
    """
    Weighted Boxes Fusion

    與 NMS 的差異:
    - NMS: 刪除重疊框，只保留最高分的
    - WBF: 融合重疊框，取加權平均

    示例:
    Expert 0: box=[100, 100, 50, 50], conf=0.9
    Expert 1: box=[105, 98, 52, 48], conf=0.85

    NMS 結果: box=[100, 100, 50, 50] (只保留 Expert 0)
    WBF 結果: box=[102, 99, 51, 49] (融合兩個框)
    """
    # ... 實作細節見 models/wbf.py
```

### WBF 合併流程

```
Expert 0 輸出          Expert 1 輸出
    │                      │
    ▼                      ▼
┌──────────┐          ┌──────────┐
│ NMS 過濾 │          │ NMS 過濾 │
│ (各自做)  │          │ (各自做)  │
└────┬─────┘          └────┬─────┘
     │                     │
     │  boxes_0            │  boxes_1
     │  scores_0           │  scores_1
     │  labels_0           │  labels_1
     │                     │
     └──────────┬──────────┘
                │
                ▼
        ┌───────────────┐
        │  WBF 合併     │
        │               │
        │ 1. 按類別分組  │
        │ 2. 計算 IoU   │
        │ 3. 融合重疊框  │
        │ 4. 加權平均   │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐
        │  最終檢測結果  │
        │  [M, 6]       │
        │  (x1,y1,x2,y2,│
        │   conf, cls)  │
        └───────────────┘
```

---

### 4. MoE Loss 計算 (`utils/loss_moe.py`)

```python
class ComputeLossMoESimple:
    """
    MoE-YOLOv7 的 Loss 計算器

    訓練流程:
    1. 每個被選中的專家分別計算 loss
    2. 根據 router 權重加權合併
    3. 加上 Load Balancing Loss

    Total Loss = Σ(w_i * expert_i_loss) + λ * aux_loss
    """

    def __init__(self, model, autobalance=False, aux_loss_weight=0.01):
        self.aux_loss_weight = aux_loss_weight
        self.model = model

    def __call__(self, moe_output, targets):
        """
        計算 MoE Loss

        Args:
            moe_output: dict from MoEYOLOv7.forward()
            targets: [N, 6] (image_idx, class, x, y, w, h)

        Returns:
            loss: scalar, 總 loss
            loss_items: tensor [4], (box_loss, obj_loss, cls_loss, total_loss)
        """
        expert_outputs = moe_output['expert_outputs']
        top_indices = moe_output['top_indices']
        top_weights = moe_output['top_weights']
        aux_loss = moe_output.get('aux_loss', 0)

        # 計算每個專家的平均權重
        expert_avg_weights = {}
        for expert_idx in expert_outputs.keys():
            mask = (top_indices == expert_idx)
            if mask.sum() > 0:
                avg_weight = top_weights[mask].mean()
                expert_avg_weights[expert_idx] = avg_weight

        # 計算每個專家的 loss 並加權
        total_loss = 0
        for expert_idx, pred in expert_outputs.items():
            weight = expert_avg_weights[expert_idx]
            loss, _ = self._compute_expert_loss(pred, targets, expert_idx)
            total_loss += weight * loss

        # 加上 aux loss
        total_loss = total_loss + self.aux_loss_weight * aux_loss

        return total_loss, loss_items
```

---

## Load Balancing Loss

### 問題：Expert Collapse

沒有約束時，Router 容易學會只選擇少數專家：

```
Expert 使用分布 (無 Load Balancing):
┌────────────────────────────────────────┐
│ Expert 0: ████████████████████ 45%     │
│ Expert 1: ████████████████     40%     │
│ Expert 2: ██                   10%     │ ← 幾乎沒用到
│ Expert 3: █                    5%      │ ← 幾乎沒用到
└────────────────────────────────────────┘
```

### 解決方案

來自 Switch Transformer (Google, 2021)：

```
Load Balancing Loss = α × N × Σᵢ (fᵢ × Pᵢ)

其中：
- N = 專家數量 (4)
- fᵢ = 第 i 個專家被選中的比例 (實際分配)
- Pᵢ = 第 i 個專家的平均路由概率 (Router softmax 輸出)
- α = loss 權重 (建議 0.01)
```

### 預期效果

```
Expert 使用分布 (有 Load Balancing):
┌────────────────────────────────────────┐
│ Expert 0: █████████████       27%      │
│ Expert 1: ████████████        24%      │
│ Expert 2: ████████████        23%      │
│ Expert 3: ████████████████    26%      │
└────────────────────────────────────────┘
            ↑ 大致平衡
```

---

## 訓練與推論流程

### 訓練流程

```
輸入圖片 [B, 3, 320, 320]
         │
         ▼
┌─────────────────────────┐
│     Backbone (共享)      │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│     Router              │
│   → top_indices [B, 2]  │
│   → top_weights [B, 2]  │
│   → aux_loss            │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  執行選中的專家          │
│  (只執行被選中的 2 個)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  計算 Loss              │
│                         │
│  L_expert_0 = YOLOLoss(pred_0, targets)
│  L_expert_1 = YOLOLoss(pred_1, targets)
│                         │
│  L_total = w0*L0 + w1*L1 + α*aux_loss
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Backward + Optimize    │
└─────────────────────────┘
```

### 推論流程

```
輸入圖片
    │
    ▼
Backbone → Router → 選擇 Expert 0, 1
    │
    ▼
┌──────────────────────────────────────┐
│  Expert 0              Expert 1      │
│     │                     │          │
│     ▼                     ▼          │
│   NMS                   NMS          │
│     │                     │          │
│  boxes_0              boxes_1        │
└──────────────────────────────────────┘
    │                       │
    └───────────┬───────────┘
                │
                ▼
        ┌───────────────┐
        │     WBF       │
        │   (合併框)     │
        └───────┬───────┘
                │
                ▼
          最終檢測結果
```

---

## 風險分析與潛在問題

### 風險 1: 輸出合併的語義問題 ⚠️ 高風險

**問題：** 直接對 bounding box 加權平均可能產生無意義的結果。

**解決方案：** 使用 WBF (Weighted Boxes Fusion)
- 各專家先各自做 NMS
- 只合併 IoU > 閾值的框
- 不同物體的預測不會被錯誤合併

### 風險 2: 梯度稀疏問題 ⚠️ 中風險

**問題：** 未被選中的專家不會收到梯度。

**解決方案：**
- Load Balancing Loss（已加入）
- 增大 batch size
- 考慮 top_k=3

### 風險 3: 訓練不穩定 ⚠️ 中風險

**問題：** Router 和 Experts 同時訓練可能不穩定。

**解決方案：**
- 使用預訓練權重初始化所有專家
- 可選：先凍結 Router 訓練專家

### 風險 4: 專家同質化 ⚠️ 中風險

**問題：** 所有專家從相同權重初始化，可能訓練後仍高度相似。

**解決方案：**
- 增加訓練時間讓專家分化
- 監控專家輸出的相似度

---

## 超參數設定

| 參數 | 建議值 | 說明 |
|------|--------|------|
| `num_experts` | 4 | 專家數量 |
| `top_k` | 2 | 每張圖選幾個專家 |
| `aux_loss_weight` | 0.01 | Load Balancing Loss 權重 α |
| `epochs` | 100-500 | 訓練輪數 |
| `batch_size` | 16-64 | 越大越好，確保專家都被選到 |
| `img_size` | 320 | 輸入圖片尺寸 |
| `wbf_iou_thres` | 0.55 | WBF IoU 閾值 |

---

## 使用指令

### 訓練

```bash
python train_moe.py \
    --data data/coco.yaml \
    --cfg cfg/training/yolov7-tiny.yaml \
    --weights yolov7-tiny.pt \
    --batch-size 16 \
    --epochs 100 \
    --num-experts 4 \
    --top-k 2 \
    --aux-loss-weight 0.01 \
    --img-size 320 \
    --project runs/train \
    --name moe_exp1
```

### 推論

```bash
python detect_moe.py \
    --weights runs/train/moe_exp1/weights/best.pt \
    --source data/images \
    --img-size 320 \
    --conf-thres 0.25 \
    --wbf-iou-thres 0.55 \
    --project runs/detect \
    --name moe_exp1
```

---

## 監控指標

| 指標 | 健康範圍 | 說明 |
|------|----------|------|
| Expert 使用率 | 各 20-30% | 過於不均表示 Load Balancing 失效 |
| Load Balance Loss | 0.5-2.0 | 過低可能過擬合，過高表示不平衡 |
| 專家輸出相似度 | < 0.9 | 過高表示專家沒有分化 |

---

## 預期結果

### 評估指標

| 指標 | Baseline (OTA) | MoE (目標) |
|------|----------------|------------|
| mAP@0.5 | 41.4% | **45%** |
| mAP@0.5:0.95 | 25.1% | - |
| 參數量 | 6M | ~25M (1 backbone + 4 experts) |

### 假設

- Router 會學習根據圖片內容選擇合適的專家
- 不同專家可能會專精於不同類型的物件/場景
- WBF 合併後的效果應該優於單一模型

---

## 與之前 4-Head 實驗的差異

| 項目 | 4-Head 實驗 | MoE 實驗 |
|------|------------|----------|
| 類別分配 | 人工分配 20 類/head | Router 自動學習 |
| 專家數量 | 4 個 Head | 4 個 Neck+Head |
| 選擇機制 | 全部執行，按類別篩選 | 只執行 Top-2 |
| 輸出合併 | 按類別拼接 | WBF 合併 |
| 訓練方式 | 分別訓練各 head | 端到端訓練 |

---

## 參考資料

- [Mixture of Experts (MoE)](https://arxiv.org/abs/1701.06538) - MoE 原始論文
- [Switch Transformer](https://arxiv.org/abs/2101.03961) - Load Balancing Loss 來源
- [Weighted Boxes Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) - WBF 實作參考
- [YOLOv7 Official Repo](https://github.com/WongKinYiu/yolov7)

---

## 更新日誌

| 日期 | 更新內容 |
|------|----------|
| 2025-01-19 | 完成所有程式碼實作 |
| 2025-01-19 | 整合完整實驗文件 |
| 2025-01-18 | 加入 WBF 合併設計 |
| 2025-01-18 | 加入 Load Balancing Loss 設計 |
| 2025-01-18 | 加入風險分析 (6 項) |
| 2025-01-18 | 建立實驗設計文件 |
