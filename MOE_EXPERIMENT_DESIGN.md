# MoE-YOLOv7-tiny 實驗設計

## 實驗目標

在 YOLOv7-tiny 上實現簡單的 Mixture of Experts (MoE) 架構，透過 Router 動態選擇最適合的專家來處理不同的輸入圖片。

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
                │   │   - Top-2 Selection    │    │
                │   │                        │    │
                │   └───────────┬────────────┘    │
                │               │                 │
                └───────────────┼─────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│   Expert 1    │       │   Expert 2    │       │   Expert 3    │   Expert 4
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
        │ Pred_1                │ Pred_2                X (未選中)
        │ (w1)                  │ (w2)
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   Weighted Average    │
        │                       │
        │   w1×Pred_1 + w2×Pred_2   │
        │   ─────────────────────   │
        │        w1 + w2            │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │    Final Output       │
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

## Router 設計

### 功能
根據 Backbone 輸出的特徵，決定選擇哪 2 個專家來處理這張圖片。

### 架構

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
│  選中: Expert 2 (0.37)  │
│        Expert 1 (0.25)  │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Normalize              │
│                         │
│  w1 = 0.37/(0.37+0.25) = 0.60  │
│  w2 = 0.25/(0.37+0.25) = 0.40  │
└─────────────────────────┘
```

### 程式碼

```python
class Router(nn.Module):
    """
    簡單的 Top-K 路由器
    根據 Backbone 輸出選擇最適合的專家
    """
    def __init__(self, in_channels=512, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        """
        Args:
            x: Backbone P5 輸出 [B, C, H, W]
        Returns:
            top_indices: 選中的專家索引 [B, top_k]
            top_weights: 歸一化後的權重 [B, top_k]
        """
        # Global Average Pooling
        x = self.pool(x).flatten(1)  # [B, C]

        # 計算每個專家的分數
        scores = self.fc(x)  # [B, num_experts]

        # Softmax 轉換為概率
        probs = F.softmax(scores, dim=-1)  # [B, num_experts]

        # 選擇 Top-K 專家
        top_weights, top_indices = probs.topk(self.top_k, dim=-1)

        # 歸一化權重 (讓 Top-K 權重總和為 1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        return top_indices, top_weights
```

---

## 專家 (Expert) 設計

### 結構
每個專家包含完整的 Neck + Head：
- Neck (Layer 29-73): SPPCSPC + FPN/PAN
- Head (Layer 74-77): IDetect

### 初始化方式
1. 從訓練好的 YOLOv7-tiny baseline 複製 Neck + Head
2. 4 個專家初始權重相同
3. 訓練時各自分化

---

## 輸出合併

### 方法
加權平均被選中的 2 個專家的輸出。

### 公式

```
Final = (w1 × Pred_1 + w2 × Pred_2) / (w1 + w2)
```

由於權重已經歸一化 (w1 + w2 = 1)，實際上：

```
Final = w1 × Pred_1 + w2 × Pred_2
```

### 程式碼

```python
def merge_expert_outputs(predictions, weights):
    """
    合併多個專家的輸出

    Args:
        predictions: List of [B, N, 85] 每個專家的預測
        weights: [B, K] 歸一化後的權重

    Returns:
        merged: [B, N, 85] 合併後的預測
    """
    # weights: [B, K] -> [B, K, 1, 1]
    weights = weights.unsqueeze(-1).unsqueeze(-1)

    # Stack predictions: [K, B, N, 85] -> [B, K, N, 85]
    stacked = torch.stack(predictions, dim=1)

    # Weighted sum
    merged = (stacked * weights).sum(dim=1)

    return merged
```

---

## 設計原則

### 不修改原始架構
- 不改動 `models/yolo.py`
- 不改動 `models/common.py`
- 用 wrapper 方式包裝原始模型

### 最簡單實現
- Router 只用 GlobalAvgPool + FC
- 不加額外的 load balancing loss
- 不加 auxiliary loss

### 保持相容性
- 輸出格式與原始 YOLOv7 相同
- 可以使用原始的 loss function
- 可以使用原始的訓練腳本 (微調)

---

## 實作步驟

### Phase 1: 環境準備
1. [ ] 複製 yolov7_1 基礎程式碼到 yolov7_3_moe
2. [ ] 確認 baseline 模型可以正常運行
3. [ ] 準備 COCO 320×320 資料集

### Phase 2: 實作 MoE 模組
1. [ ] 建立 `models/moe.py` - Router 類別
2. [ ] 建立 `models/moe_yolo.py` - MoE-YOLOv7 類別
3. [ ] 實作 Expert wrapper (複製 Neck+Head)
4. [ ] 實作輸出合併邏輯

### Phase 3: 訓練腳本
1. [ ] 建立 `train_moe.py` - MoE 訓練腳本
2. [ ] 修改 loss 計算 (如需要)
3. [ ] 加入 Router 選擇的 logging

### Phase 4: 訓練與評估
1. [ ] 訓練 MoE-YOLOv7-tiny (100 epochs)
2. [ ] 評估 mAP 並與 baseline 比較
3. [ ] 分析 Router 的選擇行為

---

## 預期結果

### 假設
- Router 會學習根據圖片內容選擇合適的專家
- 不同專家可能會專精於不同類型的物件/場景
- 合併後的效果應該優於單一模型

### 評估指標
| 指標 | Baseline (OTA) | MoE (目標) |
|------|----------------|------------|
| mAP@0.5 | 41.4% | **45%** |
| mAP@0.5:0.95 | 25.1% | - |
| 推論速度 | 1x | ~2x (選 2/4 專家) |
| 參數量 | 6M | ~25M (1 backbone + 4 experts) |

---

## 與之前 4-Head 實驗的差異

| 項目 | 4-Head 實驗 | MoE 實驗 |
|------|------------|----------|
| 類別分配 | 人工分配 20 類/head | Router 自動學習 |
| 專家數量 | 4 個 Head | 4 個 Neck+Head |
| 選擇機制 | 全部執行，按類別篩選 | 只執行 Top-2 |
| 輸出合併 | 按類別拼接 | 加權平均 |
| 訓練方式 | 分別訓練各 head | 端到端訓練 |

---

## 檔案結構

```
yolov7_3_moe/
├── MOE_EXPERIMENT_DESIGN.md   # 本文件
├── CLAUDE.md                   # Claude Code 指引
├── progress.md                 # 進度追蹤
│
├── cfg/
│   └── training/
│       └── yolov7-tiny.yaml   # 原始配置
│
├── models/
│   ├── yolo.py                # 原始模型 (不修改)
│   ├── common.py              # 原始模組 (不修改)
│   ├── moe.py                 # Router 類別 (新增)
│   └── moe_yolo.py            # MoE-YOLOv7 類別 (新增)
│
├── train_moe.py               # MoE 訓練腳本 (新增)
├── test_moe.py                # MoE 測試腳本 (新增)
│
└── runs/
    └── train/
        └── moe_exp1/          # 實驗結果
```

---

## 參考資料

- [Mixture of Experts (MoE)](https://arxiv.org/abs/1701.06538)
- [Switch Transformer](https://arxiv.org/abs/2101.03961)
- [YOLOv7 Official Repo](https://github.com/WongKinYiu/yolov7)

---

## 更新日誌

| 日期 | 更新內容 |
|------|----------|
| 2025-01-18 | 建立實驗設計文件 |
