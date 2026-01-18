# MoE-YOLOv7 監督式路由訓練方案

## 方案概述

與原方案（讓 Router 自動學習分配）不同，本方案採用**監督式預訓練 Router**：
1. 人為定義每個專家的專長（負責哪些類別）
2. 根據圖片內容，預先標註應該分配給哪些專家
3. 先單獨訓練 Router，讓它學會正確分配
4. 再訓練專家，每個專家從一開始就專精於特定類別

---

## 與原方案比較

| 項目 | 原方案 (自動學習) | 本方案 (監督式) |
|------|------------------|----------------|
| Router 訓練 | 隨 Detection Loss 一起學 | **先單獨用分類 Loss 訓練** |
| 專家分工 | 自動形成（不可控） | **人為定義（可控）** |
| 專家何時開始分化 | 訓練中逐漸分化 | **從第一個 epoch 就專精** |
| 可控性 | 低 | **高** |
| 訓練穩定性 | 可能不穩定 | **更穩定** |
| 風險 | 專家可能同質化 | 受限於人為定義的分配 |

---

## 架構圖

```
┌─────────────────────────────────────────────────────────────────┐
│                        訓練 Phase 1                              │
│                      (Router 預訓練)                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    輸入圖片                                                      │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────┐                                                │
│   │  Backbone   │  ← 凍結 (使用預訓練權重)                        │
│   │  (0-28層)   │                                                │
│   └──────┬──────┘                                                │
│          │ P5 特徵                                               │
│          ▼                                                       │
│   ┌─────────────┐      ┌─────────────────────────┐               │
│   │   Router    │ ──── │  BCE Loss               │               │
│   │  (訓練中)    │      │  預測 vs 預期專家標籤    │               │
│   └─────────────┘      └─────────────────────────┘               │
│                                                                  │
│   目標: Router 學會「看到人選 Expert 0」「看到車選 Expert 1」...   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

                              │
                              ▼

┌─────────────────────────────────────────────────────────────────┐
│                        訓練 Phase 2                              │
│                       (專家訓練)                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    輸入圖片                                                      │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────┐                                                │
│   │  Backbone   │  ← 凍結或低學習率                               │
│   └──────┬──────┘                                                │
│          │                                                       │
│          ▼                                                       │
│   ┌─────────────┐                                                │
│   │   Router    │  ← 凍結 (已經學會正確分配)                      │
│   │  選擇專家    │                                                │
│   └──────┬──────┘                                                │
│          │                                                       │
│    ┌─────┴─────┐                                                 │
│    ▼           ▼                                                 │
│ Expert 0    Expert 1    (只有被選中的專家會訓練)                   │
│ (學人物)    (學車輛)                                              │
│    │           │                                                 │
│    ▼           ▼                                                 │
│ Detection Loss (YOLO Loss)                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 專家分工定義

### 方案 A: 4 專家 × 20 類

沿用之前 4-Head 實驗的分類方式：

```python
# 專家負責的類別
EXPERT_CLASSES = {
    0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    # Expert 0: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light,
    #           fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow

    1: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
    # Expert 1: elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee,
    #           skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle

    2: [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    # Expert 2: wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
    #           broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed

    3: [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
    # Expert 3: dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven,
    #           toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
}
```

### 方案 B: 按語義分類

```python
# 更有意義的語義分類
EXPERT_CLASSES_SEMANTIC = {
    0: [0],  # Expert 0: 人類 (person)

    1: [1, 2, 3, 4, 5, 6, 7, 8],  # Expert 1: 交通工具
    # bicycle, car, motorcycle, airplane, bus, train, truck, boat

    2: [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # Expert 2: 動物
    # bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

    3: [...]  # Expert 3: 其他物品
}
```

---

## Router 標籤生成

### 標籤格式

對於每張圖片，生成一個多標籤向量，表示應該選擇哪些專家：

```
圖片中有: person, car, dog
         ↓
類別 ID: [0, 2, 16]
         ↓
查表:
  - person (0) → Expert 0
  - car (2)    → Expert 0
  - dog (16)   → Expert 0
         ↓
Router 標籤: [1, 0, 0, 0]  (只需要 Expert 0)

---

圖片中有: person, elephant
         ↓
類別 ID: [0, 20]
         ↓
查表:
  - person (0)    → Expert 0
  - elephant (20) → Expert 1
         ↓
Router 標籤: [1, 1, 0, 0]  (需要 Expert 0 和 Expert 1)
```

### 標籤生成程式碼

```python
import json
import numpy as np
from pathlib import Path

# 專家分配表
EXPERT_CLASSES = {
    0: list(range(0, 20)),   # Expert 0: 類別 0-19
    1: list(range(20, 40)),  # Expert 1: 類別 20-39
    2: list(range(40, 60)),  # Expert 2: 類別 40-59
    3: list(range(60, 80)),  # Expert 3: 類別 60-79
}

# 反向查找表: 類別 → 專家
CLASS_TO_EXPERT = {}
for expert_idx, classes in EXPERT_CLASSES.items():
    for cls in classes:
        CLASS_TO_EXPERT[cls] = expert_idx


def get_router_label(image_classes, num_experts=4):
    """
    根據圖片中的類別，生成 Router 標籤

    Args:
        image_classes: list, 這張圖片中有哪些類別 [0, 2, 16]
        num_experts: int, 專家數量

    Returns:
        label: list, [1, 0, 0, 0] 表示應該選 Expert 0
    """
    label = [0] * num_experts

    for cls in image_classes:
        if cls in CLASS_TO_EXPERT:
            expert_idx = CLASS_TO_EXPERT[cls]
            label[expert_idx] = 1

    return label


def generate_router_labels_from_coco(annotation_file, output_file):
    """
    從 COCO 標註檔生成 Router 訓練標籤

    Args:
        annotation_file: COCO 標註檔路徑 (instances_train2017.json)
        output_file: 輸出檔案路徑
    """
    with open(annotation_file, 'r') as f:
        coco = json.load(f)

    # 建立 image_id → annotations 映射
    image_annotations = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann['category_id'])

    # COCO category_id 到 0-79 的映射
    cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(coco['categories'])}

    # 生成標籤
    router_labels = {}
    for img in coco['images']:
        img_id = img['id']
        img_name = img['file_name']

        if img_id in image_annotations:
            # 獲取這張圖的所有類別
            cat_ids = image_annotations[img_id]
            classes = [cat_id_to_idx[cat_id] for cat_id in cat_ids if cat_id in cat_id_to_idx]
            classes = list(set(classes))  # 去重

            # 生成 Router 標籤
            label = get_router_label(classes)
            router_labels[img_name] = {
                'classes': classes,
                'router_label': label
            }

    # 保存
    with open(output_file, 'w') as f:
        json.dump(router_labels, f)

    print(f"Generated {len(router_labels)} router labels")
    print(f"Saved to {output_file}")

    return router_labels


# 使用範例
if __name__ == '__main__':
    generate_router_labels_from_coco(
        'coco/annotations/instances_train2017.json',
        'coco/router_labels_train.json'
    )
```

---

## Phase 1: Router 預訓練

### 訓練腳本

```python
# train_router_pretrain.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm

from models.moe_yolo import MoEYOLOv7


class RouterPretrainDataset(Dataset):
    """
    Router 預訓練資料集
    """
    def __init__(self, image_dir, label_file, img_size=320):
        self.image_dir = image_dir
        self.img_size = img_size

        with open(label_file, 'r') as f:
            self.labels = json.load(f)

        self.image_names = list(self.labels.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        # 載入圖片
        img_path = f"{self.image_dir}/{img_name}"
        img = load_and_preprocess_image(img_path, self.img_size)

        # 載入標籤
        label = torch.tensor(self.labels[img_name]['router_label'], dtype=torch.float32)

        return img, label


class RouterPretrainer:
    """
    Router 預訓練器
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

        # 凍結 Backbone
        for param in model.backbone.parameters():
            param.requires_grad = False

        # 凍結 Experts (Phase 1 不訓練)
        for expert in model.experts:
            for param in expert.parameters():
                param.requires_grad = False

        # 只訓練 Router
        self.optimizer = optim.Adam(model.router.parameters(), lr=0.001)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc='Training Router')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward: 只跑 Backbone + Router
            backbone_cache = self.model._backbone_forward(images)
            p5 = backbone_cache[28]

            # Router forward (取 logits)
            pooled = self.model.router.pool(p5).flatten(1)
            logits = self.model.router.fc(pooled)  # [B, num_experts]

            # Loss
            loss = self.criterion(logits, labels)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 統計
            total_loss += loss.item()

            # 計算準確率 (預測的專家是否包含所有應選的專家)
            pred = (torch.sigmoid(logits) > 0.5).float()
            correct += (pred == labels).all(dim=1).sum().item()
            total += labels.size(0)

            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})

        return total_loss / len(dataloader), correct / total

    def validate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                backbone_cache = self.model._backbone_forward(images)
                p5 = backbone_cache[28]

                pooled = self.model.router.pool(p5).flatten(1)
                logits = self.model.router.fc(pooled)

                pred = (torch.sigmoid(logits) > 0.5).float()
                correct += (pred == labels).all(dim=1).sum().item()
                total += labels.size(0)

        return correct / total


def train_router(epochs=50):
    """
    Router 預訓練主函數
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 建立模型
    model = MoEYOLOv7(
        cfg='cfg/training/yolov7-tiny.yaml',
        num_experts=4,
        top_k=2,
        nc=80
    ).to(device)

    # 載入預訓練權重 (Backbone)
    model.load_pretrained('yolov7-tiny.pt')

    # 資料集
    train_dataset = RouterPretrainDataset(
        image_dir='coco/images/train2017',
        label_file='coco/router_labels_train.json'
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)

    # 訓練器
    trainer = RouterPretrainer(model, device)

    # 訓練
    best_acc = 0
    for epoch in range(epochs):
        loss, acc = trainer.train_epoch(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                'model': model.state_dict(),
                'router': model.router.state_dict(),
                'epoch': epoch,
                'accuracy': acc
            }, 'router_pretrained.pt')
            print(f"  Saved best model (acc: {acc:.4f})")

    print(f"Router pretraining done. Best accuracy: {best_acc:.4f}")


if __name__ == '__main__':
    train_router(epochs=50)
```

---

## Phase 2: 專家訓練

### 訓練腳本修改

```python
# train_moe_phase2.py

def train_experts(pretrained_router_path, epochs=200):
    """
    Phase 2: 使用預訓練的 Router 訓練專家
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 建立模型
    model = MoEYOLOv7(
        cfg='cfg/training/yolov7-tiny.yaml',
        num_experts=4,
        top_k=2,
        nc=80
    ).to(device)

    # 載入預訓練的 Backbone 和 Experts
    model.load_pretrained('yolov7-tiny.pt')

    # 載入預訓練的 Router
    router_ckpt = torch.load(pretrained_router_path)
    model.router.load_state_dict(router_ckpt['router'])
    print(f"Loaded pretrained router (acc: {router_ckpt['accuracy']:.4f})")

    # ============ 凍結策略 ============
    # 選項 A: 凍結 Router
    for param in model.router.parameters():
        param.requires_grad = False

    # 選項 B: 凍結 Backbone (可選)
    # for param in model.backbone.parameters():
    #     param.requires_grad = False

    # ============ 訓練 ============
    # 使用原本的 train_moe.py 訓練邏輯
    # ...
```

---

## 完整訓練流程

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 0: 準備工作                                                │
├─────────────────────────────────────────────────────────────────┤
│  1. 下載 yolov7-tiny.pt                                          │
│  2. 準備 COCO 資料集                                             │
│  3. 執行標籤生成腳本，產生 router_labels_train.json               │
│                                                                  │
│  python generate_router_labels.py                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Router 預訓練 (Phase 1)                                 │
├─────────────────────────────────────────────────────────────────┤
│  • 凍結: Backbone + Experts                                      │
│  • 訓練: Router                                                  │
│  • Loss: BCE (多標籤分類)                                        │
│  • Epochs: 50                                                    │
│  • 輸出: router_pretrained.pt                                    │
│                                                                  │
│  python train_router_pretrain.py --epochs 50                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: 專家訓練 (Phase 2)                                      │
├─────────────────────────────────────────────────────────────────┤
│  • 凍結: Router (已預訓練好)                                     │
│  • 訓練: Backbone (低學習率) + Experts                           │
│  • Loss: YOLO Detection Loss                                     │
│  • Epochs: 200                                                   │
│  • 輸出: moe_experts.pt                                          │
│                                                                  │
│  python train_moe_phase2.py \                                    │
│      --router-weights router_pretrained.pt \                     │
│      --epochs 200                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: (可選) 端到端微調 (Phase 3)                              │
├─────────────────────────────────────────────────────────────────┤
│  • 訓練: 全部參數 (低學習率)                                     │
│  • Loss: YOLO Detection Loss + aux_loss                          │
│  • Epochs: 50                                                    │
│                                                                  │
│  python train_moe_finetune.py \                                  │
│      --weights moe_experts.pt \                                  │
│      --epochs 50                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 預期效果

### Router 預訓練後

```
輸入圖片: 有人和車
Router 輸出:
┌─────────────────────────────────────────┐
│ Expert 0 (人):     0.92  ✓ 選中         │
│ Expert 1 (交通):   0.87  ✓ 選中         │
│ Expert 2 (動物):   0.12  ✗              │
│ Expert 3 (其他):   0.08  ✗              │
└─────────────────────────────────────────┘
→ 正確分配給 Expert 0 和 Expert 1
```

### 專家訓練後

```
Expert 0: 專精人物檢測 (person 的 AP 特別高)
Expert 1: 專精車輛檢測 (car, bus, truck 的 AP 特別高)
Expert 2: 專精動物檢測 (dog, cat, bird 的 AP 特別高)
Expert 3: 專精其他物件
```

---

## 監控指標

### Phase 1 (Router 預訓練)

| 指標 | 健康範圍 | 說明 |
|------|----------|------|
| Router 準確率 | > 80% | 正確選擇應選專家的比例 |
| Loss | 持續下降 | BCE Loss |

### Phase 2 (專家訓練)

| 指標 | 健康範圍 | 說明 |
|------|----------|------|
| 各專家 mAP | 持續提升 | 監控每個專家負責類別的 AP |
| 整體 mAP | > 41.4% (baseline) | 應該超過原始 YOLOv7-tiny |

---

## 超參數建議

### Phase 1: Router 預訓練

| 參數 | 建議值 | 說明 |
|------|--------|------|
| `epochs` | 50 | Router 結構簡單，不需要太久 |
| `batch_size` | 64 | 可以較大 |
| `learning_rate` | 0.001 | Adam |
| `threshold` | 0.5 | 判斷選擇哪個專家的閾值 |

### Phase 2: 專家訓練

| 參數 | 建議值 | 說明 |
|------|--------|------|
| `epochs` | 200 | 專家需要足夠時間特化 |
| `batch_size` | 32 | 記憶體考量 |
| `backbone_lr` | 0.001 | Backbone 低學習率 |
| `expert_lr` | 0.01 | Expert 正常學習率 |

---

## 風險與注意事項

### 1. 類別分配不均

```
問題: 某些專家負責的類別在資料集中很少
解決: 檢查各專家的訓練樣本數，必要時調整分類
```

### 2. 多類別圖片

```
問題: 圖片中有太多類別，需要選太多專家
解決:
  - 限制最多選 top_k 個專家
  - 選擇物件數量最多的類別對應的專家
```

### 3. Router 過擬合

```
問題: Router 在訓練集上準確率很高，但泛化能力差
解決:
  - 加入 Dropout
  - 使用驗證集監控
  - 減少訓練 epochs
```

---

## 與 4-Head 實驗的關係

本方案結合了兩種方法的優點：

| 項目 | 4-Head 實驗 | 原 MoE 方案 | 本方案 (監督式 MoE) |
|------|------------|-------------|---------------------|
| 類別分配 | 人工固定 | 自動學習 | **人工定義 + 學習選擇** |
| 專家結構 | 只有 Head 不同 | Neck+Head 都不同 | **Neck+Head 都不同** |
| 選擇機制 | 全部執行 | Router 選擇 | **Router 選擇 (預訓練)** |
| 輸出合併 | 按類別拼接 | WBF | **WBF** |
| 可控性 | 高 | 低 | **高** |
| 靈活性 | 低 | 高 | **中** |

---

## 檔案結構

```
yolov7_3_moe/
├── MOE_EXPERIMENT_DESIGN.md        # 原方案 (自動學習)
├── MOE_SUPERVISED_ROUTER.md        # 本方案 (監督式)
│
├── scripts/
│   ├── generate_router_labels.py   # 生成 Router 標籤
│   ├── train_router_pretrain.py    # Phase 1: Router 預訓練
│   └── train_moe_phase2.py         # Phase 2: 專家訓練
│
├── models/
│   ├── moe.py
│   ├── moe_yolo.py
│   └── wbf.py
│
└── data/
    └── router_labels_train.json    # Router 訓練標籤
```

---

## 更新日誌

| 日期 | 更新內容 |
|------|----------|
| 2025-01-19 | 建立監督式路由訓練方案文件 |
