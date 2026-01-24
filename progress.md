# MoE-YOLOv7-tiny 進度報告

## 目前狀態：MoE 程式碼重新編寫完成，準備訓練

---

## 實驗概述

在 YOLOv7-tiny 上實現 Mixture of Experts (MoE) 架構：
- 共享 Backbone (Layer 0-28)
- Router 選擇 Top-2 專家
- 4 個專家 (Neck + Head)
- WBF (Weighted Boxes Fusion) 合併輸出
- Load Balancing Loss 防止專家偏好

詳細設計請見 [MOE_EXPERIMENT_DESIGN.md](MOE_EXPERIMENT_DESIGN.md)

---

## 進度追蹤

### Phase 1: 環境準備
| 項目 | 狀態 | 說明 |
|------|------|------|
| 複製基礎程式碼 | ✅ 完成 | 從官方 YOLOv7 合併 |
| 確認 baseline 運行 | ⬜ 待開始 | |
| 準備 COCO 資料集 | ⬜ 待開始 | 320×320 版本 |

### Phase 2: 實作 MoE 模組 (重新編寫)
| 項目 | 狀態 | 說明 |
|------|------|------|
| Router 類別 | ✅ 完成 | `models/moe.py` - 100% 原版邏輯 |
| MoE-YOLOv7 類別 | ✅ 完成 | `models/moe_yolo.py` - 使用 list 儲存 y，與原版完全一致 |
| WBF 合併工具 | ✅ 完成 | `models/wbf.py` - 推論時合併專家輸出 |
| MoE Loss 計算 | ✅ 完成 | `utils/loss_moe.py` - 使用原版 ComputeLoss |

### Phase 3: 訓練腳本
| 項目 | 狀態 | 說明 |
|------|------|------|
| train_moe.py | ✅ 完成 | MoE 訓練腳本 |
| detect_moe.py | ⬜ 待開始 | MoE 推論腳本 |

### Phase 4: 訓練與評估
| 項目 | 狀態 | 說明 |
|------|------|------|
| 訓練 100 epochs | ⬜ 待開始 | |
| 評估 mAP | ⬜ 待開始 | |
| 分析 Router 行為 | ⬜ 待開始 | |

---

## 新增檔案結構

```
yolov7_3_moe/
├── models/
│   ├── moe.py          # Router 類別 (Load Balancing Loss)
│   ├── moe_yolo.py     # MoE-YOLOv7 主模型 (100% 原版邏輯)
│   └── wbf.py          # Weighted Boxes Fusion 合併工具
├── utils/
│   └── loss_moe.py     # MoE Loss 計算 (使用原版 ComputeLoss)
└── train_moe.py        # 訓練腳本
```

---

## 核心功能說明

### 1. Router (`models/moe.py`)
- Global Average Pooling + FC 計算專家分數
- Top-K 選擇 (預設 Top-2)
- Load Balancing Loss: `aux_loss = N × Σ(f × P)`

### 2. MoEYOLOv7 (`models/moe_yolo.py`)
**重要：100% 使用原版 YOLOv7 的 forward 邏輯**
- 使用 `list` 儲存 y（與原版相同，支援負數索引）
- 使用 `m.i in self.save` 判斷是否保存（與原版相同）
- 共享 Backbone (Layer 0-28)
- 4 個獨立專家 (Neck + Head, Layer 29-77)

### 3. WBF (`models/wbf.py`)
- 合併多專家的偵測框
- 根據專家權重調整信心分數
- 使用 NMS 去除重複框

### 4. Loss (`utils/loss_moe.py`)
- **直接使用原版 ComputeLoss**
- 為每個專家建立獨立的 ComputeLoss 實例
- 根據 Router 權重加權合併 loss
- 加入 Load Balancing Loss

---

## 訓練指令

```bash
# 訓練 MoE-YOLOv7
python train_moe.py \
    --data data/coco.yaml \
    --cfg cfg/training/yolov7-tiny.yaml \
    --weights yolov7-tiny.pt \
    --batch-size 16 \
    --epochs 100 \
    --num-experts 4 \
    --top-k 2 \
    --aux-loss-weight 0.01 \
    --img-size 320
```

---

## 更新日誌

| 日期 | 更新內容 |
|------|----------|
| 2025-01-24 | 完全重新編寫所有 MoE 程式碼，確保 100% 與原版 YOLOv7 邏輯一致 |
| 2025-01-19 | 發現舊版程式碼使用 dict 而非 list 儲存 y，導致負數索引問題 |
| 2025-01-19 | 完成初版 MoE 程式碼實作 |
| 2025-01-18 | 建立專案，完成實驗設計文件 |

---

## 與原版 YOLOv7 的差異

### 保持一致
- ✅ `y` 使用 list 儲存（支援負數索引）
- ✅ 使用 `m.i in self.save` 判斷保存
- ✅ Loss 使用原版 ComputeLoss
- ✅ Skip connection 處理邏輯完全相同

### 新增部分
- Router 類別（選擇專家）
- 4 個專家（複製自 Layer 29-77）
- Load Balancing Loss
- WBF 合併（推論時）
