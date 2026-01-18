# MoE-YOLOv7-tiny 進度報告

## 目前狀態：MoE 程式碼修復完成，準備訓練

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

### Phase 2: 實作 MoE 模組
| 項目 | 狀態 | 說明 |
|------|------|------|
| Router 類別 | ✅ 完成 | `models/moe.py` |
| MoE-YOLOv7 類別 | ✅ 完成 | `models/moe_yolo.py` (已修復負數索引問題) |
| WBF 合併工具 | ✅ 完成 | `models/wbf.py` (已修復推論輸出格式) |
| MoE Loss 計算 | ✅ 完成 | `utils/loss_moe.py` (已修復超參數使用) |

### Phase 3: 訓練/推論腳本
| 項目 | 狀態 | 說明 |
|------|------|------|
| train_moe.py | ✅ 完成 | MoE 訓練腳本 (已添加 mAP 驗證) |
| detect_moe.py | ✅ 完成 | MoE 推論腳本 |

### Phase 2.5: 程式碼修復 (2025-01-19)
| 問題 | 修復 | 說明 |
|------|------|------|
| moe_yolo.py 負數索引 | ✅ 已修復 | 正確處理 skip connection 的相對索引 |
| wbf.py 輸出格式 | ✅ 已修復 | 處理 Detect 層推論時的 tuple 輸出 |
| loss_moe.py 超參數 | ✅ 已修復 | 正確使用 cls_pw, obj_pw, fl_gamma, label_smoothing |
| train_moe.py 驗證 | ✅ 已修復 | 添加完整 mAP 驗證，使用 mAP@0.5 作為 fitness |

### Phase 4: 訓練與評估
| 項目 | 狀態 | 說明 |
|------|------|------|
| 訓練 500 epochs | ⬜ 待開始 | |
| 評估 mAP | ⬜ 待開始 | |
| 分析 Router 行為 | ⬜ 待開始 | |

---

## 新增檔案結構

```
yolov7_3_moe/
├── models/
│   ├── moe.py          # Router 類別 (Load Balancing Loss)
│   ├── moe_yolo.py     # MoE-YOLOv7 主模型
│   └── wbf.py          # Weighted Boxes Fusion 合併工具
├── utils/
│   └── loss_moe.py     # MoE Loss 計算
├── train_moe.py        # 訓練腳本
└── detect_moe.py       # 推論腳本
```

---

## 核心功能說明

### 1. Router (`models/moe.py`)
- 使用 Global Average Pooling + FC 計算專家分數
- Top-K 選擇 (預設 Top-2)
- Load Balancing Loss: `aux_loss = N × Σ(f × P)`

### 2. MoEYOLOv7 (`models/moe_yolo.py`)
- 共享 Backbone (Layer 0-28)
- 4 個獨立專家 (Neck + Head)
- 支援預訓練權重載入

### 3. WBF (`models/wbf.py`)
- 合併多專家的偵測框
- 使用加權平均融合重疊框
- 相比 NMS，保留更多資訊

### 4. Loss (`utils/loss_moe.py`)
- 每個專家獨立計算 loss
- 根據 Router 權重加權合併
- 加入 Load Balancing Loss

---

## Baseline 對比目標

| 指標 | Baseline (OTA) | MoE 目標 |
|------|----------------|----------|
| mAP@0.5 | 41.4% | **45%** |

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

# 推論
python detect_moe.py \
    --weights runs/train/moe_exp/weights/best.pt \
    --source data/images \
    --conf-thres 0.25
```

---

## 更新日誌

| 日期 | 更新內容 |
|------|----------|
| 2025-01-19 | 修復程式碼問題: wbf.py 輸出格式、loss_moe.py 超參數、train_moe.py mAP 驗證 |
| 2025-01-19 | 完成所有 MoE 程式碼實作 |
| 2025-01-18 | 建立專案，完成實驗設計文件 |
