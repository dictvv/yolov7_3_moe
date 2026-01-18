# MoE-YOLOv7-tiny 進度報告

## 目前狀態：實驗設計完成

---

## 實驗概述

在 YOLOv7-tiny 上實現 Mixture of Experts (MoE) 架構：
- 共享 Backbone (Layer 0-28)
- Router 選擇 Top-2 專家
- 4 個專家 (Neck + Head)
- 加權平均合併輸出

詳細設計請見 [MOE_EXPERIMENT_DESIGN.md](MOE_EXPERIMENT_DESIGN.md)

---

## 進度追蹤

### Phase 1: 環境準備
| 項目 | 狀態 | 說明 |
|------|------|------|
| 複製基礎程式碼 | ⬜ 待開始 | 從 yolov7_1 複製 |
| 確認 baseline 運行 | ⬜ 待開始 | |
| 準備 COCO 資料集 | ⬜ 待開始 | 320×320 版本 |

### Phase 2: 實作 MoE 模組
| 項目 | 狀態 | 說明 |
|------|------|------|
| Router 類別 | ⬜ 待開始 | `models/moe.py` |
| MoE-YOLOv7 類別 | ⬜ 待開始 | `models/moe_yolo.py` |
| Expert wrapper | ⬜ 待開始 | |
| 輸出合併邏輯 | ⬜ 待開始 | |

### Phase 3: 訓練腳本
| 項目 | 狀態 | 說明 |
|------|------|------|
| train_moe.py | ⬜ 待開始 | |
| test_moe.py | ⬜ 待開始 | |

### Phase 4: 訓練與評估
| 項目 | 狀態 | 說明 |
|------|------|------|
| 訓練 500 epochs | ⬜ 待開始 | |
| 評估 mAP | ⬜ 待開始 | |
| 分析 Router 行為 | ⬜ 待開始 | |

---

## Baseline 對比目標

| 指標 | Baseline (OTA) | MoE 目標 |
|------|----------------|----------|
| mAP@0.5 | 41.4% | **45%** |

---

## 更新日誌

| 日期 | 更新內容 |
|------|----------|
| 2025-01-18 | 建立專案，完成實驗設計文件 |
