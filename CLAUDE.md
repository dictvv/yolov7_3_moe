# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 重要規定

1. **每次進入 Claude Code 時，必須先閱讀 `progress.md` 了解目前進度**
2. **所有回覆必須使用繁體中文**
3. **不修改原始 YOLOv7 程式碼**：`models/yolo.py` 和 `models/common.py` 保持不變
4. **新增模組獨立檔案**：MoE 相關程式碼放在 `models/moe.py` 和 `models/moe_yolo.py`
5. **實驗設計參考**：詳細設計請見 `MOE_EXPERIMENT_DESIGN.md`

## Project Overview

MoE-YOLOv7-tiny: 在 YOLOv7-tiny 上實現 Mixture of Experts 架構的實驗專案。

### 核心概念
- **共享 Backbone** (Layer 0-28): 特徵提取
- **Router**: 根據 Backbone 輸出選擇 Top-2 專家
- **4 個專家**: 每個專家 = Neck (29-73) + Head (74-77)
- **輸出合併**: 加權平均被選中專家的預測

## 檔案結構

```
yolov7_3_moe/
├── MOE_EXPERIMENT_DESIGN.md   # 實驗設計文件
├── CLAUDE.md                   # 本文件
├── progress.md                 # 進度追蹤
├── models/
│   ├── moe.py                 # Router 類別
│   └── moe_yolo.py            # MoE-YOLOv7 類別
├── train_moe.py               # 訓練腳本
└── test_moe.py                # 測試腳本
```

## 開發原則

1. **最簡單實現**: 先讓它能跑，再優化
2. **保持相容性**: 輸出格式與原始 YOLOv7 相同
3. **不加額外 loss**: 暫時不加 load balancing loss

## 常用指令

```bash
# 訓練 MoE 模型
python train_moe.py --data data/coco.yaml --cfg cfg/training/yolov7-tiny.yaml \
    --weights baseline.pt --batch-size 64 --epochs 100

# 測試 MoE 模型
python test_moe.py --data data/coco.yaml --weights runs/train/moe_exp1/weights/best.pt
```
