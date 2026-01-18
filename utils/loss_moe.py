"""
MoE-YOLOv7 Loss Function

繼承原始 YOLOv7 的 ComputeLoss，額外處理:
1. 多專家 loss 計算
2. 加權合併
3. Load Balancing Loss
"""

import torch
import torch.nn as nn

from utils.loss import ComputeLoss, ComputeLossOTA


class ComputeLossMoE:
    """
    MoE-YOLOv7 的 Loss 計算器

    訓練流程:
    1. 每個被選中的專家分別計算 loss
    2. 根據 router 權重加權合併
    3. 加上 Load Balancing Loss

    Total Loss = Σ(w_i * expert_i_loss) + λ * aux_loss
    """

    def __init__(self, model, autobalance=False, use_ota=True, aux_loss_weight=0.01):
        """
        Args:
            model: MoEYOLOv7 模型
            autobalance: 是否自動平衡 objectness loss
            use_ota: 是否使用 OTA (Optimal Transport Assignment)
            aux_loss_weight: Load Balancing Loss 的權重
        """
        self.aux_loss_weight = aux_loss_weight
        self.use_ota = use_ota

        # 創建一個用於計算單專家 loss 的計算器
        # 需要用假的 model 來初始化，因為 ComputeLoss 需要從 model 取得一些屬性
        self._init_loss_computer(model, autobalance)

    def _init_loss_computer(self, model, autobalance):
        """
        初始化 loss 計算器

        由於 ComputeLoss 需要 model.model[-1] (Detect 層)，
        我們需要從 MoEYOLOv7 的專家中取得這些資訊
        """
        # 取得超參數
        self.hyp = model.hyp if hasattr(model, 'hyp') else None

        # 取得模型資訊
        device = next(model.parameters()).device

        # 從第一個專家的最後一層 (Detect) 取得必要資訊
        # 專家的最後一層就是 Detect 層
        expert_0 = model.experts[0]
        detect = expert_0[-1]  # 最後一層是 Detect

        # 創建 loss 計算所需的屬性
        self.device = device
        self.nc = model.nc  # 類別數
        self.nl = detect.nl  # 檢測層數量
        self.na = detect.na  # 每層 anchor 數量
        self.anchors = detect.anchors
        self.stride = model.stride

        # BCE losses
        if self.hyp:
            BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['cls_pw']], device=device))
            BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['obj_pw']], device=device))

            # Label smoothing
            from utils.loss import smooth_BCE
            self.cp, self.cn = smooth_BCE(eps=self.hyp.get('label_smoothing', 0.0))

            # Focal loss
            from utils.loss import FocalLoss
            g = self.hyp['fl_gamma']
            if g > 0:
                BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

            self.BCEcls = BCEcls
            self.BCEobj = BCEobj
            self.gr = model.gr if hasattr(model, 'gr') else 1.0

            # Balance
            self.balance = {3: [4.0, 1.0, 0.4]}.get(self.nl, [4.0, 1.0, 0.25, 0.06, .02])
            self.autobalance = autobalance
            self.ssi = list(self.stride).index(16) if autobalance else 0
        else:
            self.hyp = {
                'box': 0.05,
                'obj': 1.0,
                'cls': 0.5,
                'cls_pw': 1.0,
                'obj_pw': 1.0,
                'fl_gamma': 0.0,
                'anchor_t': 4.0,
            }
            BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
            BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
            self.BCEcls = BCEcls
            self.BCEobj = BCEobj
            self.cp, self.cn = 1.0, 0.0
            self.gr = 1.0
            self.balance = [4.0, 1.0, 0.4]
            self.autobalance = False
            self.ssi = 0

    def __call__(self, moe_output, targets):
        """
        計算 MoE Loss

        Args:
            moe_output: dict from MoEYOLOv7.forward()
                - expert_outputs: {expert_idx: prediction}
                - top_indices: [B, top_k]
                - top_weights: [B, top_k]
                - aux_loss: scalar
            targets: [N, 6] (image_idx, class, x, y, w, h)

        Returns:
            loss: scalar, 總 loss
            loss_items: tensor [4], (box_loss, obj_loss, cls_loss, total_loss)
        """
        expert_outputs = moe_output['expert_outputs']
        top_indices = moe_output['top_indices']  # [B, top_k]
        top_weights = moe_output['top_weights']  # [B, top_k]
        aux_loss = moe_output.get('aux_loss', 0)

        batch_size = top_indices.shape[0]
        device = targets.device

        # 初始化 loss
        total_loss = torch.zeros(1, device=device)
        total_lbox = torch.zeros(1, device=device)
        total_lobj = torch.zeros(1, device=device)
        total_lcls = torch.zeros(1, device=device)

        # 對每個專家計算 loss
        expert_losses = {}

        for expert_idx, pred in expert_outputs.items():
            # pred 是這個專家對整個 batch 的預測
            # 計算這個專家的 loss
            loss, loss_items = self._compute_single_expert_loss(pred, targets)
            expert_losses[expert_idx] = {
                'loss': loss,
                'loss_items': loss_items
            }

        # 加權合併 loss
        # 對於每個 batch 樣本，只有被選中的專家會貢獻 loss
        for b in range(batch_size):
            sample_loss = torch.zeros(1, device=device)
            sample_lbox = torch.zeros(1, device=device)
            sample_lobj = torch.zeros(1, device=device)
            sample_lcls = torch.zeros(1, device=device)

            # 這個樣本選中的專家
            indices = top_indices[b]  # [top_k]
            weights = top_weights[b]  # [top_k]

            for i, (expert_idx, weight) in enumerate(zip(indices, weights)):
                expert_idx = int(expert_idx.item())
                weight = weight.item()

                if expert_idx in expert_losses:
                    # 從專家的總 loss 中估算這個樣本的貢獻
                    # 簡化：假設每個樣本對專家 loss 的貢獻相等
                    expert_loss = expert_losses[expert_idx]['loss'] / batch_size
                    expert_items = expert_losses[expert_idx]['loss_items'] / batch_size

                    sample_loss += weight * expert_loss
                    sample_lbox += weight * expert_items[0]
                    sample_lobj += weight * expert_items[1]
                    sample_lcls += weight * expert_items[2]

            total_loss += sample_loss
            total_lbox += sample_lbox
            total_lobj += sample_lobj
            total_lcls += sample_lcls

        # 加上 Load Balancing Loss
        if isinstance(aux_loss, torch.Tensor):
            total_loss = total_loss + self.aux_loss_weight * aux_loss

        # 組合 loss items
        loss_items = torch.cat((total_lbox, total_lobj, total_lcls, total_loss)).detach()

        return total_loss, loss_items

    def _compute_single_expert_loss(self, pred, targets):
        """
        計算單一專家的 loss (類似原始 ComputeLoss)

        Args:
            pred: 專家預測輸出 (list of tensors for each detection layer)
            targets: [N, 6]

        Returns:
            loss: scalar
            loss_items: tensor [4]
        """
        from utils.general import bbox_iou

        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)

        # pred 可能是 list (多個檢測層) 或 tensor
        if not isinstance(pred, (list, tuple)):
            # 如果是單一 tensor，可能需要處理
            # 這種情況通常不會發生，但做個防護
            return torch.zeros(1, device=device, requires_grad=True), torch.zeros(4, device=device)

        # 建立 targets
        tcls, tbox, indices, anchors = self._build_targets(pred, targets)

        # 計算 loss
        for i, pi in enumerate(pred):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)

            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)

                # Classification
                if self.nc > 1:
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = pred[0].shape[0]

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def _build_targets(self, p, targets):
        """
        建立訓練 targets (從原始 ComputeLoss 複製)

        Args:
            p: predictions list
            targets: [N, 6] (image_idx, class, x, y, w, h)

        Returns:
            tcls, tbox, indices, anchors
        """
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device).long()
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],
                            ], device=targets.device).float() * g

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]

            t = targets * gain
            if nt:
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']
                t = t[j]

                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            a = t[:, 6].long()
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)

        return tcls, tbox, indices, anch


class ComputeLossMoESimple:
    """
    簡化版 MoE Loss 計算

    直接對所有被選中專家的 loss 加權平均，不區分每個 batch 樣本
    """

    def __init__(self, model, autobalance=False, aux_loss_weight=0.01):
        """
        Args:
            model: MoEYOLOv7 模型
            autobalance: 是否自動平衡
            aux_loss_weight: Load Balancing Loss 權重
        """
        self.aux_loss_weight = aux_loss_weight
        self.model = model

        # 為每個專家創建獨立的 loss 計算器
        self.expert_loss_computers = {}

    def set_hyp(self, hyp):
        """設定超參數"""
        self.hyp = hyp
        self.model.hyp = hyp

    def __call__(self, moe_output, targets):
        """
        計算 loss

        Args:
            moe_output: MoEYOLOv7 輸出
            targets: 訓練目標

        Returns:
            loss: 總 loss
            loss_items: [lbox, lobj, lcls, total]
        """
        expert_outputs = moe_output['expert_outputs']
        top_indices = moe_output['top_indices']
        top_weights = moe_output['top_weights']
        aux_loss = moe_output.get('aux_loss', 0)

        device = targets.device
        batch_size = top_indices.shape[0]

        total_loss = torch.zeros(1, device=device)
        total_lbox = torch.zeros(1, device=device)
        total_lobj = torch.zeros(1, device=device)
        total_lcls = torch.zeros(1, device=device)

        # 計算每個專家的平均權重
        expert_avg_weights = {}
        for expert_idx in expert_outputs.keys():
            mask = (top_indices == expert_idx)
            if mask.sum() > 0:
                avg_weight = top_weights[mask].mean()
                expert_avg_weights[expert_idx] = avg_weight

        # 計算每個專家的 loss 並加權
        for expert_idx, pred in expert_outputs.items():
            if expert_idx not in expert_avg_weights:
                continue

            weight = expert_avg_weights[expert_idx]

            # 使用原始 ComputeLoss 計算
            # 需要創建一個假的 model wrapper
            loss, loss_items = self._compute_expert_loss(pred, targets, expert_idx)

            total_loss += weight * loss
            total_lbox += weight * loss_items[0]
            total_lobj += weight * loss_items[1]
            total_lcls += weight * loss_items[2]

        # 加上 aux loss
        if isinstance(aux_loss, torch.Tensor):
            total_loss = total_loss + self.aux_loss_weight * aux_loss

        loss_items = torch.cat((total_lbox, total_lobj, total_lcls, total_loss)).detach()

        return total_loss, loss_items

    def _compute_expert_loss(self, pred, targets, expert_idx):
        """
        計算單一專家的 loss

        這裡我們直接使用原始的 ComputeLoss 邏輯
        """
        from utils.general import bbox_iou

        device = targets.device
        hyp = getattr(self, 'hyp', None) or self.model.hyp

        if hyp is None:
            # 使用預設值
            hyp = {
                'box': 0.05,
                'obj': 1.0,
                'cls': 0.5,
                'anchor_t': 4.0,
            }

        # 取得專家的 Detect 層資訊
        expert = self.model.experts[expert_idx]
        detect = expert[-1]

        nl = detect.nl
        na = detect.na
        nc = self.model.nc
        anchors = detect.anchors

        # 計算 loss
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)

        if not isinstance(pred, (list, tuple)):
            return torch.zeros(1, device=device, requires_grad=True), torch.zeros(4, device=device)

        # Build targets
        tcls, tbox, indices, anch = self._build_targets_simple(pred, targets, anchors, na, nl, hyp)

        # BCE losses
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device))

        balance = [4.0, 1.0, 0.4]
        gr = 1.0

        for i, pi in enumerate(pred):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)

            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]

                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anch[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()

                tobj[b, a, gj, gi] = (1.0 - gr) + gr * iou.detach().clamp(0).type(tobj.dtype)

                if nc > 1:
                    t = torch.full_like(ps[:, 5:], 0.0, device=device)
                    t[range(n), tcls[i]] = 1.0
                    lcls += BCEcls(ps[:, 5:], t)

            obji = BCEobj(pi[..., 4], tobj)
            lobj += obji * balance[i] if i < len(balance) else obji

        lbox *= hyp['box']
        lobj *= hyp['obj']
        lcls *= hyp['cls']
        bs = pred[0].shape[0]

        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def _build_targets_simple(self, p, targets, anchors_all, na, nl, hyp):
        """簡化的 build_targets"""
        nt = targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device).long()
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        targets_ext = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
                          device=targets.device).float() * g

        for i in range(nl):
            anchors = anchors_all[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]

            t = targets_ext * gain
            if nt:
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1. / r).max(2)[0] < hyp.get('anchor_t', 4.0)
                t = t[j]

                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets_ext[0]
                offsets = 0

            b, c = t[:, :2].long().T
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            a = t[:, 6].long()
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)

        return tcls, tbox, indices, anch
