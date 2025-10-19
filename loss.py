import torch
import torch.nn.functional as F
from utils.util import dice_score  # 保留：你的 general_dice_loss 还在用 mdice

# ====== 全局：形态正则强度（已默认启用）======
_LAMBDA_SHAPE = 0.05     # 你要调强弱，只改这里
_EPS_SHAPE     = 1e-3    # 数值稳定项

# ------------------------- 小工具 -------------------------
def _binary_dice(pred_mask: torch.Tensor, true_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred_mask, true_mask: [B,1,D,H,W] in {0,1}
    处理“真值全 0”的情况：
      - 真值全 0 & 预测全 0 -> 1.0
      - 真值全 0 & 预测有正例 -> 0.0
    """
    pred_mask = pred_mask.float()
    true_mask = true_mask.float()
    inter = (pred_mask * true_mask).sum()
    denom = pred_mask.sum() + true_mask.sum()
    if true_mask.sum() == 0:
        return (pred_mask.sum() == 0).float()
    return (2 * inter + eps) / (denom + eps)

def _auto_index_for_label(p: torch.Tensor, label: torch.Tensor, target_val: int, default_idx: int) -> int:
    """
    在 label==target_val 的体素上，选择 softmax 概率平均值最大的通道索引。
    如果该类别在当前 batch 中不存在（体素数=0），返回 default_idx。
    p:     [B,C,D,H,W] softmax 概率
    label: [B,1,D,H,W] 整标签
    """
    with torch.no_grad():
        mask = (label == target_val).float()  # [B,1,D,H,W]
        vox = mask.sum()
        if vox == 0:
            return int(default_idx)
        # 逐通道计算在该区域的平均概率
        C = p.shape[1]
        best_idx, best_score = 0, None
        for c in range(C):
            score_c = (p[:, c:c+1] * mask).sum() / (vox + 1e-6)
            if (best_score is None) or (score_c > best_score):
                best_score = score_c
                best_idx = c
        return int(best_idx)

# ------------------------- metrics（自适应通道版本） -------------------------
def get_dice(seg, label):
    """
    seg: [B,4,D,H,W] logits；label: [B,1,D,H,W] in {0,1,2,4}
    集合定义（BraTS 口径）：
      WT = {1,2,4}，TC = {1,4}，ET = {4}
    预测集合构造：
      - 背景通道默认 0（常见实现如此）
      - ET 通道用 _auto_index_for_label(p, label, 4, default=3) 自适应识别
      - ED(水肿) 通道用 _auto_index_for_label(p, label, 2, default=2) 自适应识别
      - WT_pred = 预测中非背景
      - TC_pred = 预测中非背景且非 ED
      - ET_pred = 预测为 ET
    """
    assert seg.ndim == 5 and label.ndim == 5, "seg [B,C,D,H,W], label [B,1,D,H,W]"
    B, C, D, H, W = seg.shape
    assert C == 4, f"Expected 4 classes, got {C}"

    p = F.softmax(seg, dim=1)                         # [B,4,D,H,W]
    pre = torch.argmax(p, dim=1, keepdim=True)        # [B,1,D,H,W], 0..3

    # 自适应确定 ET/ED 通道索引（默认 et=3, ed=2）
    et_idx = _auto_index_for_label(p, label, target_val=4, default_idx=3)
    ed_idx = _auto_index_for_label(p, label, target_val=2, default_idx=2)
    bg_idx = 0

    pre_cls = pre  # [B,1,D,H,W]

    # 预测集合
    wt_pred = (pre_cls != bg_idx).float()
    if (label == 2).sum() > 0:
        tc_pred = ((pre_cls != bg_idx) & (pre_cls != ed_idx)).float()
    else:
        tc_pred = wt_pred.clone()
    et_pred = (pre_cls == et_idx).float()

    # 真值集合
    wt_true = ((label == 1) | (label == 2) | (label == 4)).float()
    tc_true = ((label == 1) | (label == 4)).float()
    et_true = (label == 4).float()

    dice_wt = _binary_dice(wt_pred, wt_true)
    dice_tc = _binary_dice(tc_pred, tc_true)
    dice_et = _binary_dice(et_pred, et_true)

    return dice_wt, dice_tc, dice_et

# ------------------------- helpers -------------------------
def get_label(label):
    """把 {0,1,2,4} 的整标签转成 4 个 one-vs-rest 二值通道（顺序: 0,1,2,4）"""
    extent_label = None
    for k in [0, 1, 2, 4]:
        la = label.clone()
        la[la == k] = -1
        la[la != -1] = 0
        la[la != 0] = 1
        extent_label = la if extent_label is None else torch.cat([extent_label, la], dim=1)
    return extent_label

def general_dice_loss(seg, label, mdice):
    s = F.softmax(seg, dim=1)
    la = get_label(label).detach()
    return mdice(s, la, [0.1, 0.2, 0.3, 0.4])

def U_Hemis_loss(seg, label, mdice):
    # Baseline：不加形态正则（只在 RMBTS/LMCR 里引入）
    return general_dice_loss(seg, label, mdice)

# ------------------------- RMBTS -------------------------
def RMBTS_loss(re_dic, label, inputs, mdice, m_d, miss_list, device):
    """
    返回: total_loss, dice_loss, ce_loss, rec_loss, KL_loss
    （签名保持不变；总损失里已经加入形态正则）
    """
    la  = get_label(label).detach()
    seg = re_dic['seg']

    # 1) Dice
    dice_loss = general_dice_loss(seg, label, mdice)

    # 2) Cross-Entropy（one-vs-rest，反频率权重；纯张量计算，避免 .item() 同步）
    s = F.softmax(seg, dim=1)                          # [B,4,D,H,W]
    total_pos = la.sum(dim=(0, 2, 3, 4)).clamp_min(1)  # [4]
    total_all = la.sum().clamp_min(1)
    weights = (1.0 - total_pos / total_all).view(1, -1, 1, 1, 1)  # [1,4,1,1,1]
    ce = -weights * la * torch.log(s.clamp(min=5e-3))
    ce_loss = ce.mean()

    seg_loss = dice_loss + ce_loss

    # 3) Reconstruction（仅对参与的模态）
    rec_name = ['reconstruct_t1c__','reconstruct_t1___','reconstruct_t2___','reconstruct_flair']
    rec_loss = torch.zeros((), device=device)
    for i in range(len(rec_name)):
        if m_d in miss_list[i]:
            rec_loss = rec_loss + torch.mean(torch.abs(re_dic[rec_name[i]] - inputs[i]))

    # 4) KL（仅对参与的模态）
    mu_name    = ['mu_t1c__','mu_t1___','mu_t2___','mu_flair']
    sigma_name = ['sigma_t1c__','sigma_t1___','sigma_t2___','sigma_flair']
    KL_loss = torch.zeros((), device=device)
    for i in range(len(mu_name)):
        if m_d in miss_list[i]:
            logvar = torch.log(torch.clamp(re_dic[sigma_name[i]] ** 2, min=1e-8))
            KL_loss = KL_loss + kl_loss(re_dic[mu_name[i]], logvar)

    # 5) Shape curvature regularization（默认启用；使用自适应 ET 通道）
    shape_loss = shape_curvature_loss(seg, label, eps=_EPS_SHAPE)

    total = seg_loss + 0.1 * rec_loss + 0.1 * KL_loss + _LAMBDA_SHAPE * shape_loss
    return total, dice_loss, ce_loss, rec_loss, KL_loss

def kl_loss(mu, logvar):
    # VAE KL(q||p): 0.5 * sum(mu^2 + exp(logvar) - 1 - logvar)
    loss = 0.5 * torch.sum(mu.pow(2) + torch.exp(logvar) - 1.0 - logvar, dim=1)
    return loss.mean()

# ------------------------- LMCR -------------------------
def LMCR_loss(re_dic, label, inputs, mdice, m_d, miss_list, device):
    """
    返回: total_loss, dice_loss, rec_loss
    （签名保持不变；总损失里已经加入形态正则）
    """
    seg = re_dic['seg']

    # 1) Dice
    dice_loss = general_dice_loss(seg, label, mdice)
    seg_loss  = dice_loss

    # 2) Reconstruction（仅对参与的模态）
    rec_name = ['reconstruct_t1c__','reconstruct_t1___','reconstruct_t2___','reconstruct_flair']
    rec_loss = torch.zeros((), device=device)
    for i in range(len(rec_name)):
        if m_d in miss_list[i]:
            rec_loss = rec_loss + torch.mean(torch.abs(re_dic[rec_name[i]] - inputs[i]))

    # 3) Shape curvature regularization（默认启用；使用自适应 ET 通道）
    shape_loss = shape_curvature_loss(seg, label, eps=_EPS_SHAPE)

    total = seg_loss + rec_loss + _LAMBDA_SHAPE * shape_loss
    return total, dice_loss, rec_loss

# ------------------------- shape prior（自适应 ET 通道） -------------------------
def _grad3d(u):
    # u: [B,1,D,H,W]；前向差分 + 边界复制 pad
    dx = F.pad(u[..., 1:] - u[..., :-1], (1, 0, 0, 0, 0, 0))               # ∂/∂W
    dy = F.pad(u[:, :, :, 1:, :] - u[:, :, :, :-1, :], (0, 0, 1, 0, 0, 0)) # ∂/∂H
    dz = F.pad(u[:, :, 1:, :, :] - u[:, :, :-1, :, :], (0, 0, 0, 0, 1, 0)) # ∂/∂D
    return dx, dy, dz

def _div3d(px, py, pz):
    # 与 _grad3d 对应的“离散散度”（反向差分）
    dx = F.pad(px[..., :-1] - px[..., 1:], (0, 1, 0, 0, 0, 0))
    dy = F.pad(py[:, :, :, :-1, :] - py[:, :, :, 1:, :], (0, 0, 0, 1, 0, 0))
    dz = F.pad(pz[:, :, :-1, :, :] - pz[:, :, 1:, :, :], (0, 0, 0, 0, 0, 1))
    return dx + dy + dz

def curvature_map(u, eps=1e-3):
    gx, gy, gz = _grad3d(u)
    g = torch.sqrt(gx * gx + gy * gy + gz * gz + eps)
    return _div3d(gx / g, gy / g, gz / g)

def shape_curvature_loss(seg_logits, label, eps=1e-3):
    """
    自适应选择“ET 概率通道”做曲率约束：
      - 若 batch 中存在 ET 体素，则选在 ET 区域上概率平均最高的通道；
      - 若不存在，则默认使用索引 3（不影响该 batch 的优化）。
    """
    p = F.softmax(seg_logits, dim=1)                  # [B,4,D,H,W]
    et_idx = _auto_index_for_label(p, label, target_val=4, default_idx=3)
    p_et = p[:, et_idx:et_idx + 1]                    # [B,1,D,H,W]
    y_et = (label == 4).float()
    return F.l1_loss(curvature_map(p_et, eps), curvature_map(y_et, eps))
