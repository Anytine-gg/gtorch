import torch
import torch.nn.functional as F

def calc_IoU(box1, box2):
    """
    计算 IoU

    参数:
        box1: [xmin, ymin, xmax, ymax]
        box2: [xmin, ymin, xmax, ymax]

    返回:
        IoU 值(float)
    """
    # 计算交集框的坐标
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    # 计算交集的宽和高，注意宽或高不能为负
    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)
    inter_area = inter_width * inter_height

    # 计算各自的面积
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集面积
    union_area = area_box1 + area_box2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def calc_IoU_tensor(bboxes, anchors):
    """使用tensor计算iou, bboxes和anchors需相同的shape (n,4),\
        每一行为中心坐标,width,height表示法,每一行做IoU

    Args:
        bboxes (_type_): shape
        anchors (_type_): _description_

    Returns:
        _type_: 长度为n的IoU
    """
    # 将中心点、宽、高转换为左上角和右下角, 格式: [x1, y1, x2, y2]
    bboxes_corners = torch.cat(
        [bboxes[:, :2] - bboxes[:, 2:] / 2, bboxes[:, :2] + bboxes[:, 2:] / 2], dim=1
    )
    anchors_corners = torch.cat(
        [anchors[:, :2] - anchors[:, 2:] / 2, anchors[:, :2] + anchors[:, 2:] / 2],
        dim=1,
    )

    # 计算相交区域的左上角和右下角
    inter_top_left = torch.max(bboxes_corners[:, :2], anchors_corners[:, :2])
    inter_bottom_right = torch.min(bboxes_corners[:, 2:], anchors_corners[:, 2:])
    inter_wh = (inter_bottom_right - inter_top_left).clamp(min=0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    # 计算每个框的面积
    bboxes_area = (bboxes_corners[:, 2] - bboxes_corners[:, 0]) * (
        bboxes_corners[:, 3] - bboxes_corners[:, 1]
    )
    anchors_area = (anchors_corners[:, 2] - anchors_corners[:, 0]) * (
        anchors_corners[:, 3] - anchors_corners[:, 1]
    )

    # 计算并集的面积，并计算 IoU
    union_area = bboxes_area + anchors_area - inter_area
    iou = inter_area / union_area
    return iou



import torch
import torch.nn.functional as F

def yolo3_loss(predict_feat: torch.Tensor, label_feat: torch.Tensor):
    # 确保输入形状一致
    assert predict_feat.shape == label_feat.shape, "predict and label must have the same shape"
    
    # 获取维度信息
    b, _, w, h = predict_feat.shape
    num_anchors = 3
    num_classes = (predict_feat.shape[1] // num_anchors) - 5
    
    # 将特征图重塑为 (b, anchors, 5+classes, w, h)
    predict = predict_feat.view(b, num_anchors, 5+num_classes, w, h)
    label = label_feat.view(b, num_anchors, 5+num_classes, w, h)
    
    # 分解预测结果
    pred_tx_ty = predict[:, :, 0:2, :, :]    # 坐标tx, ty
    pred_tw_th = predict[:, :, 2:4, :, :]    # 宽高tw, th
    pred_conf = predict[:, :, 4, :, :]       # 置信度 (b, a, w, h)
    pred_cls = predict[:, :, 5:, :, :]       # 分类预测 (b, a, c, w, h)
    
    # 分解标签结果
    label_tx_ty = label[:, :, 0:2, :, :]
    label_tw_th = label[:, :, 2:4, :, :]
    label_conf = label[:, :, 4, :, :]        # 置信度标签
    label_cls = label[:, :, 5:, :, :]
    
    # 生成物体掩码 (conf=1的位置)
    obj_mask = label_conf == 1               # (b, a, w, h)
    noobj_mask = label_conf == 0             # (b, a, w, h)
    
    # =====================
    # 坐标损失计算 (仅对conf=1的位置)
    # =====================
    # 对tx, ty应用sigmoid
    pred_tx_ty_sigmoid = torch.sigmoid(pred_tx_ty)
    
    # 扩展掩码到坐标维度
    obj_mask_xy = obj_mask.unsqueeze(2).expand_as(pred_tx_ty_sigmoid)
    loss_xy = F.mse_loss(
        pred_tx_ty_sigmoid[obj_mask_xy],
        label_tx_ty[obj_mask_xy],
        reduction='sum'
    )
    
    # 宽高损失
    obj_mask_wh = obj_mask.unsqueeze(2).expand_as(pred_tw_th)
    loss_wh = F.mse_loss(
        pred_tw_th[obj_mask_wh],
        label_tw_th[obj_mask_wh],
        reduction='sum'
    )
    coord_loss = loss_xy + loss_wh
    
    # =====================
    # 置信度损失计算
    # =====================
    # 正样本 (conf=1) 的置信度损失
    obj_conf_loss = F.binary_cross_entropy_with_logits(
        pred_conf[obj_mask],
        label_conf[obj_mask],
        reduction='sum'
    )
    
    # 负样本 (conf=0) 的置信度损失
    noobj_conf_loss = F.binary_cross_entropy_with_logits(
        pred_conf[noobj_mask],
        label_conf[noobj_mask],
        reduction='sum'
    )
    
    # 总置信度损失
    conf_loss = obj_conf_loss + 0.5 * noobj_conf_loss  # 增加负样本权重控制
    
    # =====================
    # 分类损失计算 (仅对conf=1的位置)
    # =====================
    # 扩展掩码到分类维度
    obj_mask_cls = obj_mask.unsqueeze(2).expand(-1, -1, num_classes, -1, -1)
    cls_loss = F.binary_cross_entropy_with_logits(
        pred_cls[obj_mask_cls],
        label_cls[obj_mask_cls],
        reduction='sum'
    )
    
    # =====================
    # 总损失
    # =====================
    # 归一化损失 (除以正样本数量)
    num_positive = obj_mask.sum().clamp(min=1)  # 至少为1，避免除零
    total_loss = (coord_loss + conf_loss + cls_loss) / num_positive
    return total_loss



if __name__ == '__main__':
   pass