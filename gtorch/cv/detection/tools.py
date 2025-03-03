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



def yolo3_loss(predict_feat: torch.Tensor, label_feat: torch.Tensor):
    """
    predict_feat 和 label_feat 形状：(batch, (5+num_class)*3, grid_h, grid_w)
    每个anchor的通道顺序为：tx, ty, tw, th, conf, class1, class2, ...
    
    计算过程：
      1. 对tx和ty做sigmoid激活，然后与label中的tx,ty计算回归（MSE）损失（仅在conf=1时计算）
      2. tw和th直接与label中的tw,th计算回归（MSE）损失（仅在conf=1时计算）
      3. 对所有anchor用 BCEWithLogitsLoss 计算conf loss
      4. 对于conf=1的anchor，用 BCEWithLogitsLoss 计算类别损失
    """
    batch, total_channels, grid_h, grid_w = predict_feat.shape
    num_anchor = 3
    num_class = total_channels // num_anchor - 5
    
    # 调整形状为 (batch, num_anchor, 5+num_class, grid_h, grid_w)
    pred = predict_feat.view(batch, num_anchor, 5 + num_class, grid_h, grid_w)
    label = label_feat.view(batch, num_anchor, 5 + num_class, grid_h, grid_w)
    
    # 对tx, ty进行sigmoid激活，分别在通道索引0,1
    pred_tx = torch.sigmoid(pred[:, :, 0, :, :])
    pred_ty = torch.sigmoid(pred[:, :, 1, :, :])
    # tw, th直接保持原样（通道索引 2,3）
    pred_tw = pred[:, :, 2, :, :]
    pred_th = pred[:, :, 3, :, :]
    
    # 组合预测边界框
    pred_box = torch.stack([pred_tx, pred_ty, pred_tw, pred_th], dim=2)
    target_box = label[:, :, 0:4, :, :]
    
    # 置信度：通道索引 4
    pred_conf = pred[:, :, 4, :, :]
    target_conf = label[:, :, 4, :, :]
    
    # 类别预测：通道从5开始 (shape: (batch, num_anchor, num_class, grid_h, grid_w))
    pred_class = pred[:, :, 5:, :, :]
    target_class = label[:, :, 5:, :, :]
    
    # 仅在目标存在（target_conf==1）的anchor上计算回归和分类loss
    object_mask = (target_conf == 1)

    # 位置（边界框）损失：仅在object_mask为True的位置计算
    loss_box = 0.0
    if object_mask.sum() > 0:
        pred_box_pos = pred_box[object_mask]
        target_box_pos = target_box[object_mask]
        loss_box = F.mse_loss(pred_box_pos, target_box_pos, reduction="mean")
    
    # 置信度loss：对所有anchor计算 (使用BCEWithLogitsLoss)
    loss_conf = F.binary_cross_entropy_with_logits(pred_conf, target_conf, reduction="mean")
    
    # 分类loss：仅在object_mask为True的位置计算
    loss_class = 0.0
    if object_mask.sum() > 0:
        # 将object_mask扩展到类别维度上
        object_mask_class = object_mask.unsqueeze(2).expand_as(pred_class)
        pred_class_pos = pred_class[object_mask_class]
        target_class_pos = target_class[object_mask_class]
        loss_class = F.binary_cross_entropy_with_logits(pred_class_pos, target_class_pos, reduction="mean")
    
    total_loss = loss_box + loss_conf + loss_class
    return total_loss



if __name__ == '__main__':
   pass