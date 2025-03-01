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
    # predict_feat和label_feat形状：(b, c, h, w)
    # c = 3 * (5 + num_classes)
    num_anchors = 3
    b, c, h, w = predict_feat.shape
    num_classes = c // num_anchors - 5

    # 重构为 (b, 3, 5+num_classes, h, w)
    pred = predict_feat.view(b, num_anchors, 5 + num_classes, h, w)
    label = label_feat.view(b, num_anchors, 5 + num_classes, h, w)

    # 分离各部分：
    # 回归部分：tx,ty,tw,th 对应索引 0:4
    pred_reg = pred[:, :, 0:4, :, :]
    label_reg = label[:, :, 0:4, :, :]
    
    # 置信度：index 4
    pred_conf = pred[:, :, 4, :, :]
    label_conf = label[:, :, 4, :, :]
    
    # 分类部分：index 5:
    pred_cls = pred[:, :, 5:, :, :]
    label_cls = label[:, :, 5:, :, :]

    # 构造正样本 mask：只有当 label_conf==1 的地方，才计算回归和分类loss
    pos_mask = label_conf == 1  # 形状: (b, 3, h, w)

    # 1. 回归loss（仅对正样本计算），使用均方误差
    if pos_mask.sum() > 0:
        # 扩展mask，使得可以作用在回归通道上
        pos_mask_reg = pos_mask.unsqueeze(2).expand_as(pred_reg)
        reg_loss = F.mse_loss(pred_reg[pos_mask_reg], label_reg[pos_mask_reg], reduction='sum')
    else:
        reg_loss = torch.tensor(0., device=predict_feat.device)

    # 2. 置信度loss，对所有区域使用二元交叉熵（这里直接用 BCEWithLogitsLoss，不需要手动sigmoid）
    conf_loss = F.binary_cross_entropy_with_logits(pred_conf, label_conf, reduction='sum')

    # 3. 分类loss，仅对正样本计算。这里题目要求先对每个类别通道做sigmoid，再与标签（独热编码）做 BCE
    if pos_mask.sum() > 0:
        pos_mask_cls = pos_mask.unsqueeze(2).expand_as(pred_cls)
        # 对预测的类别做sigmoid
        pred_cls_sig = torch.sigmoid(pred_cls)
        cls_loss = F.binary_cross_entropy(pred_cls_sig[pos_mask_cls],
                                          label_cls[pos_mask_cls],
                                          reduction='sum')
    else:
        cls_loss = torch.tensor(0., device=predict_feat.device)

    total_loss = reg_loss + conf_loss + cls_loss
    return total_loss



if __name__ == '__main__':
   pass