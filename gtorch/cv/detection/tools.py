import torch

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
