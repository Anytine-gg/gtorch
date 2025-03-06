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
    n,c,w,h = predict_feat.shape
    nClass = c // 3 - 5
    assert w == h, "Width and height of feature map must be same!"
    assert (
        w == 52 or w == 26 or w == 13
    ), """The size of feature map is not supported!
            (Supported size: 52*52 26*26 13*13)
        """
    if w == 52:
        stride = 8
    elif w == 26:
        stride = 16
    else:  # w == 13
        stride = 32
    # 每一行为
    pred_feat = pred_feat.permute(0, 2, 3, 1).reshape(-1, 5 + nClass)

    

def nms(
    pred_feat: torch.Tensor,
    conf_threshold=0.7,
    nms_threshold=0.5,
    anchor_size=[
        [(10, 13), (16, 30), (33, 23)],  # 52*52
        [(30, 61), (62, 45), (59, 119)],  # 26*26
        [(116, 90), (156, 198), (373, 326)],  # 13*13
    ],
):
    _, c, w, h = pred_feat.shape
    nClass = c // 3 - 5
    assert w == h, "Width and height of feature map must be same!"
    assert (
        w == 52 or w == 26 or w == 13
    ), """The size of feature map is not supported!
            (Supported size: 52*52 26*26 13*13)
        """
    if w == 52:
        stride = 8
        anchor_size = anchor_size[0]
    elif w == 26:
        stride = 16
        anchor_size = anchor_size[1]
    else:  # w == 13
        stride = 32
        anchor_size = anchor_size[2]
    anchor_size = torch.tensor(
        anchor_size, dtype=torch.float32, device=pred_feat.device
    )
    pred_feat = pred_feat.permute(0, 2, 3, 1).reshape(-1, 5 + nClass)
    x = torch.arange(w)
    y = torch.arange(w)
    # 生成[0 0;1 0;2 0;...;0 51;...;51 51]网格，记录位置。第一列为cx，第二列为cy
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)
    grid = grid.repeat_interleave(3, 0).to(pred_feat.device)
    anchor_size = anchor_size.repeat(w * h, 1)  # 记录每一行anchor的尺寸
    # pred_feat的每个行是一个anchor检测头，前5个是tx,ty,tw,th,conf，后面是classes置信度
    # 最后四列是anchor所在的cx,cy,以及anchor的w和h
    pred_feat = torch.cat([pred_feat, grid, anchor_size], dim=1)
    torch.sigmoid_(pred_feat[:,4:5+nClass])
    conf_col = pred_feat[:, 4]
    mask = conf_col >= conf_threshold
    pred_feat = pred_feat[mask]  # 筛选掉bbox置信度小于threshold的框
    # tx ty tw th 转为 bx by bw bh
    pred_feat[:, 0] = torch.sigmoid(pred_feat[:, 0]) + pred_feat[:, -4]  # cx
    pred_feat[:, 1] = torch.sigmoid(pred_feat[:, 1]) + pred_feat[:, -3]  # cy
    pred_feat[:, 0:2] *= stride
    pred_feat[:, 2] = pred_feat[:, -2] * torch.exp(pred_feat[:, 2])
    pred_feat[:, 3] = pred_feat[:, -1] * torch.exp(pred_feat[:, 3])
    pred_feat = pred_feat[:, :-4]  # 舍去后4列

    reserved_anchors = [] #符合条件的anchor
    # print(pred_feat[:,0:5])
    while pred_feat.numel() != 0:
        # 属于类别置信度的那部分
        class_conf = pred_feat[:, 5 : nClass + 5]
        max_class_conf = torch.max(class_conf)
        if max_class_conf < conf_threshold:  # 若最大的类别置信度小于threshold，退出
            break
        argmax_class_conf = torch.unravel_index(
            torch.argmax(class_conf), class_conf.shape
        )
        # 获取最大值，得到位置
        argmax_anchor, argmax_class = argmax_class_conf
        argmax_class += 5
        # 拿出class conf最大的，与其他anchor做iou，若大于threshold, 舍弃
        reserved_anchors.append(pred_feat[argmax_anchor,:4]) #保留最大置信度anchor
        origin = pred_feat[:, :4]
        target = pred_feat[argmax_anchor,:4].repeat(pred_feat.size(0), 1) #手动广播
        iou = calc_IoU_tensor(origin,target)
        pred_feat = pred_feat[iou < nms_threshold] # 大于阈值的被舍弃
    if reserved_anchors == []:
        return torch.empty((0, 4), device=pred_feat.device)
    reserved_anchors = torch.stack(reserved_anchors)
    return reserved_anchors


if __name__ == "__main__":

    feat = torch.rand(1, 75, 52, 52).to('cuda')
    nms(feat)
