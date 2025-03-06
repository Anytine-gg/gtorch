def get_default_anchor_size(type="yolov3"):
    defalut_yolov3_size =  [
            [(10, 13), (16, 30), (33, 23)],  # 52*52
            [(30, 61), (62, 45), (59, 119)],  # 26*26
            [(116, 90), (156, 198), (373, 326)],  # 13*13
        ]
    
    if type == "yolov3":
        return defalut_yolov3_size
