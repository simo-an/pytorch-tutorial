import torch
import torch.nn as nn

class FlattenHead(nn.Module):
    def __init__(self, 
        in_channels,        # 输入通道数
        representation_size # 经过全连接Flatten后，输出的中间表示大小
    ):
        super().__init__()

class FastRCNNPredictor(nn.Module):
    def __init__(self,
        in_channels, # 输入通道数 即 representation_size
        num_classes  # 输出类别数（含有背景）
    ):
        super().__init__()

class RoIHeads(nn.Module):
    def __init__(self,
                 box_roi_pooling,       # ROI Pooling Head
                 box_flatten_head,      # Flatten Head
                 box_predictor,         # 分类与回归
                 # Faster RCNN training
                 ps_iou_thresh_hold, ns_iou_thresh_hold,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weight,
                 # Faster RCNN inference
                 score_thresh_hold,
                 nms_thresh_hold,
                 detection_per_img     # 每张图片最多检测目标数
    ):
        super().__init__()
        