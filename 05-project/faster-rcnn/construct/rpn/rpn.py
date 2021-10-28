import torch
import torch.nn as nn
import torch.nn.functional as F

from .box import BoxBorder

'''
anchor box 生成器
'''
class AnchorsGenerator(nn.Module):
    def __init__(self,
                 anchor_sizes=(128, 256, 512),  # anchor box 大小
                 aspect_ratios=(0.5, 1.0, 2.0)  # 每种 anchor box 高宽比
                 ):
        super().__init__()


'''
通过滑动窗口计算预测目标概率与bbox regression参数
'''
class RPNHead(nn.Module):
    def __init__(self,
                 in_channels,  # 输入的特征图的通道数
                 num_anchors  # 每一个位置输出的anchor box数目
    ):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels,
                              kernel_size=3, stride=1, padding=1)
        self.classify_conv = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.reg_conv = nn.Conv2d(in_channels, 4 * num_anchors, kernel_size=1)

        # 参数初始化
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        classfy_list = []
        reg_list = []
        for idx, feature in features:
            f = F.relu(self.conv(feature))
            classfy_list.append(self.classify_conv(f))
            reg_list.append(self.reg_conv(f))

        return classfy_list, reg_list


'''
Region Proposal Network
'''
class RPN(nn.Module):
    anchor_generator: AnchorsGenerator
    rpn_head: RPNHead
    box_coder: BoxBorder

    def __init__(self,
                 anchor_generator,  # anchor box 生成器
                 rpn_head,
                 ps_iou_thresh_hold,  # positive sample IOU 阈值
                 ns_iou_thresh_hold,  # negative sample IOU 阈值
                 batch_size_per_image,  # 每张图片采样数
                 positive_fraction,  # 采样数中正样本的比例
                 pre_nms_top_n,  # nms 前保留的建议区域数
                 post_nms_top_n,  # nms 后保留的建议区域数
                 nms_thresh_hold,  # nms 的 IOU 阈值
                 score_thresh_hold=0.0  # 预测是正样本的分数阈值
        ):
        super(RPN, self).__init__()
        self.anchor_generator = anchor_generator
        self.rpn_head = rpn_head
        self.box_coder = BoxBorder(weights=(1., 1., 1., 1.))
        # 计算 anchors 与真实的 bbox 的 IOU

    def forward(self, image_list, features, targets=None):
        features = list(features)
        objectness, pred_bbox_deltas = self.rpn_head(features)