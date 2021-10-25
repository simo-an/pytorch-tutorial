import torch
import torch.nn as nn

'''
anchor box 生成器
'''
class AnchorsGenerator(nn.Module):
    def __init__(self, 
        anchor_sizes=(128, 256, 512), # anchor box 大小
        aspect_ratios=(0.5, 1.0, 2.0) # 每种 anchor box 高宽比
    ):
        super().__init__()

'''
通过滑动窗口计算预测目标概率与bbox regression参数
'''
class RPNHead(nn.Module):
    def __init__(self, 
        in_channels, # 输入的特征图的通道数
        num_anchors  # 每一个位置输出的anchor box数目 
    ):
        super().__init__()

'''
Region Proposal Network
'''
class RPN(nn.Module):
    def __init__(self, 
        anchor_generator,  # anchor box 生成器
        rpn_head,
        ps_iou_thresh_hold, # positive sample IOU 阈值
        ns_iou_thresh_hold, # negative sample IOU 阈值
        batch_size_per_image, # 每张图片采样数
        positive_fraction, # 采样数中正样本的比例
        pre_nms_top_n, # nms 前保留的建议区域数
        post_nms_top_n, # nms 后保留的建议区域数
        nms_thresh_hold, # nms 的 IOU 阈值
        score_thresh_hold=0.0 # 预测是正样本的分数阈值
    ):
        super().__init__()
