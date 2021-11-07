from typing import List, Tuple
from torch import Tensor
from collections import OrderedDict

import torch
import torch.nn as nn

from .transform import RCNNImageTransform
from torchvision.models.detection.rpn import RegionProposalNetwork as RPN
from torchvision.models.detection.roi_heads import RoIHeads

class FasterRCNNBase(nn.Module):
    transform: RCNNImageTransform
    rpn: RPN
    roi_head: RoIHeads
    def __init__(self, transform, backbone, rpn, roi_heads):
        super(FasterRCNNBase, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    '''正向传播 image 和 target 为从 __getitem__ 中获取的'''
    def forward(self, images: List[Tensor], targets=None):
        # 训练时需要target，测试时候不需要
        if self.training and targets is None:
            raise ValueError('In Training mode, targets are required')

        # 这边默认数据格式都是正确的，不再做检查

        raw_image_shape: List[Tuple[int, int]] = []

        for image in images:
            raw_image_shape.append((image.shape[1], image.shape[2]))

        # Normalize & Resize
        images, targets = self.transform(images, targets)  # 对图像进行预处理 ImageList

        # 将图像输入到 backbone 得到特征图
        feature_maps = self.backbone(images.image_list)
        feature_maps = OrderedDict([('0', feature_maps)]) # 一层特征图

        # 将特征图输入到RPN中，得到 proposals 以及对应的 loss
        proposals, proposal_losses = self.rpn(images, feature_maps, targets)

        # 将 feature、proposal、image、targets 传入 roi_heads 得到 检测目标以及损失
        detections, detector_losses = self.roi_heads(
            feature_maps,
            proposals, 
            images, 
            targets
        )
        # 将预测的bboxes还原到原始图像尺度上
        detections = self.transform.postprocess(
            detections, 
            images, 
            raw_image_shape
        )

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses, detections