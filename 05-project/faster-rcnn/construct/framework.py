from collections import OrderedDict
from typing import List, Tuple
from torch import Tensor
import torch
import torch.nn as nn

from torchvision.ops import MultiScaleRoIAlign
from .rpn import RPN
from .roi_head import RoIHeads, FlattenHead, FastRCNNPredictor
from .transform import RCNNImageTransform

class FasterRCNNBase(nn.Module):
    transform: RCNNImageTransform
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

        # 将图像输入到 backbone 得到特征图 并存放在 有序字典中
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

class FasterRCNN(FasterRCNNBase):
    def __init__(
        self,
        backbone,   # 调用前指定 backbone
        roi_heads=None,
        num_classes=None,  # N + 1 (背景)
        # transform parameters
        min_size=600, max_size=1200,  # 最大、最小尺寸
        image_mean=None, image_std=None,  # 均值和方差
        # RPN parameters
        rpn_anchor_generator=None, rpn_head=None,  # conv -> 分类 & 回归
        rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,  # nms 前保留的建议区域数
        rpn_post_nms_top_n_train=1200, rpn_post_nms_top_n_test=800,  # nms 后保留的建议区域数
        rpn_nms_thresh_hold=0.7,  # nms 时的 IOU 阈值
        # TODO Q
        rpn_ps_iou_thresh_hold=0.7, rpn_ns_iou_thresh_hold=0.3,  # rpn 计算损失时，采集正负样本设置的阈值
        rpn_batch_size_per_image=256,  # rpn 计算损失时采样的样本数
        rpn_positive_fraction=0.5,  # rpn 计算损失时采样的样本中正样本所占的比例
        rpn_score_thresh=0.0,
        # Box parameters RoI Pooling
        box_roi_pool=None, box_flatten_head=None, box_predictor=None,
        # 移除低目标概率
        box_score_thresh_hold=0.05, box_nms_thresh_hold=0.05, box_detections_per_img=100,
        box_ps_iou_thresh=0.5, box_ns_iou_thresh=0.5,  # Faster R-CNN 计算损失时，正负样本设置的阈值
        box_batch_size_per_image=512,  # Faster R-CNN 计算损失时采样的样本数
        box_positive_fraction=0.25,  # Faster R-CNN 计算损失时采样的样本中正样本所占的比例
        bbox_reg_weights=None
    ):
        out_channels = backbone.out_channels 
        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train,
            testing=rpn_pre_nms_top_n_test,
        )
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train,
            testing=rpn_post_nms_top_n_test,
        )

        # 定义区域建议网络 RPN
        rpn = RPN(
            anchor_generator=rpn_anchor_generator,
            rpn_head=rpn_head,
            ps_iou_thresh_hold=rpn_ps_iou_thresh_hold,
            ns_iou_thresh_hold=rpn_ns_iou_thresh_hold,
            batch_size_per_image=rpn_batch_size_per_image,
            positive_fraction=rpn_positive_fraction,
            pre_nms_top_n=rpn_pre_nms_top_n,
            post_nms_top_n=rpn_post_nms_top_n,
            nms_thresh_hold=rpn_nms_thresh_hold,
            score_thresh_hold=rpn_score_thresh
        )

        # 没有指定 roi_heads 则用默认的
        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'],  # 进行 ROIPooling 的特征层
                output_size=[7, 7],
                sampling_ratio=2
            )
        
        # 没有指定 FlattenHead
        if box_flatten_head is None:
            resolution = box_roi_pool.output_size[0]  # 默认为 7 （输出 7 * 7）
            representation_size = 1024                # 输出 1024 长的向量
            box_flatten_head = FlattenHead(
                in_channels = (resolution ** 2) * out_channels, # 所有像素数目
                representation_size = representation_size
            )
        
        # 没有指定 分类与回归（即预测）
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor(
                in_channels=representation_size, # 即 经过 FlattenHead 输出的向量大小 默认 1024
                num_classes=num_classes
            )
        
        # 构建ROI Head
        roi_heads = RoIHeads(
            box_roi_pooling=box_roi_pool,
            box_flatten_head=box_flatten_head,
            box_predictor=box_predictor,
            ps_iou_thresh_hold=box_ps_iou_thresh,
            ns_iou_thresh_hold=box_ns_iou_thresh,
            batch_size_per_image=box_batch_size_per_image,
            positive_fraction=box_positive_fraction,
            bbox_reg_weight=bbox_reg_weights,
            score_thresh_hold=box_score_thresh_hold,
            nms_thresh_hold=box_nms_thresh_hold,
            detection_per_img=box_detections_per_img
        )

        # 设置默认均值与方差
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406] # R G B 三层的均值
        if image_std is None:
            image_std = [0.229, 0.224, 0.225] # R G B 三层的方差

        # 定义 transform
        transform = RCNNImageTransform(
            min_size=min_size,
            max_size=max_size,
            image_mean=image_mean,
            image_std=image_std
        )

        super(FasterRCNN, self).__init__(
            transform=transform,
            backbone=backbone,
            rpn=rpn,
            roi_heads=roi_heads
        )
