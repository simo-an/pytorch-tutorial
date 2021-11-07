from typing import Dict, List, Optional, Tuple
from torch import Tensor
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .image_list import ImageList

def resize_boxes(boxes, raw_size, new_size):
    # type: (Tensor, List(int), List(int)) -> Tensor

    ratios_height = new_size[0] / raw_size[0]
    ratios_weight = new_size[1] / raw_size[1]

    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * torch.tensor(ratios_weight)
    xmax = xmax * torch.tensor(ratios_weight)
    ymin = ymin * torch.tensor(ratios_height)
    ymax = ymax * torch.tensor(ratios_height)

    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

def get_max_shape(shape_list: List[List[int]]) -> List[int]:
    max_shape = shape_list[0]
    for sub_shape in shape_list[1:]:
        for idx, shape in enumerate(sub_shape):
            max_shape[idx] = max(max_shape[idx], shape)

    return max_shape 

class RCNNImageTransform(nn.Module):
    def __init__(self, 
        min_size,   # 最小尺寸
        max_size,   # 最大尺寸
        image_mean, # 均值
        image_std   # 方差
    ):
        super(RCNNImageTransform, self).__init__()
        self.min_size=min_size      # 图片最小边长
        self.max_size=max_size      # 图片最大边长
        # 图片在标准化时需要(X - image_mean) / image_std
        self.image_mean=image_mean
        self.image_std=image_std

    '''对图片做标准化处理'''
    def normalize(self, image: Tensor):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)

        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k: List[int]) -> int:
        index = int(
            torch.empty(1)
                 .uniform_(0., float(len(k)))
                 .item()
        )

        return k[index]

    '''将图片大小调整到指定的范围'''
    def resize(self, image, target):
        height, weight = image.shape[-2:]
        max_size = max([height, weight])
        min_size = min([height, weight])

        scale_factor = self.min_size / min_size

        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size

        # interpolate 使用插值的方法来缩放图片
        image = F.interpolate(
            input=image[None], 
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=False
        )[0]

        if target is None:
            return image, target

        bbox = target['boxes']
        bbox = resize_boxes(bbox, [height, weight], image.shape[-2:])

        target['boxes'] = bbox

        return image, target

    '''将一批图片打包成一个batch'''
    def batch_images(self, 
        images,             # 输入的一批图片
        size_divisible=32   # 将图片的宽高调整到 size_divisible 的整数倍
    ):
        # type: (List[Tensor], int) -> Tensor
        # 获取一个batch的所有图片中最大的 channel、height、width
        max_shape = get_max_shape([list(img.shape) for img in images])
        stride = float(size_divisible)

        # 宽高向上调整到 stride 的整数倍
        max_shape[1] = int(math.ceil(max_shape[1] / stride) * stride)
        max_shape[2] = int(math.ceil(max_shape[2] / stride) * stride)

        batch_size = [len(images)] + max_shape

        # 创建shape为batch，且全0的tensor, 与 image[0] 设备、类型相同
        batch_image = images[0].new_full(batch_size, 0)

        for img, pad_img in zip(images, batch_image):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batch_image


    '''对网络的预测结果进行后处理 - 主要将 bboxes 还原到原图尺寸上'''
    def postprocess(self,
        result,
        image_shapes,
        raw_image_sizes
    ):
        # 测试时候需要
        pass

    def forward(self,
        images,         # type: List[Tensor]
        targets=None    # type: Optional[List[Dict[str, Tensor]]]
    ):
        '''标准化并调整宽高'''
        for idx in range(len(images)):
            image = images[idx]
            target =  targets[idx] if targets is not None else None
            image = self.normalize(image)

            image, target = self.resize(image, target)

            images[idx] = image
            if targets is not None:
                targets[idx] = target
        # 记录调整后的图片大小
        image_sizes = [img.shape[-2:] for img in images]

        images = self.batch_images(images)
        image_size_list: List[Tuple[int, int]] = []

        for img_size in image_sizes:
            image_size_list.append((img_size[0], img_size[1]))

        image_list = ImageList(images, image_size_list)

        return image_list, targets