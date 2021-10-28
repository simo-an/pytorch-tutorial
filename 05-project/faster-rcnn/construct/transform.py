from typing import Dict, List, Optional, Tuple
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import resize_boxes
from .image_list import ImageList

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
        

    '''对网络的预测结果进行后处理 - 主要将 bboxes 还原到原图尺寸上'''
    def postprocess(self,
        result,
        image_shapes,
        raw_image_sizes
    ):
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
