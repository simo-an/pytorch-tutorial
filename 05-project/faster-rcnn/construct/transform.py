from typing import List
import torch
from torch import Tensor
import torch.nn as nn

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

    def normalize(self, image: Tensor): # 标准化处理
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
    
    def resize(self, image, target):
        h, w = image.shape[-2:]
    
    def batch_images(self, images, size_divisible=32):
        pass

    def postprocess(self,
        result,
        image_shapes,
        raw_image_sizes
    ):
        pass

    def forward(self,
        images,         # type: List[Tensor]
        targets=None
    ):
        print(len(images))

        return images, targets
