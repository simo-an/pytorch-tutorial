import torch
import torch.nn as nn

class RCNNImageTransform(nn.Module):
    def __init__(self, 
        min_size,   # 最小尺寸
        max_size,   # 最大尺寸
        image_mean, # 均值
        image_std   # 方差
    ):
        super().__init__()