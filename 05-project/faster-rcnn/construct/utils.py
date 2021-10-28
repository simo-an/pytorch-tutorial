from typing import List
from torch import Tensor
import torch

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