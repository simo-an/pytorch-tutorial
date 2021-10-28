import math
import torch
from typing import List
from torch import Tensor

class BoxBorder(object):
    def __init__(self, weights, bbox_xform_clip=math.log(1000/16.)):
        super().__init__()
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip
    
    def encode(self, reference_boxes, proposals):
        pass

    def encode_single(self, reference_boxes, proposals):
        pass

    def decode(self, rel_codes, boxes):
        pass

    def decode_single(self, rel_codes, boxes):
        pass