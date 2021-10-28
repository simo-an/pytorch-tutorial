from typing import List, Tuple
from torch import Tensor

class ImageList(object):
    def __init__(self, image_list, image_size_list):
        # type: (Tensor, List[Tuple[int, int]]) -> None
        self.image_list = image_list
        self.image_size_list = image_size_list
    
    def to(self, device):
        return ImageList(
            self.image_list.to(device),
            self.image_size_list
        )