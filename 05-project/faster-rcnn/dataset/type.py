from torch import Tensor

class Target(object):
    boxes: Tensor
    labels: Tensor
    image_id: Tensor
    iscrowd: Tensor
    area: Tensor
    def __init__(self,
        boxes: Tensor,
        labels: Tensor,
        image_id: Tensor,
        iscrowd: Tensor,
        area: Tensor,
    ):
        self.boxes = boxes
        self.labels = labels
        self.image_id = image_id
        self.area = area
        self.iscrowd = iscrowd