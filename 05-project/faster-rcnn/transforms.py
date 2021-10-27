import random
import torch
from torchvision.transforms import functional as F
from dataset import Target

'''组合多个Transforms'''
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        
        return image, target

'''PIL 图像 转化为 Tensor'''
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)

        return image, target

'''随机水平翻转图像以及bboxes'''
class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image: torch.Tensor, target: Target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)              # 水平翻转图像
            
            print(target)

            bbox = target.boxes
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target.boxes = bbox
        
        return image, target