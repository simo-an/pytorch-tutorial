from torch.utils.data import Dataset
import os
import torch
from PIL import Image
from lxml import etree
from os import path
from utils import draw_bounding_boxes, parse_xml_to_dict, class_dict


class SImageDataSet(Dataset):
    image_path: str # 图片路径
    anno_path: str  # 标注路径
    class_dict: dict # 类别词典
    transforms: any # 图片预处理器

    def __init__(self, transforms=None):
        base_path = path.join(os.getcwd(), '01-common', 'custom_dataset', 'tiny')

        self.image_path = path.join(base_path, 'cat_cup.jpg')
        self.anno_path = path.join(base_path, 'cat_cup.xml')
        self.class_dict = class_dict
        self.transforms = transforms

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = Image.open(self.image_path)

        with open(self.anno_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]

        boxes = []
        labels = []
        iscrowd = []
        
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[obj["name"]])
            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def get_height_and_width(self, idx):
        with open(self.anno_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width


import random
import matplotlib.pyplot as plt
import numpy as np


train_data_set = SImageDataSet(transforms=None)

for index in random.sample(range(0, len(train_data_set)), k=1):
    image, target = train_data_set[index]
    image = draw_bounding_boxes(
        np.array(image),
        target['boxes'],
        target['labels'],
    )
    plt.imshow(image)
    plt.show()