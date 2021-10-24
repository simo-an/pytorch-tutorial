import torch
import numpy as np
from PIL import Image, ImageDraw

class_dict = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}

#将 xml 转化为 json
def parse_xml_to_dict(xml):
    if len(xml) == 0:
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])

    return {xml.tag: result}


def draw_bounding_boxes(
    image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    img_to_draw = Image.fromarray(image)
    img_boxes = boxes.to(torch.int64).tolist()
    draw = ImageDraw.Draw(img_to_draw)

    class_map = [k for k, v in class_dict.items()]

    for i, bbox in enumerate(img_boxes):
        draw.rectangle(bbox, width=2, outline='red')
        margin = 2
        draw.text((bbox[0] + margin, bbox[1] + margin),
                  class_map[labels[i] - 1], fill='red')

    return np.array(img_to_draw)
