import torch
import numpy as np
from PIL import Image, ImageDraw

class_dict = {
    "cat": 1,
    "cup": 2
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
