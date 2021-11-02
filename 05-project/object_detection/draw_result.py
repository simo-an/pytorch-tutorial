import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import DefectDataset, class_dict
from PIL import Image, ImageDraw

def draw_bounding_boxes(
    image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor = None,
) -> torch.Tensor:
    img_to_draw = Image.fromarray(image)
    img_boxes = boxes.to(torch.int64).tolist()
    draw = ImageDraw.Draw(img_to_draw)

    class_map = [k for k, v in class_dict.items()]

    for i, bbox in enumerate(img_boxes):
        if scores is None or scores[i] > 0.45:
            draw.rectangle(bbox, width=2, outline='red')
            margin = 2
            show_text = class_map[labels[i] - 1] if scores is None else f'{class_map[labels[i] - 1]} - {scores[i]}'
            draw.text((bbox[0] + margin, bbox[1] + margin), show_text, fill='blue')

    return np.array(img_to_draw)

def plot_target_and_output(target, predict):
    data_transform = { 
        "val": None # transforms.Compose([transforms.ToTensor()]) 
    }
    val_dataset = DefectDataset('./', data_transform["val"], "val.txt")

    print(f"Image: {target['image_id'][0]}")
    
    image, target = val_dataset.__getitem__(target['image_id'][0])
    copy_image = np.array(image)

    plt.figure(1)
    plt.subplot(1, 2, 1)

    image = draw_bounding_boxes(
        np.array(image),
        target['boxes'],
        target['labels'],
    )
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    image = draw_bounding_boxes(
        copy_image,
        predict['boxes'],
        predict['labels'],
        predict['scores'],
    )
    plt.imshow(image)

    plt.show()