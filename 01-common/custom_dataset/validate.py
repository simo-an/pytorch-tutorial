import os
import random
import utils
import main
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.dont_write_bytecode = True

# VOC 数据集应放在根目录下 pytorch-tutorial/VOCdevkit
train_data_set = main.VOCDataset(
    voc_root=os.getcwd(),
    year='2012',
    transforms=None,
    text_name='train.txt'
)

for index in random.sample(range(0, len(train_data_set)), k=5):
    image, target = train_data_set[index]
    image = utils.draw_bounding_boxes(
        np.array(image),
        target['boxes'],
        target['labels'],
    )
    plt.imshow(image)
    plt.show()
