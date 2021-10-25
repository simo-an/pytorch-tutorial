import os
import datetime

import torch
import numpy
import matplotlib.pyplot as plt


print('Hello Python!')

boxes = torch.tensor([[0, 0, 2, 2], [0, 0, 3, 3]])
area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # 得到面积

print(boxes)
print(area)