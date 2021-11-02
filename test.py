# from collections import OrderedDict

# import torch

# d1 = OrderedDict([('0', 'A'), ('1', 'B')])

# for k, v in d1.items():
#     print(f'{k}: {v}')

# k1 = {'a': 1, 'b': 3}
# k2 = {'b': 2, 'a': 4}

# k_set = {}

# k_set.update(k1)
# k_set.update(k2)

# print(k_set)

# ra = (1.0, 2.0, 4.0) * 8

# print(ra)

# image = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# mean = torch.tensor([2, 5, 8])
# std = torch.tensor([2, 2, 2])

# print(
#     (image - mean[:, None, None]) / std[:, None, None]
# )

# import argparse
# #获取ArgumentParser对象
# parser = argparse.ArgumentParser()
# #添加参数
# parser.add_argument('--verCode', type=int)
# parser.add_argument('--appID', type=str)

# #args是一个命名空间
# args = parser.parse_args()

# print(args)
# print(args.appID)
# print(args.verCode)

import math
import torch


# def func(a, b, c):
#     print(a, b , c)

# func(*[1, 2, 3])

# bbox = torch.Tensor([[1, 2, 3, 4], [2, 4, 6, 8]])

# # bbox[:, [0, 2]] = 2 - bbox[:, [0, 2]]

# user = {
#     "name": "Tom",
#     "age": 19
# }

# print(user.items())

# image = torch.tensor([
#     [[1, 2], [2, 3]], # R
#     [[2, 4], [4, 6]], # G
#     [[4, 6], [4, 2]]  # B
# ])

# mean = torch.tensor([2, 3, 4])
# std = torch.tensor([2, 1, 2])

# print(mean[:, None, None])
# print(std[:, None, None])

# result = (image - mean[:, None, None]) / std[:, None, None]

# print(result)

# k = (678,)
# index = int(torch.empty(1).uniform_(0., float(len(k))).item())
# print(k[index])

# box = torch.tensor([[1, 2, 3, 4], [11, 12, 13, 14]])

# print(box.unbind(1))

# print(torch.stack(box.unbind(1), dim=1))

# a = 1
# b = 'A'
# tar = {
#     'a': a,
#     'b': b
# }
# print(tar)

# shape_list = [
#     [3, 100, 200], [4, 50, 120], [3, 120, 180]
# ]
# def get_max_attr(shape_list):
#     max_shape = shape_list[0]
#     for sub_shape in shape_list[1:]:
#         for idx, shape in enumerate(sub_shape):
#             max_shape[idx] = max(max_shape[idx], shape)
    
#     return max_shape

# batch_shape = [2, 3, 4, 4]
# img = torch.ones([3, 2, 2])

# fill_image = torch.zeros(batch_shape)

# batch_image = img[0].new_full(batch_shape, 0)


# image1 = torch.zeros([3, 4, 5])
# image2 = torch.tensor([
#     [[1, 2], [3, 4]],
#     [[2, 4], [6, 8]],
# ])

# image1[: image2.shape[0], : image2.shape[1], : image2.shape[2]].copy_(image2)

# print(image1)

# result = []

# t = torch.tensor([1, 3, 2, 5, 3, 2])

# result.append(torch.max(t).item())

# print(result)

# image = torch.ones([1, 28, 28])
# print(image.shape)

anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

print(aspect_ratios)