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

import torch


def func(a, b, c):
    print(a, b , c)

func(*[1, 2, 3])

bbox = torch.Tensor([[1, 2, 3, 4], [2, 4, 6, 8]])

# bbox[:, [0, 2]] = 2 - bbox[:, [0, 2]]

user = {
    "name": "Tom",
    "age": 19
}

print(user.items())