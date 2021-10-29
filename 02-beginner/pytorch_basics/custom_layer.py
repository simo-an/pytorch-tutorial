import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    1. 最基本的层
'''
class CustomLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        return X - X.mean()
    
layer = CustomLayer()
loss = layer(torch.FloatTensor([1,2,3,4,5]))

'''
    2. 带有参数的层
'''
class ParamedLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units)) # in_units * units 大小的矩阵
        self.bias = nn.Parameter(torch.randn(units,))
    
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data # w*x + b

        return F.relu(linear)

plinear = ParamedLinear(5, 3)
print(plinear.weight)
print(plinear.bias)
print(plinear(torch.randn(2, 5)))