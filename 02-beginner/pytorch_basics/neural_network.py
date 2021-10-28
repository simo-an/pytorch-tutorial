import torch
import torch.nn as nn
import torch.nn.functional as F
####
# 模型构造
####
'''
1. 入门 - 多层感知机的两种实现
'''
mlp_model1 = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
    
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
mlp_model2 = MLP()

'''
2. 自定义 Sequential
'''
class CustomSequential(nn.Module):
    def __init__(self, *layer_list):
        super().__init__()
        for layer in layer_list:
            self._modules[layer] = layer
    
    def forward(self, X):
        for layer in self._modules.values():
            X = layer(X)
        
        return X

'''
3. 继承 nn.Module 与 Sequential 混用
'''
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.seq_net(X))

mlp_model3 = nn.Sequential(NestMLP(), nn.Linear(16, 10))