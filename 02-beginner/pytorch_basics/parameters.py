import torch
import torch.nn as nn
from torch.nn.modules.activation import PReLU

'''
    1. 一次性访问所有参数
'''
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

# print(net[0].state_dict())  # nn.Linear(4, 8) 的权重 weight、bias
# print(net[0].bias.data)
# print(net[0].bias.grad)
# print(*[(name, param) for name, param in net[0].named_parameters()])
# print(net.state_dict())


'''
    2. 从嵌套块收集参数
'''

def block1():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU()
    )

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))

# print(rgnet)

'''
    3. 内置初始化
'''
def init_normal(m):
    if isinstance(m, nn.Linear):
        # init 中含有初始化参数的工具
        nn.init.normal_(m.weight, mean=0, std=0.01) 
        nn.init.zeros_(m.bias)

net.apply(init_normal)      # 遍历 net 中的所有层
# print(net[0].weight.data[0])
# print(net[0].bias.data[0])

def init_constant(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net.apply(init_constant)
# print(net[0].weight.data[0])
# print(net[0].bias.data[0])

'''
    4. 都不同的块使用不同的初始化方法
'''
def xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    
def init_42(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 42)

net[0].apply(xavier)
net[2].apply(init_42)

# print(net[0].weight.data[0])
# print(net[2].weight.data[0])


'''
    5. 参数绑定
'''
shared = nn.Linear(8, 8)
net = nn.Sequential(
    nn.Linear(8, 8),
    nn.ReLU(),
    shared,
    nn.ReLU(),
    shared,
    nn.ReLU(),
    nn.Linear(8, 1)
)
print(net[2].weight.data[0] == net[4].weight.data[0])