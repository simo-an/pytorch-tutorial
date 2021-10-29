import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    1. 加载和保存Tensor

m1 = torch.randn([4, 4])
torch.save(m1, 'm1-file')

m2 = torch.load('m1-file')
print(m2)
# dict list 都可以
'''

'''
    2. 加载和保存模型参数
'''

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        return self.out(F.relu(self.hidden(x)))

net = MLP()
x = torch.randn((2, 20))
# print(net(x))
# 将模型的参数存储到 mlp.params
torch.save(net.state_dict(), 'mlp.params')

copy_net = MLP()
copy_net.load_state_dict(torch.load('mlp.params'))
copy_net.eval()
# print(copy_net.state_dict())
# print(net(x) == copy_net(x)) -> 输出全部为 True

