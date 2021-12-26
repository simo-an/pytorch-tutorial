import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

'''
用双隐层神经网络解决异或问题
'''

inputs = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]]).to(torch.float32)
labels = torch.tensor([[1], [0], [1], [0]]).to(torch.float32)

class XORNet(nn.Module):
    def __init__(self) -> None:
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)
    
    def forward(self, input):
        result = F.sigmoid(self.fc1(input))
        result = F.sigmoid(self.fc2(result))
        result = F.sigmoid(self.output(result))

        return result

model = XORNet().to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1, momentum=0.9)

for i in range(1000):
    optimizer.zero_grad()
    result = model(inputs)
    loss = criterion(result, labels)

    loss.backward()
    optimizer.step()

    if (i + 1) % 10 == 0:
        print (f'Step [{i+1}], Loss: {loss.item():.4f}')

test_result = model(inputs)

print(f'Input: {inputs}')
print(f'Truth: {labels}')
print(f'Pred: {test_result}')
