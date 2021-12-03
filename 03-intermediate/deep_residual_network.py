import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch import Tensor
'''
残差网络
paper: https://arxiv.org/pdf/1512.03385.pdf
'''

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 80
batch_size = 100
learning_rate = 0.001

# Image preprocessing modules
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(
    root='dataset/',
    train=True,
    transform=transform,
    download=True
)

test_dataset = torchvision.datasets.CIFAR10(
    root='dataset/',
    train=False,
    transform=transforms.ToTensor()
)

# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False
)

expansion: int = 4

class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, stride=1) -> None:
        super(Bottleneck, self).__init__()
        out_channels = base_channels * expansion
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,  bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, input: Tensor):
        output = self.bottleneck(input)
        raw = self.downsample(input)

        result = output + raw

        return self.relu(result)


class Resnet50(nn.Module):
    in_channels: int
    num_classes: int
    cur_channels: int
    def __init__(self, in_channels = 3, num_classes = 1000) -> None:
        super(Resnet50, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.cur_channels = 64
        
        self.block1 = self.make_block(64, 3)  # 64   -> 256
        self.block2 = self.make_block(128, 4) # 256  -> 512
        self.block3 = self.make_block(256, 6) # 512  -> 1024
        self.block4 = self.make_block(512, 3) # 1024 -> 2048

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, num_classes)
    
    def forward(self, input: Tensor):
        input = nn.ReLU()(self.conv1(input))
        input = self.maxpool(input)

        input = self.block1(input)
        input = self.block2(input)
        input = self.block3(input)
        input = self.block4(input)

        input = self.avgpool(input)
        input = self.fc(torch.flatten(input, 1)) # input.squeeze()

        return input

    def make_block(self, in_channels, bottleneck_num, stride=1) -> nn.Sequential:
        bottlenecks = []
        # 将当前 channels 调整到 in_channels
        bottlenecks.append(Bottleneck(self.cur_channels, in_channels, stride))
        self.cur_channels = in_channels * expansion # 经过一个Bottleneck则channels变为原来expansion倍

        for _ in range(1, bottleneck_num):
            bottlenecks.append(Bottleneck(self.cur_channels, in_channels, stride))

        return nn.Sequential(*bottlenecks)


model = Resnet50(in_channels=3, num_classes=10).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
# torch.save(model.state_dict(), 'resnet.ckpt')
