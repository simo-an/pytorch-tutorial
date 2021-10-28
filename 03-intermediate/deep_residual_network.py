import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

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

# 3x3 convolution
def conv3m3(in_channels, out_channels, stride=1, padding=1):
    return nn.Conv2d(
        in_channels,
        out_channels, 
        kernel_size=3, 
        stride=stride, 
        padding=padding
    )

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3m3(in_channels, out_channels, stride)
        