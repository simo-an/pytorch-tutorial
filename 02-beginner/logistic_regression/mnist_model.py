import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.modules import activation

class MnistModel(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MnistModel, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 6 14 14

        self.conv3 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 16 7 7

        self.conv5 = nn.Conv2d(16, 20, kernel_size=3, stride=1) # 20 5 5

        self.classifier = nn.Sequential(
            nn.Linear(20 * 5 * 5, 500),
            nn.ReLU(True),
            nn.Linear(500, 200),
            nn.ReLU(True),
            nn.Linear(200, self.num_classes)
        )

    def forward(self, images):
        images = self.conv1(images)
        images = self.conv2(images)
        images = self.pool1(images)
        images = self.conv3(images)
        images = self.conv4(images)
        images = self.pool2(images)
        images = self.conv5(images)

        images = images.view(-1, 20 * 5 * 5)

        return self.classifier(images)