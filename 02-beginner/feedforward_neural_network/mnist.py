import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as TF


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 2
batch_size = 100
lr = 0.001

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(
    root='dataset',
    train=True,
    transform=TF.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='dataset',
    train=False,
    transform=TF.ToTensor()
)

# Data loader (input pipeline)
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

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, images):
        images = self.fc1(images)
        images = self.relu(images)
        images = self.fc2(images)

        return images
    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training
for epoch in range(num_epochs):
    for idx, [images, labels] in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        # Backward & Optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx + 1) % 100 == 0:
            print (f'Epoch [{epoch+1}], Step [{idx+1}], Loss: {loss.item():.4f}')



# Testing
with torch.no_grad():
    total = 0
    correct = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        # _ 最大值 predicted 最大值的索引 （outputs.data [100, 10]）
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    print(f'Accuracy is {100 * correct / total} %')