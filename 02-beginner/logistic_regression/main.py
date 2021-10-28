import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as TF

from mnist_model import MnistModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
input_size = 1
num_classes = 10
num_epochs = 5
batch_size = 100
lr = 0.01

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

model = MnistModel(input_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

# Training the model
for epoch in range(num_epochs):
    for idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        # backward & optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Step {idx +1} => Loss {loss.item()}")

# Test the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    print(f"The accuracy is: {100 * correct / total}%")