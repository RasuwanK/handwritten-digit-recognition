import torch
import torchvision
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
import torchvision.ops as ops
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Modern 2026 data augmentation strategy
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader

# Modern 2026 data augmentation strategy
transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.RandomGrayscale(p=0.2), # Some digits might be color-coded
    transforms.RandomRotation(degrees=15),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

class DeformableNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # Deformable Layer setup
        self.offset_conv = nn.Conv2d(32, 18, kernel_size=3, padding=1)
        self.deform_conv = ops.DeformConv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        # 64 channels * 7 height * 7 width = 3136 features
        self.fc1 = nn.Linear(64 * 7 * 7, 128) 
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Stage 1: Standard Conv + Pool -> (32, 14, 14)
        x = torch.relu(self.conv1(x))
        x = self.pool(x) 
        
        # Stage 2: Deformable Conv + Pool -> (64, 7, 7)
        offsets = self.offset_conv(x)
        x = torch.relu(self.deform_conv(x, offsets))
        x = self.pool(x) 
        
        # Flatten and Classify
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeformableNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Prevents over-fitting

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, 
                                                steps_per_epoch=len(train_loader), epochs=10)

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(classification_report(all_labels, all_preds))
# Look specifically for the "F1-Score" per digit

# Create a path
PATH = "../models/classifier/mnist_deformable_net.pth"

# Save only the weights
torch.save(model.state_dict(), PATH)
print(f"Model weights saved to {PATH}")