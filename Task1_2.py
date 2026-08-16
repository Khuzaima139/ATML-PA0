import types
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import os
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def broken_forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    # out += identity  
    out = self.relu(out)
    return out

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)

val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False

for i in range(3):
    model.layer4[i].forward = types.MethodType(broken_forward, model.layer4[i])
print("skip connections disabled in layer4")

model.fc = nn.Identity()
model.eval()
model = model.to(device)

def get_features(loader):
    feats, labels = [], []
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            out = model(images.to(device))
            feats.append(out.cpu())
            labels.append(targets)
            if i % 25 == 0:
                print(f"{i}/{len(loader)}")
    return torch.cat(feats), torch.cat(labels)

if os.path.exists("resnet152_noskip_features.pt"):
    print("loading cached features...")
    Xtr, ytr, Xval, yval = torch.load("resnet152_noskip_features.pt")
else:
    print("extracting train features...")
    Xtr, ytr = get_features(train_loader)
    print("extracting val features...")
    Xval, yval = get_features(val_loader)
    torch.save((Xtr, ytr, Xval, yval), "resnet152_noskip_features.pt")

head = nn.Linear(2048, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(head.parameters(), lr=0.001)

train_feat_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)
val_feat_loader = DataLoader(TensorDataset(Xval, yval), batch_size=256)

num_epochs = 15
for epoch in range(num_epochs):
    head.train()
    correct, total, running_loss = 0, 0, 0.0
    for xb, yb in train_feat_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = head(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct += out.argmax(1).eq(yb).sum().item()
        total += yb.size(0)
    print(f"Epoch {epoch+1}: train acc {100*correct/total:.2f}%")

    head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_feat_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = head(xb)
            correct += out.argmax(1).eq(yb).sum().item()
            total += yb.size(0)
    print(f"Epoch {epoch+1}: val acc {100*correct/total:.2f}%\n")