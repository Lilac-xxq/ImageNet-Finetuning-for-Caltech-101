import torch
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

# 图像增强与预处理
def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return train_transforms, val_transforms

# 数据集划分
def split_dataset(root_dir):
    dataset = datasets.ImageFolder(root=root_dir)
    targets = dataset.targets
    class_indices = {cls: [] for cls in set(targets)}
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)
    train_indices, val_indices = [], []
    for indices in class_indices.values():
        if len(indices) < 31:
            train_indices += indices[:int(0.7 * len(indices))]
            val_indices += indices[int(0.7 * len(indices)) :]
        else:
            train_indices += indices[:30]
            val_indices += indices[30:]
    return torch.utils.data.Subset(dataset, train_indices), torch.utils.data.Subset(dataset, val_indices)

# 数据加载器构建
def get_loaders(data_dir, batch_size):
    train_transforms, val_transforms = get_transforms()
    train_data, val_data = split_dataset(data_dir)
    train_data.dataset.transform = train_transforms
    val_data.dataset.transform = val_transforms
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# 模型构建
def build_model(num_classes, weights=ResNet18_Weights.DEFAULT):
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# 训练与验证
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """
    训练模型并在每个epoch后进行验证，返回训练和验证指标历史。
    """
    best_val_acc = 0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # 验证
        model.eval()
        correct = 0
        total = 0
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_loss = val_running_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
    # Return the history lists and best accuracy
    return best_val_acc, train_losses, train_accs, val_losses, val_accs

# 设置随机数种子
def set_seed(seed=42):
    """
    设置PyTorch、NumPy和Python的随机种子，保证实验可复现。
    """
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False