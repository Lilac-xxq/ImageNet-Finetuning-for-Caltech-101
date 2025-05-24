import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet18_Weights
import utils
from torch.utils.tensorboard import SummaryWriter
import os

# 设置随机种子
utils.set_seed(42)  
print("Start training...")

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = './101_ObjectCategories'
batch_size = 32
num_classes = 101
num_epochs = 10
lr_backbone = 1e-4
lr_classifier = 1e-3

# Tensorboard writer
log_dir = 'runs/ImageNet_main' # Log directory for main training
os.makedirs(log_dir, exist_ok=True) # Create directory if it doesn't exist
writer = SummaryWriter(log_dir)

# 加载数据集
train_loader, val_loader = utils.get_loaders(data_dir, batch_size)

# 加载预训练模型并替换输出层
model = utils.build_model(num_classes)
model.to(device)

# 设置优化器（区分主干网络和输出层的学习率）
optimizer = optim.SGD([
    {"params": model.fc.parameters(), "lr": lr_classifier},
    {"params": [p for name, p in model.named_parameters() if "fc" not in name], "lr": lr_backbone}
], momentum=0.9)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 训练与验证，并获取历史数据
best_val_acc, train_losses, train_accs, val_losses, val_accs = utils.train_and_validate(
    model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

print("Training finished.")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# 将历史数据写入Tensorboard
for epoch in range(num_epochs):
    # 训练和验证 Loss（同 Run 对比用）
    writer.add_scalars('Loss_ImageNet', {'Train': train_losses[epoch], 'Validation': val_losses[epoch]}, epoch)
    # 验证准确率单独绘制
    writer.add_scalar('Accuracy/Validation', val_accs[epoch], epoch)
    
    # 分别记录 Loss/Train 和 Loss/Validation（跨 Run 对比用）
    writer.add_scalar('Loss/Train', train_losses[epoch], epoch)
    writer.add_scalar('Loss/Validation', val_losses[epoch], epoch)

# 关闭writer
writer.close()

# 保存模型
torch.save(model.state_dict(), "resnet18_caltech101.pth")