import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import ResNet18_Weights
import utils
from torch.utils.tensorboard import SummaryWriter
import os

# 设置随机种子
utils.set_seed(42)
print("Start training (from scratch)...")

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = './101_ObjectCategories'
batch_size = 32
num_classes = 101
num_epochs = 30
lr = 1e-4 # 统一学习率
# lr_backbone = 1e-4 # 原主干网络学习率
# lr_classifier = 1e-3 # 原输出层学习率

# Tensorboard writer
log_dir = 'runs/scratch_training'
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# 加载数据集
train_loader, val_loader = utils.get_loaders(data_dir, batch_size)

# 随机初始化ResNet18模型（不加载预训练权重）
model = utils.build_model(num_classes, weights=None)
model.to(device)

# 设置优化器（使用统一学习率）
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 训练与验证，并获取历史数据
best_val_acc, train_losses, train_accs, val_losses, val_accs = utils.train_and_validate(
    model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

print("Training finished.")
print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

# 将历史数据写入Tensorboard
for epoch in range(num_epochs):
    # 使用 add_scalars 将训练和验证 Loss 绘制在同一张图上（本 Run 内对比）
    writer.add_scalars('Loss_scratch', {'Train': train_losses[epoch], 'Validation': val_losses[epoch]}, epoch)
    # 验证准确率单独绘制
    writer.add_scalar('Accuracy/Validation', val_accs[epoch], epoch)

    # 使用 add_scalar 分别记录 Loss/Train 和 Loss/Validation（跨 Run 对比用）
    writer.add_scalar('Loss/Train', train_losses[epoch], epoch)
    writer.add_scalar('Loss/Validation', val_losses[epoch], epoch)

# 关闭writer
writer.close()

# 保存模型
torch.save(model.state_dict(), "resnet18_caltech101_scratch.pth") 