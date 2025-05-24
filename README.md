# ImageNet-Finetuning-for-Caltech-101

## 项目结构
```
├── ImageNet.py # 主模型（预训练微调）训练脚本
├── train_from_scratch.py # 从零开始训练模型脚本
├── tune_hyperparams.py # 超参数调优脚本
├── utils.py # 工具函数集合（数据加载、模型构建、训练验证等）
├── runs/ # Tensorboard 日志目录（运行训练脚本后生成）
└── 训练过程可视化/ # 实验结果（Tensorboard截图）
```

## 训练和测试

本项目中的“测试”功能集成在训练过程的验证阶段 (`utils.train_and_validate` 函数中的验证循环)。运行训练脚本即可同时进行验证。

1.  **训练主模型 (预训练微调):**
    运行 `ImageNet.py` 脚本进行基于 ImageNet 预训练模型的微调训练。
    ```bash
    python ImageNet.py
    ```
    训练完成后，模型权重将保存为 `resnet18_caltech101.pth`。

2.  **从零开始训练模型:**
    运行 `train_from_scratch.py` 脚本进行从零开始训练 ResNet-18 模型。
    ```bash
    python train_from_scratch.py
    ```
    训练完成后，模型权重将保存为 `resnet18_caltech101_scratch.pth`。

3.  **超参数调优:**
    运行 `tune_hyperparams.py` 脚本进行超参数调优实验。脚本会尝试预设的超参数组合，并保存表现最佳的模型权重。
    ```bash
    python tune_hyperparams.py
    ```
    调优完成后，最佳模型权重将保存为 `best_param_model.pth`。

**注意：** 运行训练脚本会在 `runs/` 目录下生成 Tensorboard 日志。重复运行同一脚本会向该脚本对应的日志目录追加数据，或者如果删除了旧的日志目录则会创建新的日志目录。为了避免日志混乱，建议在重新运行特定实验前删除对应的 `runs/` 子目录（例如，重新运行 `ImageNet.py` 前删除 `runs/ImageNet_main/`）。

## Tensorboard 可视化

训练过程中生成的日志可以使用 Tensorboard 进行可视化，观察训练和验证的损失、准确率等曲线。

在项目根目录运行以下命令启动 Tensorboard 服务：
```bash
tensorboard --logdir runs
```
然后打开浏览器访问终端输出中提供的地址（通常是 `http://localhost:6006/`）。


## 模型权重下载

训练好的模型权重文件已上传至网盘，可通过以下链接下载：

[下载链接: https://pan.baidu.com/s/1sBHG72BnWj-6NFPBIx7djQ?pwd=qyc9  提取码: qyc9]

下载的文件包括：
*   `resnet18_caltech101.pth`: 主模型（预训练微调）训练权重
*   `resnet18_caltech101_scratch.pth`: 从零开始训练模型权重
*   `best_param_model.pth`: 超参数调优后的最佳模型权重

