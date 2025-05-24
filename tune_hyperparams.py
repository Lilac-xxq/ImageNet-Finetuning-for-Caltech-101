import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import utils
import os
from torch.utils.tensorboard import SummaryWriter

# 设置随机种子
utils.set_seed(42)  

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = './101_ObjectCategories'
num_classes = 101

def run_experiments(param_grid):
    best_val_acc = 0
    best_params = None
    best_model_state = None
    keys, values = zip(*param_grid.items())
    
    # Base directory for Tensorboard logs
    base_log_dir = 'runs/tune_params'
    os.makedirs(base_log_dir, exist_ok=True)

    for param_values in itertools.product(*values):
        params = dict(zip(keys, param_values))
        print(f"\nRunning experiment with params: {params}")
        
        # Create a unique log directory for each experiment and a writer
        param_str = '_'.join([f'{k}-{v}' for k,v in params.items()])
        log_dir = os.path.join(base_log_dir, param_str)
        writer = SummaryWriter(log_dir)

        train_loader, val_loader = utils.get_loaders(data_dir, params['batch_size'])
        model = utils.build_model(num_classes)
        optimizer = optim.SGD([
            {"params": model.fc.parameters(), "lr": params['lr_classifier']},
            {"params": [p for name, p in model.named_parameters() if "fc" not in name], "lr": params['lr_backbone']}
        ], momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        # 训练与验证，并获取历史数据
        val_acc, train_losses, train_accs, val_losses, val_accs = utils.train_and_validate(
            model, train_loader, val_loader, criterion, optimizer, device, params['num_epochs'])
        
        # 将历史数据写入Tensorboard
        for epoch in range(params['num_epochs']):
            writer.add_scalar('Loss/Train', train_losses[epoch], epoch)
            writer.add_scalar('Loss/Validation', val_losses[epoch], epoch)
            writer.add_scalar('Accuracy/Validation', val_accs[epoch], epoch)
            
        # Close the writer for the current experiment
        writer.close()

        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params.copy()
            best_model_state = model.state_dict() # Save the state_dict of the best model
            
    # 保存最佳模型
    if best_model_state is not None:
        torch.save(best_model_state, 'best_param_model.pth')
        print(f"\nBest Params: {best_params}, Best Val Acc: {best_val_acc:.2f}%")
    
    return best_params, best_val_acc

if __name__ == "__main__":
    # 超参数网格
    param_grid = {
        'batch_size': [16, 32],
        'num_epochs': [10],
        'lr_backbone': [1e-4, 5e-5],
        'lr_classifier': [1e-3, 5e-4]
    }
    best_params, best_val_acc = run_experiments(param_grid)
    print("Hyperparameter tuning finished.") 