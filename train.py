import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from glob import glob
from PIL import Image
from tqdm import tqdm
import time
from Lossfunction import SSIMLoss
from model import DNSWNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# 数据集定义
class MultiModalDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, transform=None):
        self.clean_paths = sorted(glob(os.path.join(clean_dir, '*.png')))
        self.noisy_paths = sorted(glob(os.path.join(noisy_dir, '*.png')))
        self.transform = transform

        if len(self.clean_paths) == 0 or len(self.noisy_paths) == 0:
            raise ValueError(f"clean_images or noisy_images directory is empty")

        if len(self.clean_paths) != len(self.noisy_paths):
            raise ValueError(f"The number of clean images and noisy images must be the same")

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_img = Image.open(self.clean_paths[idx]).convert('RGB')
        noisy_img = Image.open(self.noisy_paths[idx]).convert('RGB')
        if self.transform:
            clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)
        return noisy_img, clean_img



# 定义超参数
batch_size = 16
learning_rate = 1e-3
num_epochs = 200
continue_train = True  # 是否加载已有模型继续训练
start_epoch = 0

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),
])

# 数据加载
clean_dir = 'data/allweather/gt'
noisy_dir = 'data/allweather/input'

# 创建数据集
train_dataset = MultiModalDataset(clean_dir, noisy_dir, transform=transform)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 模型、损失函数和优化器
model = DNSWNet(num_layers=22).to(device)

from thop import profile

# 创建一个随机的输入张量用于FLOPs计算
dummy_input = torch.randn(1, 3, 224, 224).to(device)

# 使用 thop 计算 FLOPs 和参数数量
flops, params = profile(model, inputs=(dummy_input, ))

print(f"Total number of parameters: {params}")
print(f"Total number of FLOPs: {flops}")


# **删除 FLOPs 和参数计数器属性**
for module in model.modules():
    if hasattr(module, 'total_ops'):
        del module.total_ops
    if hasattr(module, 'total_params'):
        del module.total_params

model = nn.DataParallel(model)  # 使用 DataParallel 并行化模型
criterion = SSIMLoss().to(device)
optimizer = optim.RAdam(model.parameters(), lr=learning_rate)
# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

# 记录损失值
losses = []
best_loss = float('inf')
best_model_path = 'DNSWNet_AllWeather_model.pth'

# 计算模型中所有参数的总数量
total_params = sum(param.numel() for param in model.parameters())
print(f"Total number of parameters: {total_params}")


# 加载之前保存的模型参数（如果需要）
if continue_train and os.path.exists(best_model_path):
    print(f"Loading model parameters from {best_model_path}")
    model.module.load_state_dict(torch.load(best_model_path))

# 训练过程
for epoch in range(start_epoch, num_epochs):
    model.train()
    epoch_loss = 0
    start_time = time.time()  # 记录开始时间

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for noisy_imgs, clean_imgs in train_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            loss = criterion(outputs, clean_imgs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': epoch_loss / (pbar.n + 1)})
            pbar.update(1)

    end_time = time.time()  # 记录结束时间
    epoch_duration = end_time - start_time  # 计算每轮运行时间
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)

    # 保存最好的模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.module.state_dict(), best_model_path)  # 保存模型时需使用 model.module
        print(f'Best model saved with loss: {best_loss:.4f}')

    scheduler.step(avg_loss)  # 更新学习率调度器
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f} seconds')


