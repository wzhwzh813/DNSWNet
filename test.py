import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from model import DNSWNet
from Lossfunction import SSINLoss
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


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

class SSIM:
    def __init__(self, window_size=11, size_average=True):
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.create_window(window_size, self.channel)

    def gaussian_window(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        self.window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    
    def ssim(self, img1, img2):
        (_, channel, _, _) = img1.size()
        self.create_window(self.window_size, channel)
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def __call__(self, img1, img2):
        return self.ssim(img1, img2)


# 定义超参数
batch_size = 16
learning_rate = 1e-3
num_epochs = 1
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
test_dataset = MultiModalDataset(clean_dir, noisy_dir, transform=transform)

# 创建 DataLoader
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 模型、损失函数和优化器
model = DNSWNet(num_layers=22).to(device)


model = nn.DataParallel(model)  # 使用 DataParallel 并行化模型
criterion = SSIMLoss().to(device)
optimizer = optim.RAdam(model.parameters(), lr=learning_rate)
# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)


best_model_path ='DNSWNet/DNSWNet_allweather/DNSWNet_AllWeather_MSE_model.pth'

if hasattr(model, 'module'):
    model.module.load_state_dict(torch.load(best_model_path))
else:
    model.load_state_dict(torch.load(best_model_path))

# 加载之前保存的模型参数（如果需要）
if continue_train and os.path.exists(best_model_path):
    print(f"Loading model parameters from {best_model_path}")
    model.module.load_state_dict(torch.load(best_model_path))


# 初始化SSIM计算器
ssim_calculator = SSIM(window_size=11)

# 测试集上的去噪并保存图像
model.eval()
psnr_total = 0
ssim_total = 0
num_batches = 0

# 初始化最小和最大PSNR、SSIM值
min_psnr = float('inf')
max_psnr = float('-inf')
min_ssim = float('inf')
max_ssim = float('-inf')

# 去噪示例并保存图像
with torch.no_grad():
    for noisy_imgs, clean_imgs in tqdm(test_loader, desc="Processing Batches"):
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)

        # 进行去噪
        outputs = model(noisy_imgs)

        # 将结果移回CPU
        outputs = outputs.cpu()
        noisy_imgs = noisy_imgs.cpu()
        clean_imgs = clean_imgs.cpu()

        # 计算PSNR和SSIM值
        psnr_values = [psnr(outputs[i], clean_imgs[i]) for i in range(len(outputs))]
        ssim_values = [ssim_calculator(outputs[i].unsqueeze(0), clean_imgs[i].unsqueeze(0)).item() for i in range(len(outputs))]

        # 更新总的PSNR和SSIM值
        for i in range(outputs.size(0)):
            psnr_val = psnr(outputs[i], clean_imgs[i])
            ssim_val = ssim_calculator(outputs[i].unsqueeze(0), clean_imgs[i].unsqueeze(0)).item()
            psnr_total += psnr_val
            ssim_total += ssim_val

            # 更新最小和最大PSNR、SSIM值
            min_psnr = min(min_psnr, psnr_val)
            max_psnr = max(max_psnr, psnr_val)
            min_ssim = min(min_ssim, ssim_val)
            max_ssim = max(max_ssim, ssim_val)

        num_batches += 1

# 计算平均PSNR和SSIM
avg_psnr = psnr_total / (num_batches * batch_size)
avg_ssim = ssim_total / (num_batches * batch_size)

print(f'Average PSNR: {avg_psnr:.4f} dB')
print(f'Average SSIM: {avg_ssim:.4f}')
print(f'Minimum PSNR: {min_psnr:.4f} dB')
print(f'Maximum PSNR: {max_psnr:.4f} dB')
print(f'Minimum SSIM: {min_ssim:.4f}')
print(f'Maximum SSIM: {max_ssim:.4f}')

with torch.no_grad():
    dataset = MultiModalDataset(clean_dir, noisy_dir, transform=transform)
    noisy_imgs, clean_imgs = [], []
    indices = [0, 100, 200, 300, 400, 500] 
    for i in indices:
        noisy_img, clean_img = dataset[i]
        noisy_imgs.append(noisy_img)
        clean_imgs.append(clean_img)
    noisy_imgs = torch.stack(noisy_imgs).to(device)
    clean_imgs = torch.stack(clean_imgs).to(device)

    outputs = model(noisy_imgs)
    outputs = outputs.cpu()
    noisy_imgs = noisy_imgs.cpu()
    clean_imgs = clean_imgs.cpu()

    # 计算PSNR值
    psnr_values = [psnr(outputs[i], clean_imgs[i]) for i in range(6)]
    psnr_values_noisy = [psnr(noisy_imgs[i], clean_imgs[i]) for i in range(6)]

    # 初始化SSIM计算器
    ssim_calculator = SSIM(window_size=11)

    # 计算SSIM值
    ssim_values = [ssim_calculator(outputs[i].unsqueeze(0), clean_imgs[i].unsqueeze(0)).item() for i in range(6)]
    ssim_values_noisy = [ssim_calculator(noisy_imgs[i].unsqueeze(0), clean_imgs[i].unsqueeze(0)).item() for i in range(6)]

    fig, axs = plt.subplots(3, 6, figsize=(18, 9))  # 设置3行6列的子图布局
    for i in range(6):
        axs[0, i].imshow(transforms.ToPILImage()(noisy_imgs[i]))
        axs[0, i].set_title(f'Noisy\nPSNR: {psnr_values_noisy[i]:.2f} dB')
        axs[0, i].axis('off')
        axs[1, i].imshow(transforms.ToPILImage()(outputs[i]))
        axs[1, i].set_title(f'Denoised\nPSNR: {psnr_values[i]:.2f} dB,SSIM: {ssim_values[i]:.4f}')
        axs[1, i].axis('off')
        axs[2, i].imshow(transforms.ToPILImage()(clean_imgs[i]))
        axs[2, i].set_title('Clean')
        axs[2, i].axis('off')

# 保存并关闭图像
plt.tight_layout()
plt.savefig('DNSWNet/DNSWNet_allweather/MSE_test.png')
plt.close()
