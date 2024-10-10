import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from diffusion_model import DenoiseDiffusion
from unet import UNet

# 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# 超参数
batch_size = 128
n_epochs = 100
lr = 1e-4
n_steps = 1000

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 初始化模型
unet = UNet(image_channels=1, n_channels=64, ch_mults=[1, 2, 4, 8], is_attn=[False, False, True, True]).to(device)
diffusion = DenoiseDiffusion(unet, n_steps=n_steps, device=device)

# 优化器
optimizer = Adam(unet.parameters(), lr=lr)

# 训练循环
for epoch in range(n_epochs):
    pbar = tqdm(train_loader)
    for i, (images, _) in enumerate(pbar):
        images = images.to(device)
        
        optimizer.zero_grad()
        loss = diffusion.loss(images)
        loss.backward()
        optimizer.step()
        
        pbar.set_description(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")
    
    # 每10个epoch保存一次模型
    if (epoch + 1) % 10 == 0:
        torch.save(unet.state_dict(), f'unet_epoch_{epoch+1}.pth')
        torch.save(diffusion.state_dict(), f'diffusion_epoch_{epoch+1}.pth')

print("Training completed!")

# 生成样本
@torch.no_grad()
def generate_samples(num_samples=16):
    samples = torch.randn(num_samples, 1, 28, 28).to(device)
    for i in tqdm(reversed(range(n_steps)), desc='Sampling'):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        samples = diffusion.p_sample(samples, t)
    
    samples = (samples + 1) / 2  # 将[-1, 1]范围转换为[0, 1]
    return samples.cpu()

# 生成并保存一些样本
samples = generate_samples()
import torchvision.utils as vutils
vutils.save_image(samples, 'generated_samples.png', nrow=4)
