#!/usr/bin/env python3
"""
简化的AI大模型训练脚本
用于训练织机识别图生成模型
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('simple_ai_training.log'),
            logging.StreamHandler()
        ]
    )

class SimpleDataset(Dataset):
    """简化的数据集"""
    def __init__(self, source_dir="../uploads", target_dir="../target_images"):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        
        # 获取所有图片文件
        self.source_files = list(self.source_dir.glob("*.jpg")) + list(self.source_dir.glob("*.png"))
        self.target_files = list(self.target_dir.glob("*.jpg")) + list(self.target_dir.glob("*.png"))
        
        # 确保数量匹配
        min_count = min(len(self.source_files), len(self.target_files))
        self.source_files = self.source_files[:min_count]
        self.target_files = self.target_files[:min_count]
        
        print(f"数据集大小: {len(self.source_files)} 对图片")
        
        # 转换
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.source_files)
    
    def __getitem__(self, idx):
        source_path = self.source_files[idx]
        target_path = self.target_files[idx]
        
        source_img = Image.open(source_path).convert('RGB')
        target_img = Image.open(target_path).convert('RGB')
        
        source_tensor = self.transform(source_img)
        target_tensor = self.transform(target_img)
        
        return source_tensor, target_tensor

class SimpleGenerator(nn.Module):
    """简化的生成器"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class SimpleDiscriminator(nn.Module):
    """简化的判别器"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def train_simple_gan():
    """训练简化的GAN"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据集
    dataset = SimpleDataset()
    if len(dataset) == 0:
        print("错误: 数据集为空！")
        return
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 创建模型
    generator = SimpleGenerator().to(device)
    discriminator = SimpleDiscriminator().to(device)
    
    # 优化器
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # 损失函数
    criterion = nn.BCELoss()
    l1_loss = nn.L1Loss()
    
    # 训练
    epochs = 2
    print(f"开始训练，共{epochs}个epoch")
    
    for epoch in range(epochs):
        for batch_idx, (real_source, real_target) in enumerate(dataloader):
            real_source = real_source.to(device)
            real_target = real_target.to(device)
            
            batch_size = real_source.size(0)
            
            # 获取判别器输出尺寸
            with torch.no_grad():
                sample_output = discriminator(real_target)
                output_size = sample_output.size()
            
            # 根据判别器输出尺寸创建标签
            real_label = torch.ones(output_size).to(device)
            fake_label = torch.zeros(output_size).to(device)
            
            # 训练判别器
            d_optimizer.zero_grad()
            
            # 真实图片
            real_output = discriminator(real_target)
            d_real_loss = criterion(real_output, real_label)
            
            # 生成图片
            fake_target = generator(real_source)
            fake_output = discriminator(fake_target.detach())
            d_fake_loss = criterion(fake_output, fake_label)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # 训练生成器
            g_optimizer.zero_grad()
            
            fake_output = discriminator(fake_target)
            g_adv_loss = criterion(fake_output, real_label)
            g_content_loss = l1_loss(fake_target, real_target)
            
            g_loss = g_adv_loss + 10 * g_content_loss
            g_loss.backward()
            g_optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"G_Loss: {g_loss.item():.4f} D_Loss: {d_loss.item():.4f}")
    
    # 保存模型
    os.makedirs("../trained_models", exist_ok=True)
    torch.save(generator.state_dict(), "../trained_models/generator_simple.pth")
    torch.save(discriminator.state_dict(), "../trained_models/discriminator_simple.pth")
    print("模型保存完成！")
    
    return generator, discriminator

if __name__ == "__main__":
    setup_logging()
    train_simple_gan() 