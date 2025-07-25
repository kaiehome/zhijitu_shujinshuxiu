import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import os
import json
import logging
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from pathlib import Path

class LoomRecognitionDataset(Dataset):
    """
    织机识别图数据集
    用于训练AI模型生成专业识别图
    """
    def __init__(self, 
                 source_dir: str = "uploads",
                 target_dir: str = "target_images", 
                 transform=None,
                 target_transform=None):
        # 从backend目录回到项目根目录
        self.source_dir = Path("../" + source_dir)
        self.target_dir = Path("../" + target_dir)
        self.transform = transform
        self.target_transform = target_transform
        
        # 获取所有源图片
        self.source_files = list(self.source_dir.glob("*.jpg")) + list(self.source_dir.glob("*.png"))
        self.target_files = list(self.target_dir.glob("*.jpg")) + list(self.target_dir.glob("*.png"))
        
        # 确保源图片和目标图片数量匹配
        self.source_files = self.source_files[:len(self.target_files)]
        self.target_files = self.target_files[:len(self.source_files)]
        
        print(f"数据集大小: {len(self.source_files)} 对图片")
        
    def __len__(self):
        return len(self.source_files)
    
    def __getitem__(self, idx):
        # 加载源图片（原始图片）
        source_path = self.source_files[idx]
        source_img = Image.open(source_path).convert('RGB')
        
        # 加载目标图片（专业识别图）
        target_path = self.target_files[idx]
        target_img = Image.open(target_path).convert('RGB')
        
        if self.transform:
            source_img = self.transform(source_img)
        if self.target_transform:
            target_img = self.target_transform(target_img)
            
        return source_img, target_img

class LoomRecognitionGenerator(nn.Module):
    """
    织机识别图生成器
    基于U-Net架构的深度学习模型
    """
    def __init__(self, input_channels=3, output_channels=3, features=64):
        super(LoomRecognitionGenerator, self).__init__()
        
        # 编码器
        self.encoder1 = self._make_layer(input_channels, features, kernel_size=4, stride=2, padding=1)
        self.encoder2 = self._make_layer(features, features * 2, kernel_size=4, stride=2, padding=1)
        self.encoder3 = self._make_layer(features * 2, features * 4, kernel_size=4, stride=2, padding=1)
        self.encoder4 = self._make_layer(features * 4, features * 8, kernel_size=4, stride=2, padding=1)
        
        # 瓶颈层
        self.bottleneck = self._make_layer(features * 8, features * 16, kernel_size=4, stride=2, padding=1)
        
        # 解码器
        self.decoder4 = self._make_layer(features * 16, features * 8, kernel_size=4, stride=2, padding=1, transpose=True)
        self.decoder3 = self._make_layer(features * 8, features * 4, kernel_size=4, stride=2, padding=1, transpose=True)
        self.decoder2 = self._make_layer(features * 4, features * 2, kernel_size=4, stride=2, padding=1, transpose=True)
        self.decoder1 = self._make_layer(features * 2, features, kernel_size=4, stride=2, padding=1, transpose=True)
        
        # 输出层
        self.output = nn.ConvTranspose2d(features, output_channels, kernel_size=4, stride=2, padding=1)
        
        # 激活函数
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def _make_layer(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False):
        if transpose:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
    
    def forward(self, x):
        # 编码
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        
        # 瓶颈
        bottleneck = self.bottleneck(enc4)
        
        # 解码（带跳跃连接）
        dec4 = self.decoder4(bottleneck)
        dec3 = self.decoder3(dec4)
        dec2 = self.decoder2(dec3)
        dec1 = self.decoder1(dec2)
        
        # 输出
        output = self.output(dec1)
        return self.tanh(output)

class LoomRecognitionDiscriminator(nn.Module):
    """
    织机识别图判别器
    用于对抗训练
    """
    def __init__(self, input_channels=3, features=64):
        super(LoomRecognitionDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # 第一层
            nn.Conv2d(input_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            # 第二层
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2),
            
            # 第三层
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2),
            
            # 第四层
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2),
            
            # 输出层
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class LoomRecognitionTrainer:
    """
    织机识别图AI模型训练器
    """
    def __init__(self, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 learning_rate: float = 0.0002,
                 beta1: float = 0.5,
                 beta2: float = 0.999):
        
        self.device = device
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        
        # 初始化模型
        self.generator = LoomRecognitionGenerator().to(device)
        self.discriminator = LoomRecognitionDiscriminator().to(device)
        
        # 初始化优化器
        self.g_optimizer = optim.Adam(self.generator.parameters(), 
                                     lr=learning_rate, 
                                     betas=(beta1, beta2))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), 
                                     lr=learning_rate, 
                                     betas=(beta1, beta2))
        
        # 损失函数
        self.criterion = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        print(f"使用设备: {device}")
        print(f"生成器参数数量: {sum(p.numel() for p in self.generator.parameters())}")
        print(f"判别器参数数量: {sum(p.numel() for p in self.discriminator.parameters())}")
    
    def prepare_data(self, 
                    source_dir: str = "uploads",
                    target_dir: str = "target_images",
                    batch_size: int = 4,
                    image_size: int = 256):
        """
        准备训练数据
        """
        # 数据变换
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # 创建数据集
        dataset = LoomRecognitionDataset(
            source_dir=source_dir,
            target_dir=target_dir,
            transform=transform,
            target_transform=transform
        )
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        return dataloader
    
    def train_step(self, real_source, real_target):
        """
        单步训练
        """
        batch_size = real_source.size(0)
        
        # 获取判别器输出尺寸
        with torch.no_grad():
            sample_output = self.discriminator(real_target)
            output_size = sample_output.size()
        
        # 根据判别器输出尺寸创建标签
        real_label = torch.ones(output_size).to(self.device)
        fake_label = torch.zeros(output_size).to(self.device)
        
        # 训练判别器
        self.d_optimizer.zero_grad()
        
        # 真实图片
        real_output = self.discriminator(real_target)
        d_real_loss = self.criterion(real_output, real_label)
        
        # 生成图片
        fake_target = self.generator(real_source)
        fake_output = self.discriminator(fake_target.detach())
        d_fake_loss = self.criterion(fake_output, fake_label)
        
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()
        
        # 训练生成器
        self.g_optimizer.zero_grad()
        
        # 对抗损失
        fake_output = self.discriminator(fake_target)
        g_adv_loss = self.criterion(fake_output, real_label)
        
        # 内容损失（L1损失）
        g_content_loss = self.l1_loss(fake_target, real_target)
        
        # 总损失
        g_loss = g_adv_loss + 100 * g_content_loss  # 内容损失权重更高
        g_loss.backward()
        self.g_optimizer.step()
        
        return g_loss.item(), d_loss.item()
    
    def train(self, 
              dataloader,
              epochs: int = 100,
              save_interval: int = 10,
              log_interval: int = 10,
              model_save_dir: str = "trained_models"):
        """
        训练模型
        """
        # 禁用W&B，使用本地日志
        use_wandb = False
        # try:
        #     import wandb
        #     use_wandb = True
        #     wandb.init(project="loom-recognition-gan", name="training_run")
        # except:
        #     print("W&B未配置，使用本地日志记录")
        #     use_wandb = False
        
        print(f"开始训练，共{epochs}个epoch")
        print(f"使用设备: {self.device}")
        print(f"禁用W&B，使用本地日志记录")
        
        # 创建保存目录
        os.makedirs(model_save_dir, exist_ok=True)
        
        for epoch in range(epochs):
            epoch_g_loss = 0
            epoch_d_loss = 0
            
            for batch_idx, (real_source, real_target) in enumerate(dataloader):
                real_source = real_source.to(self.device)
                real_target = real_target.to(self.device)
                
                # 训练一步
                g_loss, d_loss = self.train_step(real_source, real_target)
                
                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                
                # 记录日志
                if batch_idx % log_interval == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                          f"G_Loss: {g_loss:.4f} D_Loss: {d_loss:.4f}")
                    
                    # 本地日志记录
                    if use_wandb:
                        wandb.log({
                            "epoch": epoch + 1,
                            "batch": batch_idx,
                            "generator_loss": g_loss,
                            "discriminator_loss": d_loss
                        })
            
            # 计算平均损失
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader)
            
            print(f"Epoch [{epoch+1}/{epochs}] 完成 - "
                  f"平均G_Loss: {avg_g_loss:.4f} 平均D_Loss: {avg_d_loss:.4f}")
            
            # 保存模型
            if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
                self.save_models(model_save_dir, f"epoch_{epoch+1}")
                
                # 生成示例图片
                if len(real_source) > 0:
                    self.generate_sample_images(real_source[:4], real_target[:4], model_save_dir, f"epoch_{epoch+1}")
            
            # 本地日志记录
            if use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "avg_generator_loss": avg_g_loss,
                    "avg_discriminator_loss": avg_d_loss
                })
        
        # 保存最终模型
        self.save_models(model_save_dir, "final")
        print("训练完成！")
        
        # 关闭W&B
        if use_wandb:
            wandb.finish()
    
    def save_models(self, save_dir: str, epoch: str):
        """
        保存模型
        """
        generator_path = os.path.join(save_dir, f"generator_epoch_{epoch}.pth")
        discriminator_path = os.path.join(save_dir, f"discriminator_epoch_{epoch}.pth")
        
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)
        
        print(f"模型已保存到 {save_dir}")
    
    def load_models(self, model_dir: str, epoch: str = "final"):
        """
        加载模型
        """
        generator_path = os.path.join(model_dir, f"generator_epoch_{epoch}.pth")
        discriminator_path = os.path.join(model_dir, f"discriminator_epoch_{epoch}.pth")
        
        if os.path.exists(generator_path):
            self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
            print(f"生成器模型已加载: {generator_path}")
        
        if os.path.exists(discriminator_path):
            self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=self.device))
            print(f"判别器模型已加载: {discriminator_path}")
    
    def generate_sample_images(self, real_source, real_target, save_dir: str, epoch: str):
        """
        生成示例图片
        """
        self.generator.eval()
        with torch.no_grad():
            fake_target = self.generator(real_source)
            
            # 转换回图片格式
            def tensor_to_image(tensor):
                tensor = (tensor + 1) / 2  # 从[-1,1]转换到[0,1]
                tensor = tensor.clamp(0, 1)
                return tensor.cpu().permute(0, 2, 3, 1).numpy()
            
            real_source_img = tensor_to_image(real_source)
            real_target_img = tensor_to_image(real_target)
            fake_target_img = tensor_to_image(fake_target)
            
            # 获取批次大小
            batch_size = real_source.size(0)
            num_samples = min(batch_size, 4)  # 最多显示4个样本
            
            # 保存图片
            if num_samples == 1:
                fig, axes = plt.subplots(3, 1, figsize=(4, 12))
                axes = axes.reshape(-1, 1)
            else:
                fig, axes = plt.subplots(3, num_samples, figsize=(4*num_samples, 12))
            
            for i in range(num_samples):
                axes[0, i].imshow(real_source_img[i])
                axes[0, i].set_title(f"Source {i+1}")
                axes[0, i].axis('off')
                
                axes[1, i].imshow(real_target_img[i])
                axes[1, i].set_title(f"Target {i+1}")
                axes[1, i].axis('off')
                
                axes[2, i].imshow(fake_target_img[i])
                axes[2, i].set_title(f"Generated {i+1}")
                axes[2, i].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_epoch_{epoch}.png"))
            plt.close()
        
        self.generator.train()
    
    def generate_image(self, source_image_path: str, output_path: str = None):
        """
        使用训练好的模型生成图片
        """
        self.generator.eval()
        
        # 加载和预处理图片
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        source_img = Image.open(source_image_path).convert('RGB')
        source_tensor = transform(source_img).unsqueeze(0).to(self.device)
        
        # 生成图片
        with torch.no_grad():
            fake_target = self.generator(source_tensor)
            
            # 转换回图片格式
            fake_target = (fake_target + 1) / 2
            fake_target = fake_target.clamp(0, 1)
            fake_target = fake_target.cpu().squeeze(0).permute(1, 2, 0).numpy()
            
            # 转换为PIL图片
            fake_target = (fake_target * 255).astype(np.uint8)
            fake_target_img = Image.fromarray(fake_target)
            
            # 保存图片
            if output_path is None:
                output_path = source_image_path.replace('.', '_ai_generated.')
            
            fake_target_img.save(output_path)
            print(f"AI生成图片已保存: {output_path}")
            
            return output_path
        
        self.generator.train()

def create_target_images_dataset():
    """
    创建目标图片数据集
    将现有的专业识别图作为训练目标
    """
    # 从backend目录回到项目根目录
    target_dir = "../target_images"
    os.makedirs(target_dir, exist_ok=True)
    
    # 首先检查target_images目录中是否已有图片
    existing_targets = []
    if os.path.exists(target_dir):
        for file in os.listdir(target_dir):
            if file.endswith(('.jpg', '.png', '.jpeg')):
                existing_targets.append(os.path.join(target_dir, file))
    
    if existing_targets:
        print(f"找到 {len(existing_targets)} 个现有目标图片")
        return len(existing_targets)
    
    # 如果没有现有目标图片，尝试从其他目录收集
    source_dirs = [
        "auto_tune_ai_results",
        "auto_tune_improved_results", 
        "panda_force_limited_results",
        "panda_optimized_results",
        "panda_precise_results"
    ]
    
    target_count = 0
    for source_dir in source_dirs:
        if os.path.exists(source_dir):
            for file in os.listdir(source_dir):
                if file.endswith('.png') and 'color_table' in file:
                    source_path = os.path.join(source_dir, file)
                    target_path = os.path.join(target_dir, f"target_{target_count:04d}.png")
                    
                    # 复制文件
                    import shutil
                    shutil.copy2(source_path, target_path)
                    target_count += 1
    
    print(f"创建了 {target_count} 个目标图片")
    return target_count

if __name__ == "__main__":
    # 创建目标数据集
    target_count = create_target_images_dataset()
    
    if target_count == 0:
        print("错误：没有找到目标图片，请先生成一些专业识别图")
        exit(1)
    
    # 初始化训练器
    trainer = LoomRecognitionTrainer()
    
    # 准备数据
    dataloader = trainer.prepare_data(batch_size=4, image_size=256)
    
    # 开始训练
    trainer.train(dataloader, epochs=50, save_interval=5)
    
    print("AI模型训练完成！") 