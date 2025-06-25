import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from config import Config


class ImageDataset(Dataset):
    """Dataset class for image colorization"""
    
    def __init__(self, paths, train=True):
        self.paths = paths
        self.train = train
        
        if train:
            self.transforms = transforms.Compose([
                transforms.Resize((Config.image_size_1, Config.image_size_2)),
                transforms.RandomHorizontalFlip()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((Config.image_size_1, Config.image_size_2))
            ])
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        
        # Convert to LAB color space
        lab = rgb2lab(img).astype("float32")
        lab = transforms.ToTensor()(lab)
        
        # Normalize L and ab channels
        L = lab[[0], ...] / 50 - 1  # L channel: [-1, 1]
        ab = lab[[1, 2], ...] / 128  # ab channels: [-1, 1]
        
        return {'L': L, 'ab': ab}


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count


def create_loss_meters():
    """Create loss meters for tracking training progress"""
    return {
        'disc_loss_gen': AverageMeter(),
        'disc_loss_real': AverageMeter(),
        'disc_loss': AverageMeter(),
        'loss_G_GAN': AverageMeter(),
        'loss_G_L1': AverageMeter(),
        'loss_G': AverageMeter()
    }


def update_losses(model, loss_meter_dict, count):
    """Update loss meters with current batch losses"""
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)


def lab_to_rgb(L, ab):
    """Convert LAB images to RGB"""
    L = (L + 1.) * 50
    ab = ab * 128
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)


def visualize_results(model, data, save=False, save_path=None):
    """Visualize colorization results"""
    model.generator.eval()
    with torch.no_grad():
        model.prepare_input(data)
        model.forward()
    
    fake_color = model.gen_output.detach()
    real_color = model.ab
    L = model.L
    
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    
    fig = plt.figure(figsize=(15, 8))
    for i in range(min(5, len(L))):
        # Grayscale input
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax.set_title("Input (L)")
        
        # Generated colorization
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax.set_title("Generated")
        
        # Ground truth
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
        ax.set_title("Ground Truth")
    
    plt.tight_layout()
    plt.show()
    
    if save and save_path:
        fig.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.close(fig)


def log_results(loss_meter_dict):
    """Log training results"""
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")


def save_model(model, epoch, save_path):
    """Save complete model"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'generator_state_dict': model.generator.state_dict(),
        'discriminator_state_dict': model.discriminator.state_dict(),
        'gen_optimizer_state_dict': model.gen_optim.state_dict(),
        'disc_optimizer_state_dict': model.disc_optim.state_dict(),
    }
    
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def save_generator_only(generator, save_path):
    """Save only the generator model"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(generator.state_dict(), save_path)
    print(f"Generator saved to {save_path}")


def load_model(model, model_path, device):
    """Load full model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.generator.load_state_dict(checkpoint['generator_state_dict'])
    model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    model.gen_optim.load_state_dict(checkpoint['gen_optimizer_state_dict'])
    model.disc_optim.load_state_dict(checkpoint['disc_optimizer_state_dict'])

    model.to(device)
    print(f"Model loaded from {model_path}")
    return model



def pretrain_generator(generator, train_loader, epochs, device, save_path):
    """Pretrain generator with L1 loss"""
    print("Starting generator pretraining...")
    
    optimizer = torch.optim.Adam(generator.parameters(), lr=Config.pretrain_lr)
    criterion = torch.nn.L1Loss()
    
    for epoch in range(epochs):
        loss_meter = AverageMeter()
        generator.train()
        
        for data in train_loader:
            L, ab = data['L'].to(device), data['ab'].to(device)
            
            optimizer.zero_grad()
            preds = generator(L)
            loss = criterion(preds, ab)
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), L.size(0))
        
        print(f"Pretrain Epoch {epoch + 1}/{epochs} - L1 Loss: {loss_meter.avg:.5f}")
    
    # Save pretrained generator
    save_generator_only(generator, save_path)
    print("Generator pretraining completed!")
    
    return generator