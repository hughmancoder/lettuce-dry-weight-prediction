import random
from pathlib import Path
from tkinter import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torchvision.transforms.functional as TF


class LettuceNet(nn.Module):
    def __init__(self, pretrained=True):
        super(LettuceNet, self).__init__()
        # EfficientNet-B0 or ResNet18 are best for small datasets (N=232)
        self.backbone = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        
        # Modify first layer: 3 RGB + 1 Height Map = 4 Channels
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight[:, :3] = original_conv.weight
                self.backbone.conv1.weight[:, 3] = original_conv.weight.mean(dim=1)

        self.backbone.fc = nn.Identity()
        
        # Multi-task heads: Using FreshWeight and Height as auxiliary signals
        self.head_dry = nn.Linear(512, 1)
        self.head_fresh = nn.Linear(512, 1)
        self.head_plant_height = nn.Linear(512, 1)

    def forward(self, x):
        features = self.backbone(x)
        return {
            "dry": self.head_dry(features).squeeze(-1),
            "fresh": self.head_fresh(features).squeeze(-1),
            "height": self.head_plant_height(features).squeeze(-1)
        }


class LettuceDataset(Dataset):
    def __init__(self, df, processed_root, is_training=True):
        self.df = df
        self.root = Path(processed_root)
        self.is_training = is_training

    def __len__(self):
        return len(self.df)

    def transform(self, rgb, height_map):
        # Synchronised geometric augmentations
        if self.is_training:
            # Random Horizontal/Vertical Flips
            if random.random() > 0.5:
                rgb = TF.hflip(rgb)
                height_map = TF.hflip(height_map)
            
            if random.random() > 0.5:
                rgb = TF.vflip(rgb)
                height_map = TF.vflip(height_map)

            # Random Rotation (0-360)
            angle = random.uniform(0, 360)
            rgb = TF.rotate(rgb, angle)
            height_map = TF.rotate(height_map, angle)

            # Colour Jitter (RGB only!)
            color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)
            rgb = color_jitter(rgb)

        # Convert to Tensors
        rgb_tensor = TF.to_tensor(rgb)
        height_tensor = TF.to_tensor(height_map) # Converts [0, 1] numpy to tensor

        # Normalise RGB (ImageNet stats)
        rgb_tensor = TF.normalize(rgb_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        return torch.cat([rgb_tensor, height_tensor], dim=0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = int(row['image_id'])
        
        # Paths match your preprocessing script output
        rgb = Image.open(self.root / "RGB" / f"{img_id}.png").convert("RGB")
        h_map_raw = np.load(self.root / "Depth" / f"{img_id}.npy")
        
        # Convert height map to PIL for easy synchronised transforms
        h_map_pil = Image.fromarray(h_map_raw)
        
        image_tensor = self.transform(rgb, h_map_pil)
            
        labels = {
            "dry": torch.tensor(row['DryWeightShoot'], dtype=torch.float32),
            "fresh": torch.tensor(row.get('FreshWeightShoot', 0.0), dtype=torch.float32),
            "height": torch.tensor(row.get('Height', 0.0), dtype=torch.float32)
        }
        return image_tensor, labels
