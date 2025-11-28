import torch
import torch.nn as nn
from torchvision import models
DROPOUT = 0.3

class RGBDResNet(nn.Module):
    def __init__(self):
        super(RGBDResNet, self).__init__()
        
        # Load pre-trained ResNet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 4 channel input (R,G,B,Depth)
        original_weights = self.model.conv1.weight.data.clone()
        
        # Create new conv layer with 4 input channels
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():
            
            self.model.conv1.weight[:, :3, :, :] = original_weights

            # Initialize the Depth channel weight as the average of RGB over random
            self.model.conv1.weight[:, 3, :, :] = torch.mean(original_weights, dim=1)

        # modified output layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, 1) # Dry Weight Shoot 
        )

    def forward(self, x):
        return self.model(x)