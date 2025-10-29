import torch
import torch.nn as nn
import torchvision.models as models


class SpatialStream(nn.Module):
    def __init__(self, n_class=10):
        super().__init__()
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Use all layers except the fc
        self.backbone = nn.Sequential(*list(model.children())[:-1])

        # Custom classification head
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),       
            nn.Linear(512, n_class)
        )

    def forward(self, x):
        x = self.backbone(x)          
        x = torch.flatten(x, 1)      
        x = self.fc(x)                
        return x


class TemporalStream(nn.Module):
    def __init__(self, n_class=10, T=9):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify the first layer to accept 2*T input channels (u, v)
        base.conv1 = nn.Conv2d(
            2*T, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Remove the last fully connected layer
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        # Custom classification head
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, n_class)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        # Merge flow frames into channels as [B, 2*T, H, W]
        x = x.view(B, C*T, H, W)
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
