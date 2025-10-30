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