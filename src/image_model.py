import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ImagePhishingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)
