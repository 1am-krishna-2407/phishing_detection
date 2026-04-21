import torch.nn as nn
from torchvision.models import resnet18


class ImagePhishingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.model(x).squeeze(1)
