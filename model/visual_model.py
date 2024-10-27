# models/visual_model.py

import torch.nn as nn
from torchvision import models

class VisualModel(nn.Module):
    def __init__(self, pretrained=True, dropout=0.3):
        super(VisualModel, self).__init__()
        self.base_model = models.resnet18(pretrained=pretrained)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()  # Remove the classification layer
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(num_ftrs, 128)  # Output size for fusion
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        return self.out(x)
