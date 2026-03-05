import torch
import torch.nn as nn
from torchvision import models


class VisionModel(nn.Module):
    """
    Binary image classifier based on a pretrained ResNet18 backbone.
    """

    def __init__(self, num_classes=2):

        super().__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(weights="IMAGENET1K_V1")

        # Freeze all convolutional layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Get number of input features for the classifier
        in_features = self.model.fc.in_features

        # Replace the final fully connected layer
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)