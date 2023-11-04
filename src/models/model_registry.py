#@ ishapira 20231103
"""
serves as a registry of model architectures. It provides
factory functions that return instances of various models, such as AlexNet, ResNet, and
LeNet, which all inherit from the base class defined in base_model.py.
"""

import torch.nn as nn
import torchvision.models as models
from .base_model import BaseModel

# LeNet architecture
class LeNet(BaseModel):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# AlexNet architecture
class AlexNet(BaseModel):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = models.alexnet(pretrained=False).features
        self.avgpool = models.alexnet(pretrained=False).avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ResNet architecture
class ResNet(BaseModel):
    def __init__(self, num_classes=1000):
        super(ResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def create_lenet(num_classes=10):
    return LeNet(num_classes=num_classes)

def create_alexnet(num_classes=1000):
    return AlexNet(num_classes=num_classes)

def create_resnet(num_classes=1000):
    return ResNet(num_classes=num_classes)

# Example usage
if __name__ == "__main__":
    # Create a LeNet model instance
    lenet_model = create_lenet(num_classes=10)
    print(lenet_model)

    # Create an AlexNet model instance
    alexnet_model = create_alexnet(num_classes=1000)
    print(alexnet_model)

    # Create a ResNet model instance
    resnet_model = create_resnet(num_classes=1000)
    print(resnet_model)
