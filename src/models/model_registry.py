#@ ishapira 20231103
"""
serves as a registry of model architectures. It provides
factory functions that return instances of various models, such as AlexNet, ResNet, and
LeNet, which all inherit from the base class defined in base_model.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.models.normalization_utils import replace_bn, add_ln

# Definition for the Inception-v3 model, suitable for ImageNet by default
class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, dropout_rate=0.5, use_batch_norm=False, use_layer_norm=False, pretrained=False):
        super(InceptionV3, self).__init__()

        weights = models.InceptionV3_Weights.DEFAULT if pretrained else None
        self.model = models.inception_v3(weights=weights, aux_logits=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        if use_batch_norm and use_layer_norm:
            raise ValueError("Batch normalization and layer normalization cannot both be true")

        if input_channels != 3:
            # Inception v3 expects (299, 299) sized images and 3 input channels,
            # customizing input channels is non-trivial and not supported out-of-the-box.
            raise ValueError("InceptionV3 requires 3 input channels")
        # Replace BatchNorm layers if not using batch normalization
        if not use_batch_norm:
            replace_bn(self.model)

        # Implement layer normalization if required
        if use_layer_norm:
            add_ln(self.model)

    def forward(self, x):
        x = self.dropout(self.model(x))
        return x

# Definition for the ResNet50 model, suitable for ImageNet by default
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, dropout_rate=0.5, use_batch_norm=False, use_layer_norm=False,
                 pretrained=False):
        super(ResNet50, self).__init__()

        # Check for conflicting normalization settings
        if use_batch_norm and use_layer_norm:
            raise ValueError("Batch normalization and layer normalization cannot both be true")

        # Set weights for pretrained model
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.model = models.resnet50(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

        # Replace the first convolutional layer if the input channels are not 3
        if input_channels != 3:
            self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace BatchNorm layers if not using batch normalization
        if not use_batch_norm:
            replace_bn(self.model)

        # Implement layer normalization if required
        if use_layer_norm:
            add_ln(self.model)

    def forward(self, x):
        x = self.dropout(self.model(x))
        return x


# Definition for the LeNet model, suitable for MNIST by default
class LeNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=1, dropout_rate=0.5, use_batch_norm=False, use_layer_norm=False):
        super(LeNet, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm2d(6)
            self.bn2 = nn.BatchNorm2d(16)
        if self.use_layer_norm:
            self.ln1 = nn.LayerNorm([6, 24, 24])
            self.ln2 = nn.LayerNorm([16, 8, 8])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.use_batch_norm:
            x = self.bn1(x)
        elif self.use_layer_norm:
            x = self.ln1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        if self.use_batch_norm:
            x = self.bn2(x)
        elif self.use_layer_norm:
            x = self.ln2(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(self.dropout(x)))
        x = F.relu(self.fc2(self.dropout(x)))
        x = self.fc3(self.dropout(x))
        return x


# Definition for the AlexNet model, suitable for CIFAR-10 by default
class AlexNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=3, dropout_rate=0.5, use_batch_norm=False, use_layer_norm=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
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


# Definition for the ResNet18 model, suitable for ImageNet by default
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, dropout_rate=0.5, use_batch_norm=False,
                 use_layer_norm=False, pretrained=False):
        super(ResNet18, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

        # Check for conflicting normalization settings
        if use_batch_norm and use_layer_norm:
            raise ValueError("Batch normalization and layer normalization cannot both be true")
        # Replace first conv layer if not using 3 input channels
        if input_channels != 3:
            self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Replace BatchNorm layers if not using batch normalization
        if not use_batch_norm:
            replace_bn(self.model)

        # Implement layer normalization if required
        if use_layer_norm:
            add_ln(self.model)

    def forward(self, x):
        x = self.dropout(self.model(x))
        return x


# The create_model function now includes 'resnet50' and 'inceptionv3' cases and the 'pretrained' parameter.
def create_model(arch, num_classes=1000, input_channels=3, mini=False, dropout_rate=0.5, use_batch_norm=False, use_layer_norm=False, pretrained=False):
    if arch == 'lenet':
        model = LeNet(num_classes=10 if mini else num_classes,
                      input_channels=1,
                      dropout_rate=dropout_rate,
                      use_batch_norm=use_batch_norm,
                      use_layer_norm=use_layer_norm)
    elif arch == 'alexnet':
        model = AlexNet(num_classes=10 if mini else num_classes,
                        input_channels=3 if mini else input_channels,
                        dropout_rate=dropout_rate,
                        use_batch_norm=use_batch_norm,
                        use_layer_norm=use_layer_norm)
    elif arch == 'resnet18':
        model = ResNet18(num_classes=1000 if mini else num_classes,
                         input_channels=3 if mini else input_channels,
                         dropout_rate=dropout_rate,
                         use_batch_norm=use_batch_norm,
                         use_layer_norm=use_layer_norm,
                         pretrained=pretrained)
    elif arch == 'resnet50':
        model = ResNet50(num_classes=1000 if mini else num_classes,
                         input_channels=3 if mini else input_channels,
                         dropout_rate=dropout_rate,
                         use_batch_norm=use_batch_norm,
                         use_layer_norm=use_layer_norm,
                         pretrained=pretrained)
    elif arch == 'inceptionv3':
        if mini:
            raise ValueError("InceptionV3 is not suitable for 'mini' datasets due to its input size requirement")
        model = InceptionV3(num_classes=num_classes,
                            input_channels=input_channels,
                            dropout_rate=dropout_rate,
                            use_batch_norm=use_batch_norm,
                            use_layer_norm=use_layer_norm,
                            pretrained=pretrained)
    else:
        raise ValueError(f"Unknown architecture '{arch}'")
    return model


# Example usage
if __name__ == "__main__":
    # Create a mini LeNet model instance for MNIST
    lenet_model = create_model('lenet', mini=True)
    print(lenet_model)

    # Create an AlexNet model instance for CIFAR-10
    alexnet_model = create_model('alexnet', mini=True)
    print(alexnet_model)

    # Create a ResNet18 model instance for ImageNet
    resnet_model = create_model('resnet18')
    print(resnet_model)

    # Example usage (Create a ResNet50 model instance for ImageNet)
    resnet50_model = create_model('resnet50', pretrained=True)
    print(resnet50_model)

    # Example usage (Create an Inception-v3 model instance for ImageNet)
    inceptionv3_model = create_model('inceptionv3', pretrained=True)
    print(inceptionv3_model)
