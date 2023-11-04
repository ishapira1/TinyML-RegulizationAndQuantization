# @ ishapira 20231103
"""
defines the training process including the training loop,
validation, and testing procedures.
"""

import torch
import torch.optim as optim
from .data.dataset_loader import get_dataloader
from .models.model_registry import create_alexnet, create_resnet, create_lenet


class Trainer:
    def __init__(self, model, dataloaders, criterion, optimizer, device):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in self.dataloaders['train']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(self.dataloaders['train'].dataset)
            print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}')

            # Add validation and testing loops if needed


def main():
    # Example setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get data loaders
    dataloaders = {
        'train': get_dataloader('CIFAR10', 'train'),
        'val': get_dataloader('CIFAR10', 'val')
    }

    # Choose and initialize model
    model = create_resnet(num_classes=10)

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Create trainer instance
    trainer = Trainer(model, dataloaders, criterion, optimizer, device)

    # Run training
    trainer.train(num_epochs=25)


if __name__ == "__main__":
    main()
