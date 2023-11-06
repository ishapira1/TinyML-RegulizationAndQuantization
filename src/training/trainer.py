# @ ishapira 20231103
"""
defines the training process including the training loop,
validation, and testing procedures.
"""
import torch
import torch.optim as optim
from src.data.dataset_loader import load_dataset
from src.models.model_registry import create_model



class Trainer:
    def __init__(self, model, dataloaders, criterion, optimizer, device, regularization=None, verbose=True):
        """
        Initialize the Trainer.

        :param model: The neural network model.
        :param dataloaders: A dictionary containing the 'train' and 'test' DataLoader.
        :param criterion: The loss function.
        :param optimizer: The optimizer.
        :param device: The device to run the training on.
        :param regularization: A dict with key 'l1', 'l2', or 'linf' and the regularization strength as the value.
        :param verbose: If True, print detailed information during training.
        """
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.regularization = regularization if regularization else {}
        self.verbose = verbose

    def _apply_regularization(self, loss):
        """
        Apply the specified regularization to the loss.

        :param loss: The current loss value before regularization.
        :return: The loss value after applying regularization.
        """
        if 'l1' in self.regularization:
            l1_reg = sum(param.abs().sum() for param in self.model.parameters())
            loss = loss + self.regularization['l1'] * l1_reg
        elif 'l2' in self.regularization:
            l2_reg = sum(param.pow(2.0).sum() for param in self.model.parameters())
            loss = loss + self.regularization['l2'] * l2_reg
        elif 'linf' in self.regularization:
            for param in self.model.parameters():
                param.data = torch.clamp(param.data, min=-self.regularization['linf'], max=self.regularization['linf'])
        return loss

    def train(self, num_epochs):
        """
        Train the model for a specified number of epochs.

        :param num_epochs: The number of epochs to train the model.
        """
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in self.dataloaders['train']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss = self._apply_regularization(loss)
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(self.dataloaders['train'].dataset)

            if self.verbose:
                print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}')
                self._log_test_loss()

    def _log_test_loss(self):
        """
        Log the test loss if the verbose option is enabled.
        """
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.dataloaders['test']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)

        test_loss /= len(self.dataloaders['test'].dataset)
        print(f'Test Loss: {test_loss:.4f}')


def main():
    # Example setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    dataset_name = 'MNIST'  # or 'CIFAR10', 'ImageNet', depending on your needs

    # Assuming load_dataset now returns data loaders
    train_loader = load_dataset(dataset_name=dataset_name, batch_size=batch_size, train=True)
    val_loader = load_dataset(dataset_name=dataset_name, batch_size=batch_size, train=False)
    dataloaders = {
        'train': train_loader,
        'test': val_loader
    }

    # Choose and initialize model
    # Here you need to decide which model to use and pass the appropriate arguments
    # Example: model = create_model('lenet', num_classes=10, mini=True)
    model = create_model('lenet', mini=True)

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
