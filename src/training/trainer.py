# @ ishapira 20231103
"""
defines the training process including the training loop,
validation, and testing procedures.
"""
import torch
from torch.optim import Adam
from src.data_loaders.dataset_loader import load_dataset
from src.models.model_registry import create_model
from tqdm import tqdm



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

        self.criterion_name = self.criterion.__class__.__name__
        self.optimizer_name = self.optimizer.__class__.__name__

        self.device = device
        self.regularization = regularization if regularization else {}
        self.verbose = verbose

        # after training
        self.train_loss = 0.0
        self.test_loss = 0.0
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0
        self.epoch_train_losses = []
        self.epoch_test_losses = []
        self.iterations = 0.0


    def _compute_accuracy(self, dataloader):  # TODO go to eval
        """
        Compute the accuracy given the output of the model and the labels.

        :param outputs: The logits output from the model.
        :param labels: The true labels.
        :return: The accuracy as a percentage.
        """
        correct = 0
        total = 0
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy
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
        return loss

    from tqdm import tqdm

    def train(self, num_epochs):
        """
        Train the model for a specified number of epochs and return the model
        along with the train and test losses and accuracies for each epoch.

        :param num_epochs: The number of epochs to train the model.
        :return: Tuple containing the train loss, test loss, train accuracy, test accuracy, and lists of train and test losses per epoch.
        """
        epoch_train_losses = []  # List to store train loss after each epoch
        epoch_test_losses = []  # List to store test loss after each epoch

        for epoch in tqdm(range(num_epochs), desc='Epochs'):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in self.dataloaders['train']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss = self._apply_regularization(loss)
                loss.backward()
                self.optimizer.step()

                # Apply L-infinity regularization directly to the parameters
                if 'linf' in self.regularization:
                    for param in self.model.parameters():
                        param.data = torch.clamp(param.data, min=-self.regularization['linf'],
                                                 max=self.regularization['linf'])

                running_loss += loss.item() * inputs.size(0)

            # Evaluation phase
            self.model.eval()
            test_running_loss = 0.0
            train_running_loss = 0.0
            with torch.no_grad():
                for inputs, labels in self.dataloaders['test']:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    test_running_loss += loss.item() * inputs.size(0)
                for inputs, labels in self.dataloaders['train']:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    train_running_loss += loss.item() * inputs.size(0)

            # Calculate average losses for the current epoch
            train_loss = train_running_loss / len(self.dataloaders['train'].dataset)
            test_loss = test_running_loss / len(self.dataloaders['test'].dataset)

            # Append average losses to the lists
            epoch_train_losses.append(train_loss)
            epoch_test_losses.append(test_loss)

            if self.verbose:
                tqdm.write(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        # Calculate accuracies after all epochs
        train_accuracy = self._compute_accuracy(self.dataloaders['train'])
        test_accuracy = self._compute_accuracy(self.dataloaders['test'])

        # save:
        self.train_loss = train_loss
        self.test_loss = test_loss
        self.train_accuracy = train_accuracy
        self.test_accuracy = test_accuracy
        self.epoch_train_losses = epoch_train_losses
        self.epoch_test_losses = epoch_test_losses
        self.iterations = len(self.dataloaders['train']) * num_epochs


        # Return the final losses, accuracies, and lists of losses per epoch
        return train_loss, test_loss, train_accuracy, test_accuracy, epoch_train_losses, epoch_test_losses


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
    optimizer = Adam(model.parameters(), lr=0.001)


    # Create trainer instance
    trainer = Trainer(model, dataloaders, criterion, optimizer, device)

    # Run training
    trainer.train(num_epochs=25)



if __name__ == "__main__":
    main()
