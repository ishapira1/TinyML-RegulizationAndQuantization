# @ ishapira 20231103
"""
designed to handle the loading and preprocessing of datasets
"""

from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
import os
script_path = os.path.abspath(__file__)

# Find the parent directory of the script, which is assumed to be 'data'
parent_directory = os.path.dirname(script_path)

script_dir = os.path.dirname(os.path.abspath(__file__))  # "data"
main_dir = os.path.dirname(os.path.dirname(script_dir))  # main dir
data_dir = os.path.join(main_dir, 'data')


# Dataset configuration class to handle dimensions and other properties
class DatasetConfig:
    def __init__(self, name):
        self.name = name
        self.dims = {
            'CIFAR-10': (3, 32, 32),
            'MNIST': (1, 28, 28),
            'ImageNet': (3, 224, 224),  # Assuming you're using a scaled-down version
            'FashionMNIST': (1, 28, 28)
        }

    def get_dims(self):
        return self.dims[self.name]


# Function to get the appropriate transformations
def get_transforms(dataset_name):
    if dataset_name in ['MNIST', 'FashionMNIST']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset_name == 'CIFAR-10':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_name == 'ImageNet':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"Transforms for dataset {dataset_name} are not defined.")


# Function to load datasets
def load_dataset(dataset_name, batch_size, train=True, download=True):
    transforms = get_transforms(dataset_name)
    dataset_directory = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_directory, exist_ok=True) # Make sure the dataset directory exists



    if dataset_name == 'CIFAR-10':
        dataset = datasets.CIFAR10(root=dataset_directory, train=train, download=download, transform=transforms)
    elif dataset_name == 'MNIST':
        dataset = datasets.MNIST(root=dataset_directory, train=train, download=download, transform=transforms)
    elif dataset_name == 'ImageNet':
        dataset = datasets.ImageFolder(root=dataset_directory, transform=transforms)  # Adjust the path as needed
    elif dataset_name == 'FashionMNIST':
        dataset = datasets.FashionMNIST(root=dataset_directory, train=train, download=download, transform=transforms)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


# Example usage:
if __name__ == "__main__":
    # Example: Load CIFAR-10 training data
    batch_size = 64
    dataset_name = 'CIFAR-10'
    config = DatasetConfig(dataset_name)


    print(f"loading {dataset_name} with data dimensions: {config.get_dims()}")
    train_loader = load_dataset(dataset_name, batch_size, train=True, download=True)

    print(f"Loaded {dataset_name} with data dimensions: {config.get_dims()}")

    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Iterate over the data
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # Here you would typically feed your images to the model
        break  # Just to illustrate we don't run through all data here
