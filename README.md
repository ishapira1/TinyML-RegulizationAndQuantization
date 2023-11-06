# TinyML-RegulizationAndQuantization

Before running any scripts, install the package to configure the necessary environment and avoid issues with relative paths:

```sh
pip install -e .
```
## Directory Structure Overview

- `data/`: Contains all datasets used in the experiments.
- `results/`: Stores all results including model checkpoints, logs, and CSV files, automatically capturing metadata such as the user, timestamp, and key statistics.
- `src/`: The main codebase for this project.
  - `src/data_loader/dataset_loader.py`: Script for loading various datasets.
  - `src/models/`: Contains the definitions and utilities for the models.git
  - `src/training/`: Manages the training process of the models.
  - `src/logs/`: Includes `logger.py` for logging experiment details and `parser.py` for converting logs into a summarized CSV file.

## Usage

To run an experiment with a predefined set of parameters, execute `src/main.py`. 

minimalist examples (implemented in `src/main.py` ):

```python
import torch
from src.data_loaders.dataset_loader import load_dataset
from src.models.model_registry import create_model
from src.training.trainer import Trainer
from src.logs.logger import Logger
from tqdm import tqdm
from torch.optim import Adam

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 42
set_seed(seed)

# Define your epochs, learning rate, dataset names, model names, and regularizations here
epochs = 2
batch_size = 256
lr = 0.001
DATASETS = ['MNIST','CIFAR-10', 'MNIST', 'IMAGENET', 'FASHIONMNIST']
MODELS = ['lenet','alexnet', 'resnet18']
COMPATIBLE_MODELS = {
    # Define which models are compatible with which datasets
    'CIFAR-10': ['lenet', 'resnet18'],
    'MNIST': ['lenet'],
    # etc.
}
REGULARIZATIONS = {
    'none': None,  # No regularization parameters needed for baseline
    'batch_norm': None,  # Batch normalization typically does not require explicit parameters
    'layer_norm': None,  # Layer normalization also typically does not require explicit parameters
    'dropout': [0.3, 0.5, 0.7],  # Different dropout rates to experiment with
    'l1': [0.1, 0.01, 0.001, 0.0001],  # Different L1 regularization strengths
    'l2': [0.1, 0.01, 0.001, 0.0001],  # Different L2 regularization strengths
    'l_infinty': [0.1, 0.01, 0.001]  # Different L-infinity regularization strengths
}

logger = Logger()
run_experiments(device)
```
