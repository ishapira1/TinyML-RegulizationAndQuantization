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
  - `src/quantization/`: handles quantization process

- `jobs/`: Define, handles and log slurm jobs


## Usage

To run an experiment with a predefined set of parameters, execute `src/main.py`. 

minimalist examples:

This example demonstrates how to run the script on the CIFAR-10 dataset using the ResNet18 model with a specific learning rate and regularization parameter.

```bash
cd src && python3 main.py --dataset_name CIFAR-10 --model_name resnet18 --lr 0.001 --regularization_type l2 --regularization_param 0.1 --epochs 50 --batch_size 64
```

This example shows how to run the script on the MNIST dataset using the LeNet model with a specific batch size, number of epochs, and L1 regularization.
```bash
cd src && python3 main.py --dataset_name MNIST --model_name lenet --batch_size 32 --epochs 100 --regularization_type l1 --regularization_param 0.05
```


Our quantizer script allows you to easily quantize a pre-trained model to a specified bit width.  To quantize a model, use the following command:

```python
cd src && python3 quantizer_main.py --path <model_path> --bit_width <bit_width>
```