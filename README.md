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
  - `src/models/`: Contains the definitions and utilities for the models.
  - `src/training/`: Manages the training process of the models.
  - `src/logs/`: Includes `logger.py` for logging experiment details and `parser.py` for converting logs into a summarized CSV file.

