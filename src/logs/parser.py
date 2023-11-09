# @ ishapira 20231106
# parser.py
import os
import json
import pandas as pd
from src.logs.logger import Logger, RESULTS_FILE_NAME_IN_LOGS, CHECKPOINT_FILE_NAME_IN_LOGS
from torch import load
import torch
from tqdm import tqdm
import datetime


def get_model_size_in_mb(model):
    """
    Calculate the total size of a PyTorch model in megabytes (MB).

    :param model: The PyTorch model.
    :return: The total size of the model in MB.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    total_size = param_size + buffer_size
    total_size_mb = total_size / (1024 ** 2)  # Convert bytes to MB
    return total_size_mb


def calculate_weight_statistics(checkpoint_path):
    """
    Calculate the statistics of the weights from the model checkpoint.

    :param checkpoint_path: Path to the model checkpoint file.
    :return: A dictionary with total number of weights, L1 norm, L2 norm, max and min weight values, and model size in MB.
    """
    state_dict = load(checkpoint_path, map_location=torch.device('cpu'))
    # Filter out non-floating point parameters
    float_params = [param for param in state_dict.values() if torch.is_floating_point(param)]

    total_weights = sum(param.numel() for param in float_params)
    l1_norm = sum(torch.norm(param, 1).item() for param in float_params)
    l2_norm = sum(torch.norm(param, 2).item() for param in float_params)
    max_weight = max(param.max().item() for param in float_params)
    min_weight = min(param.min().item() for param in float_params)

    # Calculate model size in bytes and convert to megabytes
    total_size_bytes = sum(param.nelement() * param.element_size() for param in float_params)
    total_size_mb = total_size_bytes / (1024 ** 2)

    return {
        'total_weights': total_weights,
        'l1_norm': l1_norm,
        'l2_norm': l2_norm,
        'max_weight': max_weight,
        'min_weight': min_weight,
        'model_size_mb': total_size_mb
    }


def parser():
    # Create a Logger instance to get the log directory
    logger = Logger()
    log_dir = logger.log_dir

    # List to hold all experiment records
    all_records = []

    # Iterate over all experiment directories in the log directory with a progress bar
    for exp_name in tqdm(os.listdir(log_dir), desc='Parsing experiment results'):
        exp_dir = os.path.join(log_dir, exp_name)
        results_file = os.path.join(exp_dir, RESULTS_FILE_NAME_IN_LOGS)
        checkpoint_path = os.path.join(exp_dir, CHECKPOINT_FILE_NAME_IN_LOGS)

        # Check if the checkpoint file exists
        if os.path.isfile(checkpoint_path):
            weight_stats = calculate_weight_statistics(checkpoint_path)

            # Check if the results file exists
            if os.path.isfile(results_file):
                # Open the results file and load the JSON data
                with open(results_file, 'r') as f:
                    record = json.load(f)
                    record.update(weight_stats)
                    all_records.append(record)

    # Create a DataFrame from the records
    df = pd.DataFrame(all_records)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # Save the DataFrame to a CSV file
    csv_file = os.path.join(logger.log_dir, f'all_results_{timestamp}.csv')

    df.to_csv(csv_file, index=False)

# To run the parser function when the script is executed
if __name__ == '__main__':
    parser()
