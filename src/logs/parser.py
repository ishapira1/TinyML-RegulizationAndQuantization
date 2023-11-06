# @ ishapira 20231106
# parser.py
import os
import json
import pandas as pd
from logger import Logger, RESULTS_FILE_NAME_IN_LOGS, CHECKPOINT_FILE_NAME_IN_LOGS
from torch import load
import torch
from tqdm import tqdm



def calculate_weight_statistics(checkpoint_path):
    """
    Calculate the statistics of the weights from the model checkpoint.

    :param checkpoint_path: Path to the model checkpoint file.
    :return: A dictionary with total number of weights, L1 norm, L2 norm, max and min weight values.
    """
    state_dict = load(checkpoint_path, map_location=torch.device('cpu'))
    # Filter out non-floating point parameters
    float_params = [param for param in state_dict.values() if torch.is_floating_point(param)]

    total_weights = sum(param.numel() for param in float_params)
    l1_norm = sum(torch.norm(param, 1).item() for param in float_params)
    l2_norm = sum(torch.norm(param, 2).item() for param in float_params)
    max_weight = max(param.max().item() for param in float_params)
    min_weight = min(param.min().item() for param in float_params)

    return {
        'total_weights': total_weights,
        'l1_norm': l1_norm,
        'l2_norm': l2_norm,
        'max_weight': max_weight,
        'min_weight': min_weight
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

    # Save the DataFrame to a CSV file
    csv_file = os.path.join(logger.log_dir, 'all_results.csv')
    df.to_csv(csv_file, index=False)

# To run the parser function when the script is executed
if __name__ == '__main__':
    parser()
