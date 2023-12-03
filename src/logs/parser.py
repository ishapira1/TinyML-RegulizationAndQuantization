# @ ishapira 20231106
# parser.py
import os
import json
import pandas as pd
from src.logs.logger import Logger, RESULTS_FILE_NAME_IN_LOGS, CHECKPOINT_FILE_NAME_IN_LOGS, save_dataframe
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



def load_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data


def calculate_weight_statistics(checkpoint_path):
    """
    Calculate the statistics of the weights from the model checkpoint, including the total number
    of trainable parameters.

    :param checkpoint_path: Path to the model checkpoint file.
    :return: A dictionary with total number of weights, number of trainable parameters,
             L1 norm, L2 norm, max and min weight values, and model size in MB.
    """
    state_dict = load(checkpoint_path, map_location=torch.device('cpu'))

    # Filter out parameters by dtype, accounting for quantized values as well.
    float_params = []
    quantized_params = []
    trainable_params = []
    for key in state_dict:
        # FIXME: I think I can make this flexibly handle qint8, qint16, and quint8, also,
        # not sure if all quantized models will have a "dtype" flag I can use to check for quantization...
        if key.split(".")[-1] == "dtype": # for quantized models, the dtype is stored in the state dict
            if state_dict[key] == torch.qint8:
                param = state_dict[".".join(key.split(".")[:-1] + ["_packed_params"])]
                quantized_params.append(param[0]) # weights
                quantized_params.append(param[0]) # bias
            else:
                print("Unsupported dtype present in quantized model: {}".format(state_dict[key]))
        elif key.split(".")[-1] == "_packed_params":
            continue
        else:
            value = state_dict[key]
            if torch.is_floating_point(value):
                float_params.append(value)
                if value.requires_grad:
                    trainable_params.append(value)

    total_params_float_precision = float_params + [torch.dequantize(param) for param in quantized_params]

    total_weights = sum(param.numel() for param in total_params_float_precision)
    l1_norm = sum(torch.norm(param, 1).item() for param in total_params_float_precision)
    l2_norm = sum(torch.norm(param, 2).item() for param in total_params_float_precision)
    max_weight = max(param.max().item() for param in total_params_float_precision)
    min_weight = min(param.min().item() for param in total_params_float_precision)

    # Calculate model size in bytes and convert to megabytes
    total_size_bytes = sum(param.nelement() * param.element_size() for param in float_params)
    total_size_bytes += sum(param.nelement() * torch.int_repr(param).element_size() for param in quantized_params)
    total_size_mb = total_size_bytes / (1024 ** 2)

    return {
        'total_weights': total_weights,
        # 'l1_norm': l1_norm,
        # 'l2_norm': l2_norm,
        # 'max_weight': max_weight,
        # 'min_weight': min_weight,
        # 'model_size_mb': total_size_mb
    }



def my_eval(x):
    try:
        return eval(x)
    except:
        return x

def extract_key_if_dict(value):
    if isinstance(value, dict):
        # Extract and return the first key from the dictionary
        try:
            return next(iter(value))
        except:
            return 'none'
    else:
        # Return the value as is if it's not a dictionary
        return value



def clean_data(df):
    # Merge 'num_epochs' and 'epochs' columns
    
    if "epochs" in df.columns:
        df['num_epochs'] = df['num_epochs'].fillna(df['epochs'])
        df = df.drop(columns=['epochs'])
    df["quantization_method"] = df["quantization_method"].fillna('none')


    # Standardize 'regularization' values
    df['regularization'] = df['regularization'].apply(extract_key_if_dict)

    # Columns to consider for identifying duplicates
    duplicate_cols = ['model_name', 'seed', 'dataset_name', 
                      'regularization', 'regularization_param', 'num_epochs', 
                      'train_loss', 'test_loss']

    # Drop duplicates based on specified columns
    df = df.drop_duplicates(subset=duplicate_cols)

    return df




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

        # Initialize record for each experiment
        experiment_record = {}

        # Check if the checkpoint file exists
        if os.path.isfile(results_file):
            try:
                weight_stats = calculate_weight_statistics(checkpoint_path)
                experiment_record.update(weight_stats)
            except:
                print("="*100)
                pass

            # Check if the results file exists
            if os.path.isfile(results_file):
                with open(results_file, 'r') as f:
                    record = json.load(f)
                    experiment_record.update(record)
                    experiment_record['path'] = exp_dir


        # Load quantization models
        if os.path.isdir(exp_dir):
            for quantization_method in os.listdir(exp_dir):
                
                quantization_dir = os.path.join(exp_dir, quantization_method)
                results_file = os.path.join(quantization_dir, RESULTS_FILE_NAME_IN_LOGS)
                checkpoint_path = os.path.join(quantization_dir, CHECKPOINT_FILE_NAME_IN_LOGS)
                
                if os.path.isfile(results_file):
                    # Check if the results file exists
                    if os.path.isfile(results_file):
                        with open(results_file, 'r') as f:
                            try:
                                sub_record = json.load(f)
                            except:
                                continue
                            bit_width = sub_record.get('bit_width', 'unknown')
                            sub_record_path = f'bit_{bit_width}_path'
                            experiment_record[sub_record_path] = quantization_dir


                            # Prefix sub-record keys and add to experiment record
                            for key in ['train_loss', 'test_loss', 'train_accuracy', 'test_accuracy']:
                                if key in sub_record:
                                    prefixed_key = f'bit_{bit_width}_{key}'
                                    experiment_record[prefixed_key] = sub_record[key]

        # Add the experiment record to the list
        all_records.append(experiment_record)
    
    print()
    for record in all_records:
        if 'model_name' not in record:
            print(f"bad record! {record}")

    # Create a DataFrame from the records
    df = pd.DataFrame(all_records)
    df = clean_data(df)
    
    save_dataframe(df, logger)
    return df


# To run the parser function when the script is executed
if __name__ == '__main__':
    parser()
