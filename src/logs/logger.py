import os
import json
import datetime
from torch import save
import os
import getpass
import glob

RESULTS_FILE_NAME_IN_LOGS = 'results.json'
CHECKPOINT_FILE_NAME_IN_LOGS = 'model_checkpoint.pth'

from src.training.trainer import Trainer

def is_json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False

def get_attributes(trainer_instance):
    return {attr: value for attr, value in vars(trainer_instance).items() if is_json_serializable(value)}

def get_model_size_in_mb(model, bit_width=None):
    """
    Calculate the total size of a PyTorch model in megabytes (MB).

    :param model: The PyTorch model.
    :return: The total size of the model in MB.
    """
    param_size = 0
    if bit_width is not None:
        for param in model.parameters():
            # if is?nstance(param, )
            param_size += param.nelement() * bit_width/8
    else:
        for param in model.parameters():
            # if is?nstance(param, )
            param_size += param.nelement() * param.element_size()
    # buffer_size = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    total_size = param_size #+ buffer_size
    total_size_mb = total_size / (1024 ** 2)  # Convert bytes to MB

    return total_size_mb

class Logger:
    def __init__(self, log_dir="logs"):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))  # "logs"
        self.main_dir = os.path.dirname(os.path.dirname(self.script_dir)) # main dir

        self.log_dir = os.path.join(self.main_dir, 'results')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def get_checkpoint(self, model_name, dataset_name, reg_name, param):
        """
        Gets the model checkpoint path, ignoring timestamp and user
        If there are multiple identical configurations with different users/timestamps, will return them all
        returns various checkpoint paths as a list
        """
        exp_name = f"*_{model_name}_{dataset_name}_{reg_name}_{param}_*"

        experiments = []
        for filename in glob.glob(os.path.join(self.log_dir, exp_name)):
            experiments.append(os.path.join(filename, CHECKPOINT_FILE_NAME_IN_LOGS))
        return experiments

    def log(self, model, trainer, model_name, dataset_name, reg_name, param, **kwargs):
        # Create a unique identifier for the experiment based on the current timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        user = getpass.getuser()
        exp_name = f"{user}_{model_name}_{dataset_name}_{reg_name}_{param}_{timestamp}"
        exp_dir = os.path.join(self.log_dir, exp_name)
        os.makedirs(exp_dir)
        model_size = get_model_size_in_mb(model)

        # Save parameters and results
        params_and_results = {
            'runner_id':user,
            'model_name':model_name,
            'dataset_name':dataset_name,
            'regularization': reg_name,
            'regularization_param': param,
            'timestamp': timestamp,
            'bit_width': 32,
            'model_size_mb': model_size
        }

        params_and_results.update(get_attributes(trainer))

        # Update the dictionary with any additional keyword arguments
        params_and_results.update(kwargs)

        with open(os.path.join(exp_dir, RESULTS_FILE_NAME_IN_LOGS), 'w') as f:
            json.dump(params_and_results, f, indent=4)

        # Save the model checkpoint
        checkpoint_path = os.path.join(exp_dir, CHECKPOINT_FILE_NAME_IN_LOGS)
        save(model.state_dict(), checkpoint_path)

    def append_log(self, checkpoint_path, bit_width, model, train_loss, test_loss, model_name, dataset_name, reg_name, param, train_accuracy, test_accuracy, **kwargs):
        exp_dir = os.path.split(checkpoint_path)[0]

        sub_dir = os.path.join(exp_dir, "bit_width_{}".format(bit_width))
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        user = getpass.getuser()
        model_size = get_model_size_in_mb(model, bit_width=bit_width)

        params_and_results = {
            'runner_id':user,
            'model_name':model_name,
            'dataset_name':dataset_name,
            'bit_width': bit_width,
            'regularization': reg_name,
            'regularization_param': param,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'timestamp': timestamp,
            'model_size_mb': model_size
        }
        
        params_and_results.update(kwargs)

        with open(os.path.join(sub_dir, RESULTS_FILE_NAME_IN_LOGS), 'w') as f:
            json.dump(params_and_results, f, indent=4)

        # stop saving model checkpoint file
        # save(model.state_dict(), os.path.join(sub_dir, CHECKPOINT_FILE_NAME_IN_LOGS)) 

    def save_model_weights(self, model, exp_dir):
        weights_path = os.path.join(exp_dir, 'model_weights.pth')
        save(model.state_dict(), weights_path)

# Example usage:
# logger = Logger()
# logger.log(params={
#                'dataset_name': 'CIFAR-10',
#                'model_name': 'resnet18',
#                'batch_size': 64,
#                'learning_rate': 0.001,
#                ...
#            },
#            regularization='l2',
#            model=model,
#            train_loss=train_loss,
#            test_loss=test_loss,
#            train_accuracy=train_accuracy,
#            test_accuracy=test_accuracy)
