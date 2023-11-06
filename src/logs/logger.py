import os
import json
import datetime
from torch import save
import os
import getpass

RESULTS_FILE_NAME_IN_LOGS = 'results.json'
CHECKPOINT_FILE_NAME_IN_LOGS = 'model_checkpoint.pth'


class Logger:
    def __init__(self, log_dir="logs"):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))  # "logs"
        self.main_dir = os.path.dirname(os.path.dirname(self.script_dir)) # main dir

        self.log_dir = os.path.join(self.main_dir, 'results')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def log(self, model, train_loss, test_loss, model_name, dataset_name, reg_name, param, train_accuracy, test_accuracy, **kwargs):
        # Create a unique identifier for the experiment based on the current timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        user = getpass.getuser()
        exp_name = f"{user}_{model_name}_{dataset_name}_{reg_name}_{param}_{timestamp}"
        exp_dir = os.path.join(self.log_dir, exp_name)
        os.makedirs(exp_dir)

        # Save parameters and results
        params_and_results = {
            'runner_id':user,
            'model_name':model_name,
            'dataset_name':dataset_name,
            'regularization': reg_name,
            'regularization_param': param,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'timestamp': timestamp
        }

        # Update the dictionary with any additional keyword arguments
        params_and_results.update(kwargs)

        with open(os.path.join(exp_dir, RESULTS_FILE_NAME_IN_LOGS), 'w') as f:
            json.dump(params_and_results, f, indent=4)

        # Save the model checkpoint
        checkpoint_path = os.path.join(exp_dir, CHECKPOINT_FILE_NAME_IN_LOGS)
        save(model.state_dict(), checkpoint_path)

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
