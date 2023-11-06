import os
import json
import datetime
from torch import save

class Logger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def log(self, params, regularization, model, train_loss, test_loss, train_accuracy, test_accuracy):
        # Create a unique identifier for the experiment based on the current timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        exp_name = f"{params['model_name']}_{params['dataset_name']}_{regularization}_{timestamp}"
        exp_dir = os.path.join(self.log_dir, exp_name)
        os.makedirs(exp_dir)

        # Save parameters and results
        params_and_results = {
            'params': params,
            'regularization': regularization,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'timestamp': timestamp
        }
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(params_and_results, f, indent=4)

        # Save the model checkpoint
        checkpoint_path = os.path.join(exp_dir, 'model_checkpoint.pth')
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
