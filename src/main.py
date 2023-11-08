import torch
from src.data_loaders.dataset_loader import load_dataset
from src.models.model_registry import create_model
from src.training.trainer import Trainer
from src.logs.logger import Logger
from tqdm import tqdm
from torch.optim import Adam

def set_seed(seed):
    """
    Set the seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def train_model(model, dataset, device, batch_size=128, num_epochs=50, lr=0.001, verbose = True, reg_name=None, reg_param=None ):
    """
    Train the model and return the results.
    """
    train_loader = load_dataset(dataset_name=dataset, batch_size=batch_size, train=True)
    val_loader = load_dataset(dataset_name=dataset, batch_size=batch_size, train=False)
    dataloaders = {
        'train': train_loader,
        'test': val_loader
    }
    optimizer = Adam(model.parameters(), lr=lr)
    regularization = {} if not reg_name else {reg_name:reg_param}
    trainer = Trainer(model, dataloaders, criterion = torch.nn.CrossEntropyLoss(), optimizer=optimizer, device=device, regularization=regularization, verbose=verbose)
    return trainer.train(num_epochs)



def run_experiments(device):
    """
    Run experiments across different datasets, models, and regularizations.
    """
    for dataset_name in tqdm(DATASETS, desc='Datasets', leave=False):
        for model_name in tqdm(MODELS, desc='Models', leave=False):
            # Compatibility check
            if model_name not in COMPATIBLE_MODELS[dataset_name]:
                continue

            for reg_name, reg_params in tqdm(REGULARIZATIONS.items(), desc='Regularizations', leave=False):
                if reg_params:  # Regularizations with parameters
                    for param in tqdm(reg_params, desc='Params', leave=False):
                        dropout_rate = param if reg_name == 'dropout' else 0

                        model = create_model(model_name, mini=False, dropout_rate=dropout_rate,
                                     use_batch_norm=False,
                                     use_layer_norm=False)

                        model.to(device)

                        train_loss, test_loss, train_accuracy, test_accuracy = train_model(model, dataset_name, reg_name=reg_name, reg_param=param, device=device, batch_size=batch_size, num_epochs=epochs,
                                    lr=lr, verbose=True)
                        logger.log(model, train_loss, test_loss, model_name, dataset_name, reg_name, param, train_accuracy, test_accuracy, epochs=epochs, lr=lr, device=str(device), batch_size=batch_size, seed=seed)
                else:  # Regularizations without parameters
                    # Set flags for batch_norm and layer_norm based on reg_name
                    use_batch_norm = reg_name == 'batch_norm'
                    use_layer_norm = reg_name == 'layer_norm'

                    model = create_model(model_name, mini=False, dropout_rate=0,
                                 use_batch_norm=use_batch_norm,
                                 use_layer_norm=use_layer_norm)
                    model.to(device)
                    train_loss, test_loss, train_accuracy, test_accuracy = train_model(model, dataset_name, device=device, batch_size=batch_size,num_epochs=epochs, lr=lr, verbose=True)
                    logger.log(model, train_loss, test_loss, model_name, dataset_name, reg_name, None, train_accuracy, test_accuracy, epochs=epochs, lr=lr, device=str(device), batch_size=batch_size, seed=seed)



if __name__ == '__main__':
    # Check for CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")
    seed = 42
    set_seed(seed)

    # Define your epochs, learning rate, dataset names, model names, and regularizations here
    epochs = 10
    batch_size = 256
    lr = 0.001
    DATASETS = ['CIFAR-10'] #['CIFAR-10', 'MNIST', 'IMAGENET', 'FASHIONMNIST']
    MODELS = ['resnet18']# 'alexnet', 'resnet18']
    COMPATIBLE_MODELS = {
        # Define which models are compatible with which datasets
        #'CIFAR-10': ['lenet', 'resnet18'],
        'CIFAR-10': ['resnet18'],
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
