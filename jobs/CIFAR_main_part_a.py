import torch
from src.data_loaders.dataset_loader import load_dataset
from src.models.model_registry import create_model
from src.training.trainer import Trainer
from src.logs.logger import Logger
from tqdm import tqdm
from torch.optim import Adam
# import warnings
# from urllib3.exceptions import InsecureRequestWarning
#
# warnings.filterwarnings("ignore", category=InsecureRequestWarning)

def set_seed(seed):
    """
    Set the seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def train_model(model, dataset, device, batch_size=128, num_epochs=50, lr=0.001,momentum=0.9, verbose = True, reg_name=None, reg_param=None ):
    """
    Train the model and return the results.
    """
    train_loader = load_dataset(dataset_name=dataset, batch_size=batch_size, train=True)
    val_loader = load_dataset(dataset_name=dataset, batch_size=batch_size, train=False)
    dataloaders = {
        'train': train_loader,
        'test': val_loader
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    lr_decay_epochs = [num_epochs // 2, num_epochs * 3 // 4]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_epochs, gamma=0.1)

    regularization = {} if not reg_name else {reg_name:reg_param}
    trainer = Trainer(model, dataloaders, criterion = torch.nn.CrossEntropyLoss(), optimizer=optimizer, device=device, regularization=regularization, verbose=verbose, lr_scheduler=lr_scheduler)
    trainer.train(num_epochs)
    return trainer



def run_experiments(device, pretrained=False, num_classes=10):
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

                        model = create_model(model_name,num_classes=num_classes, mini=False, dropout_rate=dropout_rate,
                                     use_batch_norm=True,
                                     use_layer_norm=False, pretrained=pretrained)

                        model.to(device)

                        trainer = train_model(model, dataset_name, reg_name=reg_name, reg_param=param, device=device, batch_size=batch_size, num_epochs=epochs,
                                    lr=lr, verbose=True)
                        logger.log(model, trainer, model_name, dataset_name, reg_name, param,
                                   lr = lr, device=str(device), batch_size=batch_size, seed=seed, pretrained=pretrained, num_epochs=epochs)
                else:  # Regularizations without parameters
                    # Set flags for batch_norm and layer_norm based on reg_name
                    use_batch_norm = reg_name == 'batch_norm'
                    use_layer_norm = reg_name == 'layer_norm'

                    model = create_model(model_name,num_classes, mini=False, dropout_rate=0,
                                 use_batch_norm=True,
                                 use_layer_norm=use_layer_norm, pretrained=pretrained)
                    print(model)
                    model.to(device)
                    trainer = train_model(model, dataset_name, reg_name=reg_name, reg_param=None,device=device, batch_size=batch_size,num_epochs=epochs, lr=lr, verbose=True)
                    logger.log(model, trainer, model_name, dataset_name, reg_name, None,
                               lr=lr, device=str(device), batch_size=batch_size, seed=seed,
                               pretrained=pretrained, num_epochs=epochs)



if __name__ == '__main__':
    # Check for CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")
    seed = 42
    set_seed(seed)

    # # Define your epochs, learning rate, dataset names, model names, and regularizations here
    # epochs = 10
    # batch_size = 256
    # lr = 0.001
    # DATASETS = ['CIFAR-10'] #['CIFAR-10', 'MNIST', 'ImageNet', 'FASHIONMNIST']
    # MODELS = ['resnet18']# 'alexnet', 'resnet18']
    # COMPATIBLE_MODELS = {
    #     # Define which models are compatible with which datasets
    #     #'CIFAR-10': ['lenet', 'resnet18'],
    #     'CIFAR-10': ['resnet18'],
    #     # etc.
    # }


    # imagenet pretrained:
    epochs = 20
    batch_size = 128
    lr = 0.001
    DATASETS = ['CIFAR-10'] #['CIFAR-10', 'MNIST', 'ImageNet', 'FASHIONMNIST']
    MODELS = ['resnet18']# 'alexnet', 'resnet18']
    pretrained = True
    COMPATIBLE_MODELS = {
        # Define which models are compatible with which datasets
        'CIFAR-10': ['resnet18'],
       # 'ImageNet': ['resnet50'],
        # etc.

    }



    REGULARIZATIONS = {
        'none': None,  # No regularization parameters needed for baseline
        'batch_norm': None,  # Batch normalization typically does not require explicit parameters
        #'layer_norm': None,  # Layer normalization also typically does not require explicit parameters
        'dropout': [0.3, 0.5, 0.7],  # Different dropout rates to experiment with
        'l1': [
            1e-5,  # Very Small
            2e-5,
            5e-5,
            1e-4,  # Small
            2e-4,
            5e-4,
            1e-3,  # Moderate
            2e-3,
            5e-3,  # High
            1e-2   # Very High
        ],
        # 'l2': [
        #         1e-5,  # Very Low
        #         2e-5,
        #         5e-5,  # Low
        #         1e-4,
        #         2e-4,
        #         5e-4,  # Medium
        #         1e-3,  # High
        #         2e-3,
        #         5e-3,  # Very High
        #         1e-2
        #     ],
        # 'l_infinty': [0.1, 0.01, 1,10,100]  # Different L-infinity regularization strengths
    }

    logger = Logger()
    run_experiments(device, pretrained = False, num_classes=10)
