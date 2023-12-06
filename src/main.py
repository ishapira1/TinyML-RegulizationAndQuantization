import torch
from src.data_loaders.dataset_loader import load_dataset
from src.models.model_registry import create_model
from src.training.trainer import Trainer
from src.logs.logger import Logger
from tqdm import tqdm
from torch.optim import Adam
import argparse

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
    if reg_param > 0:  # Regularizations with parameters
        dropout_rate = reg_param if reg_name == 'dropout' else 0

        model = create_model(model_name,num_classes=num_classes, mini=False, dropout_rate=dropout_rate,
                        use_batch_norm=False,
                        use_layer_norm=False, pretrained=pretrained)

        model.to(device)

        trainer = train_model(model, dataset_name, reg_name=reg_name, reg_param=reg_param, device=device, batch_size=batch_size, num_epochs=epochs,
                    lr=lr, verbose=True)
        logger.log(model, trainer, model_name, dataset_name, reg_name, reg_param,
                    lr = lr, device=str(device), batch_size=batch_size, seed=seed, pretrained=pretrained, num_epochs=epochs)
    else:  # Regularizations without parameters
        # Set flags for batch_norm and layer_norm based on reg_name
        use_batch_norm = reg_name == 'batch_norm'
        use_layer_norm = reg_name == 'layer_norm'

        model = create_model(model_name,num_classes, mini=False, dropout_rate=0,
                        use_batch_norm=False,
                        use_layer_norm=use_layer_norm, pretrained=pretrained)
        model.to(device)
        trainer = train_model(model, dataset_name, reg_name=reg_name, reg_param=None,device=device, batch_size=batch_size,num_epochs=epochs, lr=lr, verbose=True)
        logger.log(model, trainer, model_name, dataset_name, reg_name, None,
                    lr=lr, device=str(device), batch_size=batch_size, seed=seed,
                    pretrained=pretrained, num_epochs=epochs)




def setup_argparser():
    """
    Set up the argument parser for the script.
    """
    parser = argparse.ArgumentParser(description="Run experiments with different configurations")

    # Experiment configuration arguments
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument("--dataset_name", type=str, default="CIFAR-10", help="Name of the dataset to be used")
    parser.add_argument("--model_name", type=str, default="resnet18", help="Model architecture to be used")
    
    # Regularization configuration arguments
    parser.add_argument("--regularization_type", choices=['l1', 'l2', 'l_infinty', 'none'], required=True,
                        help="Type of regularization (l1, l2, l_infinty, none)")
    parser.add_argument("--regularization_param", type=float, required=False, default=0,
                        help="Regularization parameter value")

    return parser.parse_args()



if __name__ == '__main__':
    args = setup_argparser()

    # Set device and seed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")
    set_seed(args.seed)

    # Define other variables based on args
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    dataset_name = args.dataset_name
    model_name = args.model_name
    reg_name = args.regularization_type
    reg_param = args.regularization_param
    # Run experiments with parsed arguments
    logger = Logger()
    run_experiments(device, pretrained=False, num_classes=10)

