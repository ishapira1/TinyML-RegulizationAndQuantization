import torch
from src.data_loaders.dataset_loader import load_dataset
from src.models.model_registry import create_model
from src.training.trainer import Trainer
from src.logs.logger import Logger
from src.quantization.quantizer import Quantizer
from tqdm.auto import tqdm
from torch.optim import Adam
import os, argparse, sys

def set_seed(seed):
    """
    Set the seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def quantize_model(model, dataset, device,  bit_width=8, batch_size=128, verbose = True):
    """
    Train the model and return the results.
    """
    train_loader = load_dataset(dataset_name=dataset, batch_size=batch_size, train=True)
    test_loader = load_dataset(dataset_name=dataset, batch_size=batch_size, train=False)
    calibration_loader = load_dataset(dataset_name=dataset, batch_size=batch_size, train=True)
    dataloaders = {
        'train': train_loader,
        'test': test_loader,
        'calibration': calibration_loader,
    }
    quantizer = Quantizer(model, dataloaders, criterion = torch.nn.CrossEntropyLoss(), device=device, bit_width=bit_width, verbose=verbose)

    return quantizer.quantize()

def run_experiments(device, num_classes=10, pretrained=False):
    """
    Run experiments across different datasets, models, and regularizations.
    """
    for bit_width in tqdm(BIT_WIDTH, desc='Bit Widths', leave=False):
        for dataset_name in tqdm(DATASETS, desc='Datasets', leave=False):
            for model_name in tqdm(MODELS, desc='Models', leave=False):
                # Compatibility check
                if model_name not in COMPATIBLE_MODELS[dataset_name]:
                    continue

                for reg_name, reg_params in tqdm(REGULARIZATIONS.items(), desc='Regularizations', leave=False):
                    if reg_params:  # Regularizations with parameters
                        for param in tqdm(reg_params, desc='Params', leave=False):
                            checkpoint_paths = logger.get_checkpoint(model_name, dataset_name, reg_name, param)
                            for checkpoint_path in tqdm(checkpoint_paths, desc='Checkpoints', leave=False):
                                dropout_rate = param if reg_name == 'dropout' else 0


                                model = create_model(model_name, num_classes=num_classes,mini=False, dropout_rate=dropout_rate,
                                            use_batch_norm=False,
                                            use_layer_norm=False)
                                print(f'quantizing model at {checkpoint_path}')
                                state_dict = torch.load(checkpoint_path)
                                model.load_state_dict(state_dict)

                                model.to(device)

                                quantized_model, train_loss, test_loss, train_accuracy, test_accuracy, train_kl, test_kl, reconstruction_loss = quantize_model(model, dataset_name,  bit_width=bit_width, device=device, batch_size=batch_size, verbose=True)

                                quantized_checkpoint_path = os.path.join(os.path.split(checkpoint_path)[0], "quantized_checkpoint.pth")

                                logger.append_log(quantized_checkpoint_path, bit_width, quantized_model, train_loss, test_loss, model_name, dataset_name, reg_name, param, train_accuracy, test_accuracy, train_kl=train_kl, test_kl=test_kl, reconstruction_loss=reconstruction_loss, lr=None, device=str(device), batch_size=batch_size, seed=seed, pretrained=pretrained)
                    else:  # Regularizations without parameters
                        # Set flags for batch_norm and layer_norm based on reg_name
                        checkpoint_paths = logger.get_checkpoint(model_name, dataset_name, reg_name, None)
                        for checkpoint_path in tqdm(checkpoint_paths, desc='Checkpoints', leave=False):
                            use_batch_norm = reg_name == 'batch_norm'
                            use_layer_norm = reg_name == 'layer_norm'

                            model = create_model(model_name, mini=False, dropout_rate=0,num_classes=num_classes,
                                        use_batch_norm=use_batch_norm,
                                        use_layer_norm=use_layer_norm)
                            
                            state_dict = torch.load(checkpoint_path)
                            model.load_state_dict(state_dict)

                            model.to(device)

                            quantized_model, train_loss, test_loss, train_accuracy, test_accuracy, train_kl, test_kl, reconstruction_loss = quantize_model(model, dataset_name, bit_width=bit_width, device=device, batch_size=batch_size, verbose=True)
                            quantized_checkpoint_path = os.path.join(os.path.split(checkpoint_path)[0], "quantized_checkpoint.pth")

                            logger.append_log(quantized_checkpoint_path, bit_width, quantized_model, train_loss, test_loss, model_name, dataset_name, reg_name, None, train_accuracy, test_accuracy, train_kl=train_kl, test_kl=test_kl, reconstruction_loss=reconstruction_loss, lr=None, device=str(device), batch_size=batch_size, seed=seed, pretrained=pretrained)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiments with specified parameters.')
    parser.add_argument("--regularization_type", choices=['l1', 'l2'], required=True,
                        help="Type of regularization (l1, l2")
    parser.add_argument('--bit_width', default=2, type=int, help='Bit width (default: 2)')
    # parser.add_argument('--path', required=True, type=str, help='Path the dir')
    args = parser.parse_args()
    # print(f'running bit_width={args.bit_width}, reg_type={args.regularization_type}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")
    seed = 42
    set_seed(seed)

    # Define your epochs, learning rate, dataset names, model names, and regularizations here
    # epochs = 150
    batch_size = 128
    # lr = 0.001
    DATASETS = ['CIFAR-10', 'FashionNIST'] #['CIFAR-10', 'MNIST', 'IMAGENET', 'FASHIONMNIST']
    MODELS = ['resnet18', 'lenet']# 'alexnet', 'resnet18']
    COMPATIBLE_MODELS = {
        # Define which models are compatible with which datasets
        # 'CIFAR-10': ['lenet'],
        'CIFAR-10': ['resnet18'],
        # 'FashionMNIST': ['lenet'],
        #'CIFAR-10': ['resnet18']
        # etc.
    }
    REGULARIZATIONS = {
        'l1': [0.00071, 0.00133, 0.00194, 0.00255, 0.00316, 0.00377, 0.005],
        'l2': [0.00713, 0.01325, 0.01938, 0.0255, 0.03163, 0.03775, 0.04388, 0.05] #, 0.001
    }

    # ########## this activates fahsionmnist + lenet; if commented then resnet + cifar is run
    # DATASETS = ['FashionMNIST']
    # MODELS = ['lenet']
    # COMPATIBLE_MODELS = {'FashionMNIST': ['lenet']}
    # REGULARIZATIONS_FASHIONMNIST_LENET = {
    #     'l1': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001,  0.005],
    #     'l2': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
    # }
    # REGULARIZATIONS = REGULARIZATIONS_FASHIONMNIST_LENET
    # #########
    REGULARIZATIONS = {args.regularization_type: REGULARIZATIONS[args.regularization_type]}

    BIT_WIDTH = {
        args.bit_width
        # 2,
        # 4,
        # 8,
        # 16
    }

    print(REGULARIZATIONS)
    print(BIT_WIDTH)
    #
    logger = Logger()
    run_experiments(device, pretrained=False)