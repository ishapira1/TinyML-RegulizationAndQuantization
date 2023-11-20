import torch
from src.data_loaders.dataset_loader import load_dataset
from src.models.model_registry import create_model
from src.training.trainer import Trainer
from src.logs.logger import Logger
from src.quantization.quantizer import Quantizer
from tqdm import tqdm
from torch.optim import Adam
import os

QUANTIZATION_METHODS = {
    'dynamic_quantization',
    'static_quantization'
}


def set_seed(seed):
    """
    Set the seed for reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def quantize_model(model, dataset, device,  quantization_method=None, batch_size=128, verbose = True):
    """
    Train the model and return the results.
    """
    train_loader = load_dataset(dataset_name=dataset, batch_size=batch_size, train=True)
    val_loader = load_dataset(dataset_name=dataset, batch_size=batch_size, train=False)
    dataloaders = {
        'train': train_loader,
        'test': val_loader
    }
    quantizer = Quantizer(model, dataloaders, criterion = torch.nn.CrossEntropyLoss(), device=device, quantization_method=quantization_method, verbose=verbose)

    return quantizer.quantize()

def run_experiments(device, num_classes=10, pretrained=False):
    """
    Run experiments across different datasets, models, and regularizations.
    """
    for quantization_method in tqdm(QUANTIZATION_METHODS, desc='Quantization Methods', leave=False):
        for dataset_name in tqdm(DATASETS, desc='Datasets', leave=False):
            for model_name in tqdm(MODELS, desc='Models', leave=False):
                # Compatibility check
                if model_name not in COMPATIBLE_MODELS[dataset_name]:
                    continue
                print(model_name)

                for reg_name, reg_params in tqdm(REGULARIZATIONS.items(), desc='Regularizations', leave=False):
                    if reg_params:  # Regularizations with parameters
                        for param in tqdm(reg_params, desc='Params', leave=False):
                            checkpoint_paths = logger.get_checkpoint(model_name, dataset_name, reg_name, param)
                            for checkpoint_path in tqdm(checkpoint_paths, desc='Checkpoints', leave=False):
                                dropout_rate = param if reg_name == 'dropout' else 0


                                model = create_model(model_name, num_classes=num_classes,mini=False, dropout_rate=dropout_rate,
                                            use_batch_norm=False,
                                            use_layer_norm=False)
                                
                                state_dict = torch.load(checkpoint_path)
                                model.load_state_dict(state_dict)

                                model.to(device)

                                quantized_model, train_loss, test_loss, train_accuracy, test_accuracy = quantize_model(model, dataset_name,  quantization_method=quantization_method, device=device, batch_size=batch_size, verbose=True)

                                quantized_checkpoint_path = os.path.join(os.path.split(checkpoint_path)[0], "quantized_checkpoint.pth")

                                logger.append_log(quantized_checkpoint_path, quantization_method, quantized_model, train_loss, test_loss, model_name, dataset_name, reg_name, param, train_accuracy, test_accuracy, lr=lr, device=str(device), batch_size=batch_size, seed=seed, pretrained=pretrained)
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
                            print(checkpoint_path)
                            model.load_state_dict(state_dict)

                            model.to(device)
                            quantized_model, train_loss, test_loss, train_accuracy, test_accuracy = quantize_model(model, dataset_name, quantization_method=quantization_method, device=device, batch_size=batch_size, verbose=True)
                            quantized_checkpoint_path = os.path.join(os.path.split(checkpoint_path)[0], "quantized_checkpoint.pth")

                            logger.append_log(quantized_checkpoint_path, quantization_method, quantized_model, train_loss, test_loss, model_name, dataset_name, reg_name, None, train_accuracy, test_accuracy, lr=lr, device=str(device), batch_size=batch_size, seed=seed, pretrained=pretrained)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")
    seed = 42
    set_seed(seed)

    # Define your epochs, learning rate, dataset names, model names, and regularizations here
    epochs = 2
    batch_size = 256
    lr = 0.001
    DATASETS = ['MNIST'] #['CIFAR-10', 'MNIST', 'IMAGENET', 'FASHIONMNIST']
    MODELS = ['lenet']# 'alexnet', 'resnet18']
    COMPATIBLE_MODELS = {
        # Define which models are compatible with which datasets
        # 'CIFAR-10': ['lenet'],
        'MNIST': ['lenet'],
        # 'FASHIONMNIST': ['lenet'],
        #'CIFAR-10': ['resnet18']
        # etc.
    }
    REGULARIZATIONS = {
        'none': None,  # No regularization parameters needed for baseline
        'batch_norm': None,  # Batch normalization typically does not require explicit parameters
        #'layer_norm': None,  # Layer normalization also typically does not require explicit parameters
        'dropout': [0.3, 0.5, 0.7],  # Different dropout rates to experiment with
        'l1':[
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.0002,
    0.0003,
    0.0004,
    0.0006,
    0.0007,
    0.0008,
    0.0009,
],
        'l2':[
    0.05,
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.0002,
    0.0003,
    0.0004,
    0.0006,
    0.0007,
    0.0008,
    0.0009,
    0.003
],
        'l_infinty': [0.001, 0.01 , 0.1  ]  # Different L-infinity regularization strengths
    }

    QUANTIZATION_METHODS = {
        'dynamic_quantization',
        'static_quantization'
    }
    #
    logger = Logger()
    run_experiments(device, pretrained=False)