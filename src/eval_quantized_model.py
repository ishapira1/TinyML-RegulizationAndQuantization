import torch
from src.data_loaders.dataset_loader import load_dataset
from src.models.model_registry import create_model
from src.training.trainer import Trainer
from src.logs.logger import Logger
from src.quantization.quantizer import Quantizer
from tqdm.auto import tqdm
from torch.optim import Adam
from brevitas.nn import QuantConv2d, QuantLinear
import os, sys

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
    train_loader = load_dataset(dataset_name=dataset, batch_size=batch_size, train=True, subset=True)
    test_loader = load_dataset(dataset_name=dataset, batch_size=batch_size, train=False, subset=True)
    calibration_loader = load_dataset(dataset_name=dataset, batch_size=batch_size, train=True, subset=True)
    dataloaders = {
        'train': train_loader,
        'test': test_loader,
        'calibration': calibration_loader,
    }
    quantizer = Quantizer(model, dataloaders, criterion = torch.nn.CrossEntropyLoss(), device=device, bit_width=bit_width, verbose=verbose)

    return quantizer.quantize()

def weight_space_l2_distance(model, quantized_model):
    # for name, param in quantized_named_tensors[:10]:
    #     print(name, param.size())
    quantized_named_tensors = []
    for module_name, module in quantized_model.named_modules():
        # if 'conv' in module_name:
        #     print(module_name, type(module_name))
        if isinstance(module, (QuantConv2d, QuantLinear)):
            # print(f'{module_name} of type {type(module).mro()} is a quant shit')
            # scaled_weight = module.quant_weight()
            quantized_named_tensors.append((module_name, module.quant_weight().value))
        # else:
        #     for name, param in module.named_parameters():
        #         if isinstance(param, torch.nn.Parameter):
        #             quantized_named_tensors.append((name, param))

    named_parameters = []
    for name, param in model.named_parameters():
        if 'bias' in name:
            continue
        named_parameters.append((name, param))

    # for name_param, quantized_name_param in list(zip(model.named_parameters(), quantized_model.named_parameters()))[:15]:
    overall_param_count, d2, overall_norm = 0, 0., 0.
    for name_param, quantized_name_param in list(zip(named_parameters, quantized_named_tensors)):
        name, param = name_param
        quantized_name, quantized_param = quantized_name_param
        overall_param_count += param.numel()
        d2 += ((param - quantized_param) ** 2).sum()
        overall_norm += (param ** 2).sum()
        # print(name, quantized_name, param.size(), quantized_param.size(), type(quantized_param).mro())
        assert param.size() == quantized_param.size() and name == quantized_name + '.weight'
    print(f'l2 norm = {d2 / overall_norm * 100.} percent')
    sys.exit()
    return d2

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
                                
                                state_dict = torch.load(checkpoint_path)
                                model.load_state_dict(state_dict)

                                model.to(device)

                                print(f'bit_width {bit_width}:')
                                quantized_model,_,_,_,_,_,_= quantize_model(model, dataset_name,  bit_width=bit_width, device=device, batch_size=batch_size, verbose=True)

                                quantized_state_dict = torch.load(os.path.join(os.path.split(checkpoint_path)[0], f"bit_width_{bit_width}", "model_checkpoint.pth"))

                                quantized_model.load_state_dict(quantized_state_dict)

                                # print(quantized_model.model)
                                # print(quantized_model.model.conv1.quant_weight())
                                # print(model.model.conv1)
                                weight_space_l2_distance(model, quantized_model)

                    else:  # Regularizations without parameters
                        # Set flags for batch_norm and layer_norm based on reg_name
                        checkpoint_paths = logger.get_checkpoint(model_name, dataset_name, reg_name, None)
                        for checkpoint_path in tqdm(checkpoint_paths, desc='Checkpoints', leave=False):
                            use_batch_norm = reg_name == 'batch_norm'
                            use_layer_norm = reg_name == 'layer_norm'

                            model = create_model(model_name, mini=False, dropout_rate=0,num_classes=num_classes,
                                        use_batch_norm=use_batch_norm,
                                        use_layer_norm=use_layer_norm)
                            
                            model.to(device)


                            print(f'bit_width {bit_width}:')
                            quantized_model, _,_,_,_= quantize_model(model, dataset_name, bit_width=bit_width, device=device, batch_size=batch_size, verbose=True)

                            quantized_state_dict = torch.load(os.path.join(os.path.split(checkpoint_path)[0], f"bit_width_{bit_width}", "model_checkpoint.pth"))

                            quantized_model.load_state_dict(quantized_state_dict)

                            # print(quantized_model.model)
                            # print(quantized_model.model.conv1.quant_weight())
                            # print(model.model.conv1)
                            weight_space_l2_distance(model, quantized_model)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")
    seed = 42
    set_seed(seed)

    # Define your epochs, learning rate, dataset names, model names, and regularizations here
    # epochs = 150
    batch_size = 128
    lr = 0.001
    DATASETS = ['CIFAR-10'] #['CIFAR-10', 'MNIST', 'IMAGENET', 'FASHIONMNIST']
    MODELS = ['resnet18']# 'alexnet', 'resnet18']
    COMPATIBLE_MODELS = {
        # Define which models are compatible with which datasets
        # 'CIFAR-10': ['lenet'],
        'CIFAR-10': ['resnet18'],
        # 'FASHIONMNIST': ['lenet'],
        #'CIFAR-10': ['resnet18']
        # etc.
    }
    REGULARIZATIONS = {
        # 'none': None,  # No regularization parameters needed for baseline
        # 'batch_norm': None,  # Batch normalization typically does not require explicit parameters
        # #'layer_norm': None,  # Layer normalization also typically does not require explicit parameters
        # 'dropout': [0.3, 0.5, 0.7],  # Different dropout rates to experiment with
        # 'l1': [
        #     1e-4,
        #     2e-4,
        #     1e-5,
        #     2e-5,
        #     5e-5
        # ],
        'l2':[
            1e-3,
            # 2e-4,
            # 1e-5,
            # 2e-5,
            # 5e-5
        ],
        # 'l_infinty': [0.001, 0.01 , 0.1  ]  # Different L-infinity regularization strengths
    }

    BIT_WIDTH = {
        2,
        4,
        8,
        16
    }

    print(REGULARIZATIONS)
    print(BIT_WIDTH)
    #
    logger = Logger()
    run_experiments(device, pretrained=False)