import torch
from src.data_loaders.dataset_loader import load_dataset
from src.models.model_registry import create_model
from src.training.trainer import Trainer
from src.logs.logger import Logger, RESULTS_FILE_NAME_IN_LOGS,CHECKPOINT_FILE_NAME_IN_LOGS
from src.quantization.quantizer import Quantizer
from src.logs.parser import load_json
from tqdm import tqdm
import os
import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

import argparse




def process(model_name, dataset, batch_size, device, bit_width, checkpoint_path,regularization, exp_dir, regularization_param):
    train_loader = load_dataset(dataset_name=dataset, batch_size=batch_size, train=True)
    val_loader = load_dataset(dataset_name=dataset, batch_size=batch_size, train=False)
    dataloaders = {
        'train': train_loader,
        'test': val_loader,
        'calibration':train_loader
    }
    try:
        model = create_model(model_name)
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
    except:
        model = create_model(model_name, use_batch_norm=False)
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)       

    # Load the weights into your model
    

    model.to(device)

    print("1")
    quantizer = Quantizer(model, dataloaders, torch.nn.CrossEntropyLoss(), device, bit_width=bit_width)
    quantized_model, train_loss, test_loss,train_accuracy, test_accuracy = quantizer.quantize()

    logger = Logger()
    try:
        (reg_name, rag_parm), = regularization.items()
    except:
            reg_name = regularization
            rag_parm = regularization_param


    logger.append_log(quantization_method=f"PTQ_bit_{bit_width}",bit_width=bit_width,
                      model = quantized_model, train_loss=train_loss, test_loss=test_loss,train_accuracy=train_accuracy, test_accuracy=test_accuracy,model_name=model_name, 
                      reg_name=reg_name, param=rag_parm, dataset_name=dataset, exp_dir=exp_dir)
    print("logger.append_log( - DONE")
    return model


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run experiments with specified parameters.')
    parser.add_argument('--bit_width', default=2, type=int, help='Bit width (default: 2)')
    parser.add_argument('--path', required=True, type=str, help='Path the dir')
    args = parser.parse_args()


    instance = load_json(os.path.join(args.path, RESULTS_FILE_NAME_IN_LOGS))    
    # Check for CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    process(instance["model_name"], instance["dataset_name"], instance["batch_size"], device, args.bit_width, 
            os.path.join(args.path, CHECKPOINT_FILE_NAME_IN_LOGS), instance["regularization"], args.path, instance["regularization_param"])

if __name__ == '__main__':
    main()