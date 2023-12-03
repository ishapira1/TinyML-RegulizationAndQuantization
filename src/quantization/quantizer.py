import torch
from src.quantization.ptq_common import quantize_model, calibrate
from brevitas.graph.quantize import preprocess_for_quantize
from brevitas.nn import QuantConv2d, QuantLinear
# from torch.nn.functional import kl_div

class Quantizer:
    def __init__(self, model, dataloaders, criterion, device, bit_width=8, regularization=None, verbose=True):
        """
        Initialize the Trainer. 

        :param model: The neural network model.
        :param dataloaders: A dictionary containing the 'train' and 'test' DataLoader.
        :param device: The device to run the training on.
        :param quantization_method: A dict with key FIXME
        :param verbose: If True, print detailed information during training.
        """
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.device = device
        print(f"self.device = {device}")
        self.bit_width = bit_width
        self.bit_width = bit_width
        self.regularization = regularization if regularization else {}
        self.quantized_model = None

    def _compute_quantized_stats(self, dataloader):
        self.quantized_model.eval()
        self.model.eval()
        
        correct, total = 0, 0
        running_loss, running_kl = 0., 0.
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                quantized_outputs = self.quantized_model(inputs)
                _, predicted = torch.max(quantized_outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += self.criterion(quantized_outputs, labels)
                running_kl += self._kl_divergence(torch.softmax(quantized_outputs, dim=1), torch.softmax(outputs, dim=1)).sum()
        accuracy = 100 * correct / total
        running_loss /= total
        running_kl /= total
        return accuracy, running_loss.item(), running_kl.item()
    
    def _apply_quantization(self, model):
        preprocessed_model = preprocess_for_quantize(
            model,
            equalize_iters=0,
            equalize_merge_bias=False
        )


        quantized_model = quantize_model(
            preprocessed_model,
            backend="generic",
            act_bit_width=self.bit_width,
            weight_bit_width=self.bit_width,
            bias_bit_width="int32",
            scaling_per_output_channel=False,
            act_quant_percentile=99.99,
            act_quant_type='asymmetric',
            scale_factor_type='float32',
            weight_narrow_range=False
        )

        quantized_model.to(self.device)

        calibrate(self.dataloaders["calibration"], quantized_model, bias_corr=True)
        return quantized_model

    def quantize(self):
        """
        Quantize model, and compute train/test loss, train/test accuracy
        """
        self.quantized_model = self._apply_quantization(self.model).to(self.device)

        train_accuracy, train_loss, train_kl = self._compute_quantized_stats(self.dataloaders['train'])
        test_accuracy, test_loss, test_kl = self._compute_quantized_stats(self.dataloaders['test'])

        reconstruction_loss = self._weight_space_l2_distance(self.model, self.quantized_model)
        print(f'train_acc={train_accuracy}, test_acc={test_accuracy}\ntrain_loss={train_loss}, test_loss={test_loss}\ntrain_kl={train_kl}, test_kl={test_kl}\nreconstruction_loss={reconstruction_loss}')
        return self.quantized_model, train_loss, test_loss, train_accuracy, test_accuracy, train_kl, test_kl, reconstruction_loss

    @staticmethod
    def _weight_space_l2_distance(model, quantized_model):
        """ norm of the distance of the weight vectors as a percentage of the overall norm of
        the weight vector corresponding to the unquantized model """
        quantized_named_tensors = []
        for module_name, module in quantized_model.named_modules():
            if isinstance(module, (QuantConv2d, QuantLinear)):
                quantized_named_tensors.append((module_name, module.quant_weight().value))
        
        overall_param_count, d2, overall_norm = 0, 0., 0.
        for name_param, quantized_name_param in zip(model.named_parameters(), quantized_named_tensors):
            name, param = name_param
            quantized_name, quantized_param = quantized_name_param
            overall_param_count += param.numel()
            d2 += ((param - quantized_param) ** 2).sum()
            overall_norm += (param ** 2).sum()
            assert param.size() == quantized_param.size() and name == quantized_name + '.weight'
        return (d2 / overall_norm * 100.).item()

    @staticmethod
    def _kl_divergence(p, q):
        return torch.sum(p * (torch.log(p) - torch.log(q)), dim=1)