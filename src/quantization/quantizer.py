import torch
import pact
from ptq_common import quantize_model, calibrate


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
        self.bit_width = bit_width
        self.regularization = regularization if regularization else {}
        self.verbose = verbose

    def _compute_accuracy(self, dataloader):
        """
        Compute the accuracy given the output of the model and the labels.

        :param outputs: The logits output from the model.
        :param labels: The true labels.
        :return: The accuracy as a percentage.
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return accuracy
    
    def _apply_quantization(self, model):
        # if "dynamic" in self.quantization_method:
        #     quantized_model = quantize_dynamic(model, layers_to_quantize=[torch.nn.Linear, torch.nn.Conv2d])
        # elif "static" in self.quantization_method:
        #     quantized_model = quantzize_static(model, self.dataloaders['train']) ##FIXME figure out right fusion
        # else:
        #     print("Quantization Method Not Recognized. Returning original Model")
        #     quantized_model = model
        quantized_model = quantize_model(
            model,
            backend="generic",
            act_bit_width=self.bit_width,
            weight_bit_width=self.bit_width,
            bias_bit_width="int16",
            scaling_per_output_channel=False,
            act_quant_percentile=99.99,
            act_quant_type='symmetric',
            scale_factor_type='float32',
            weight_narrow_range=False
        )
        
        calibrate(self.dataloaders["train"], quantized_model, bias_corr=True)

        return quantized_model

    def quantize(self):
        """
        Quantize model, and compute train/test loss, train/test accuracy
        """
        self.quantized_model = self._apply_quantization(self.model)

        self.quantized_model.eval()

        test_running_loss = 0.0
        train_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.dataloaders['test']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_running_loss += loss.item() * inputs.size(0)
            for inputs, labels in self.dataloaders['train']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                train_running_loss += loss.item() * inputs.size(0)

        test_loss = test_running_loss / len(self.dataloaders['test'].dataset)
        train_loss = train_running_loss / len(self.dataloaders['train'].dataset)

        train_accuracy = self._compute_accuracy(self.dataloaders['train'])
        test_accuracy = self._compute_accuracy(self.dataloaders['test'])
        return self.quantized_model, train_loss, test_loss,train_accuracy, test_accuracy    


def quantize_dynamic(model, layers_to_quantize = [torch.nn.Linear]):
    """
    Takes in a model, dynamically quantizes the model, computes loss

    Note dynamic quantization means that the weights are quantized to int8 statically, and activations are dynamically quantized during inference (estimation of scale happens during inference)

    Dynamic quantization docs:
    https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html
    """
    # print("Dynamic Quantization. Quantizing the following modules: \n{}".format(layers_to_quantize))

    model_int8 = torch.ao.quantization.quantize_dynamic(
        model,  # the original model
        set(layers_to_quantize),  # a set of layers to dynamically quantize
        dtype=torch.qint8
    )

    return model_int8

def quantzize_static(model, dataloader, fuse_modules = None):
    """
    Takes in a model, dataloader, and which modules to fuse, and statically quantizes the model, computes loss

    Note static quantization means that the weights and activations are quantized to int8. Requires a representative dataset to estimate range of activations over

    Also, requires fusion of activations and preceding layers, which will change depending on the activations we use.

    Static quantization docs:
    TODO
    """
    # print("Static Quantization. Fusing the following modules: \n{}".format(fuse_modules))

    model.eval()

    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')

    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    if fuse_modules is not None:
        model_fused = torch.ao.quantization.fuse_modules(model, fuse_modules)
    else:
        model_fused = model
    model_prepared = torch.ao.quantization.prepare(model_fused)

    for data, _ in dataloader:
        model_prepared(data)

    model_int8 = torch.ao.quantization.convert(model_prepared)

    return model_int8
