import torch

def quantize_dynamic(model, layers_to_quantize = [torch.nn.Linear]):
    """
    Takes in a model, dynamically quantizes the model, computes loss

    Note dynamic quantization means that the weights are quantized to int8 statically, and activations are dynamically quantized during inference (estimation of scale happens during inference)

    Dynamic quantization docs:
    https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html
    """
    print("Dynamic Quantization. Quantizing the following modules: \n{}".format(layers_to_quantize))

    model_int8 = torch.ao.quantization.quantize_dynamic(
        model,  # the original model
        set(layers_to_quantize),  # a set of layers to dynamically quantize
        dtype=torch.qint8
    )

    model_int8(torch.randn(1, 784))
    
    return model_int8

def quantzize_static(model, dataloader, fuse_modules = [['conv', 'relu']]):
    """
    Takes in a model, dataloader, and which modules to fuse, and statically quantizes the model, computes loss

    Note static quantization means that the weights and activations are quantized to int8. Requires a representative dataset to estimate range of activations over

    Also, requires fusion of activations and preceding layers, which will change depending on the activations we use.

    Dynamic quantization docs:
    https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html
    """
    print("Static Quantization. Fusing the following modules: \n{}".format(fuse_modules))

    model.eval()

    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')

    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    model_fused = torch.ao.quantization.fuse_modules(model, fuse_modules)
    model_prepared = torch.ao.quantization.prepare(model_fused)

    for data, _ in dataloader:
        model_prepared(data)

    model_int8 = torch.ao.quantization.convert(model_prepared)

    return model_int8