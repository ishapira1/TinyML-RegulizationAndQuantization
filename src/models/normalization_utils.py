import torch.nn as nn

# Utility function to replace BatchNorm layers with Identity (no normalization)
def replace_bn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.Identity())
        else:
            replace_bn(child)

# Utility function to add LayerNorm after each convolutional layer
def add_ln(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            # Assuming 'child' outputs a tensor with shape [N, C, H, W]
            # where N is the batch dimension that layer norm will not normalize over.
            norm_layer = nn.LayerNorm([child.out_channels, *child.kernel_size])
            # The normalization layer is inserted after the convolutional layer
            setattr(module, name, nn.Sequential(child, norm_layer))
        else:
            add_ln(child)
