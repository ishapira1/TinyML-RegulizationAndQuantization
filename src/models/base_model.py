#@ ishapira 20231103
"""
defines an abstract base class
It outlines the expected interface for models, such as methods for forward passes,
saving and loading weights, and device handling.
"""

import torch
import torch.nn as nn
import os

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, x):
        """
        Forward pass logic for the model.
        :param x: Input data.
        :return: Model output.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def save_checkpoint(self, file_path):
        """
        Save model checkpoints.
        :param file_path: Path to the file where to store the state dict.
        """
        torch.save(self.state_dict(), file_path)
        print(f"Checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path):
        """
        Load model checkpoints.
        :param file_path: Path to the file from where to load the state dict.
        """
        if os.path.isfile(file_path):
            self.load_state_dict(torch.load(file_path))
            print(f"Checkpoint loaded from {file_path}")
        else:
            raise ValueError(f"No checkpoint found at {file_path}")

    def to_device(self, device):
        """
        Move the model to a specified device.
        :param device: Device to move the model to.
        """
        self.to(device)
        print(f"Model moved to {device}")


if __name__ == "__main__":
    # Example instantiation of the BaseModel - this will not work directly
    # because the forward method is not implemented and BaseModel is designed
    # to be an abstract class.
    try:
        model = BaseModel()
        dummy_input = torch.rand(1, 3, 224, 224)  # Example input tensor shape
        output = model(dummy_input)
    except NotImplementedError as e:
        print("Expected error:", e)
