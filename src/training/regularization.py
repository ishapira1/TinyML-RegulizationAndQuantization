#@ ishapira 20231103
"""
defines various regularization methods that can be applied **during** training
"""

import torch

def l1_regularization(model, loss, lambda_term):
    """
    Apply L1 regularization to the model.
    """
    l1_loss = sum(param.abs().sum() for param in model.parameters())
    loss = loss + lambda_term * l1_loss
    return loss

def l2_regularization(model, loss, lambda_term):
    """
    this can be directly applied through optimizer in PyTorch using
    weight decay parameter.
    """
    l2_loss = sum(param.pow(2.0).sum() for param in model.parameters())
    loss = loss + lambda_term * l2_loss
    return loss

def l_infinity_regularization(model, loss, lambda_term, epsilon):
    """
    Apply L-infinity regularization to the model.
    """
    linf_loss = max(param.abs().max() for param in model.parameters())
    loss = loss + lambda_term * (linf_loss - epsilon).clamp(min=0)
    return loss

# Example usage
if __name__ == "__main__":
    # Assuming we have a model and a loss value from somewhere in our training loop:
    # model = ...  # The neural network model
    # loss = ...   # The current loss value without regularization
    # lambda_l1 = 1e-5  # Example L1 regularization strength
    # lambda_l2 = 1e-4  # Example L2 regularization strength
    # lambda_linf = 1e-5  # Example L-infinity regularization strength
    # epsilon_linf = 1e-3  # Epsilon for L-infinity norm
    #
    # # Apply L1 regularization
    # loss = l1_regularization(model, loss, lambda_l1)
    #
    # # Apply L2 regularization (or use weight decay in optimizer)
    # loss = l2_regularization(model, loss, lambda_l2)
    #
    # # Apply L-infinity regularization
    # loss = l_infinity_regularization(model, loss, lambda_linf, epsilon_linf)

    # Continue with the training process
