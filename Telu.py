import torch
import torch.nn as nn

class TeLU(nn.Module):
    """
    TeLU (Tanh Exponential Linear Unit) activation function.

    This activation function is defined as:
        TeLU(x) = x * tanh(exp(x))

    It combines a linear term (x) with a non-linear term (tanh(exp(x))) to introduce
    non-linearity while potentially mitigating the vanishing gradient problem.

    Attributes:
        None (inherits from nn.Module)
    """
    def __init__(self):
        """
        Initializes the TeLU module.

        This method calls the constructor of the parent class (nn.Module) to ensure
        proper initialization.

        Args:
            None
        """
        super().__init__()

    def forward(self, x):
        """
        Forward pass of the TeLU activation function.

        This method applies the TeLU function to the input tensor x.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the TeLU function.
        """
        return x * torch.tanh(torch.exp(x))
