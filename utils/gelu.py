import torch.nn as nn
import torch
import numpy as np


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit for use in PyTorch.

    Original paper: https://arxiv.org/abs/1606.08415

    """

    def forward(self, x):
        """
        Application of the non-linearity.

        Parameters
        ----------
        x: torch.Tensor
            To perform the activation on.

        Returns
        -------
        `x` with the GELU activation applied.

        """
        cdf = 0.5 * (1.0 * torch.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))

        return cdf * x
