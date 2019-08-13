import torch
import torch.nn.functional as F


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)


def compute_maximum_mean_discrepancy(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


def loss_function(x, x_reconstructed, true_samples, z):
    nnl = (x_reconstructed - x).pow(2).mean()
    mmd = compute_maximum_mean_discrepancy(true_samples, z)
    loss = nnl + mmd
    return {'loss': loss, 'Negative-Loglikelihood': nnl, 'Maximum_Mean_Discrepancy': mmd}