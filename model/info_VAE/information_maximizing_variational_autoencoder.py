import torch
import torch.nn as nn

from utils import initialize_weights
from utils.gelu import GELU


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, channels, height, width):
        super(UnFlatten, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)


class InferenceNetwork(nn.Module):
    def __init__(self, params):
        super(InferenceNetwork, self).__init__()
        self.params = params

        inference_layers = [
            nn.Conv2d(in_channels=self.params.channels[0],
                      out_channels=self.params.channels[1],
                      kernel_size=self.params.kernel_size[0],
                      stride=self.params.stride[0],),
            GELU(),
            nn.BatchNorm2d(self.params.channels[1]),
            nn.Conv2d(in_channels=self.params.channels[1],
                      out_channels=self.params.channels[2],
                      kernel_size=self.params.kernel_size[1],
                      stride=self.params.stride[1]),
            GELU(),
            nn.BatchNorm2d(self.params.channels[2]),
            nn.Conv2d(in_channels=self.params.channels[2],
                      out_channels=self.params.channels[3],
                      kernel_size=self.params.kernel_size[2],
                      stride=self.params.stride[2]),
            GELU(),
            nn.BatchNorm2d(self.params.channels[3]),
            Flatten(),
        ]

        self.encoder = nn.Sequential(*inference_layers)

        self.activation_fn = GELU()
        self.fc_latent = nn.Linear(in_features=self.params.hidden_dim, out_features=self.params.latent_dim)

        initialize_weights(self)

    def forward(self, x):
        h1 = self.encoder(x)
        z = self.activation_fn(self.fc_latent(h1))
        return z


class GenerativeNetwork(nn.Module):
    def __init__(self, params, height=4, width=4):
        super(GenerativeNetwork, self).__init__()
        self.params = params

        generative_layers = [
            nn.Linear(in_features=self.params.latent_dim, out_features=self.params.hidden_dim),
            GELU(),
            UnFlatten(self.params.channels[3], height, width),
            nn.ConvTranspose2d(in_channels=self.params.channels[3],
                               out_channels=self.params.channels[4],
                               kernel_size=self.params.kernel_size[3],
                               stride=self.params.stride[3]),
            GELU(),
            nn.BatchNorm2d(self.params.channels[4]),
            nn.ConvTranspose2d(in_channels=self.params.channels[4],
                               out_channels=self.params.channels[5],
                               kernel_size=self.params.kernel_size[4],
                               stride=self.params.stride[4]),
            GELU(),
            nn.BatchNorm2d(self.params.channels[5]),
            nn.ConvTranspose2d(in_channels=self.params.channels[5],
                               out_channels=self.params.channels[6],
                               kernel_size=self.params.kernel_size[5],
                               stride=self.params.stride[5]),
            nn.Sigmoid(),
        ]

        self.decoder = nn.Sequential(*generative_layers)

        initialize_weights(self)

    def forward(self, z):
        x_reconstructed = self.decoder(z)
        return x_reconstructed


class VariationalAutoencoder(nn.Module):
    def __init__(self, params):
        super(VariationalAutoencoder, self).__init__()
        self.params = params
        self.inference_network = InferenceNetwork(params)
        self.generative_network = GenerativeNetwork(params)

    def sample(self, eps=None):
        if eps is None:
            eps = torch.randn(torch.Size([1, self.params.hidden_dim]))
        return self.decode(eps)

    def encode(self, x):
        z = self.inference_network(x)
        return z

    def decode(self, z):
        return self.generative_network(z)

    def forward(self, x):
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z

