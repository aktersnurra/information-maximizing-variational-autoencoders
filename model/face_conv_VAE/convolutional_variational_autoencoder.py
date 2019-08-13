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
                      kernel_size=(self.params.kernel_size[0], self.params.kernel_size[1]),
                      stride=self.params.stride[0],),
            GELU(),
            nn.BatchNorm2d(self.params.channels[1]),
            nn.Conv2d(in_channels=self.params.channels[1],
                      out_channels=self.params.channels[2],
                      kernel_size=(self.params.kernel_size[2], self.params.kernel_size[3]),
                      stride=self.params.stride[1]),
            GELU(),
            nn.BatchNorm2d(self.params.channels[2]),
            nn.Conv2d(in_channels=self.params.channels[2],
                      out_channels=self.params.channels[3],
                      kernel_size=(self.params.kernel_size[4], self.params.kernel_size[5]),
                      stride=self.params.stride[2]),
            GELU(),
            nn.BatchNorm2d(self.params.channels[3]),
            Flatten(),
        ]

        self.encoder = nn.Sequential(*inference_layers)

        self.activation_fn = GELU()
        self.fc_mu = nn.Linear(in_features=self.params.hidden_dim, out_features=self.params.latent_dim)
        self.fc_logvar = nn.Linear(in_features=self.params.hidden_dim, out_features=self.params.latent_dim)

        initialize_weights(self)

    def forward(self, x):
        h1 = self.encoder(x)
        mu = self.activation_fn(self.fc_mu(h1))
        logvar = self.activation_fn(self.fc_logvar(h1))

        return mu, logvar


class GenerativeNetwork(nn.Module):
    def __init__(self, params):
        super(GenerativeNetwork, self).__init__()
        self.params = params

        generative_layers = [
            nn.Linear(in_features=self.params.latent_dim, out_features=self.params.hidden_dim),
            GELU(),
            UnFlatten(self.params.channels[3], self.params.hidden_height, self.params.hidden_width),
            nn.ConvTranspose2d(in_channels=self.params.channels[3],
                               out_channels=self.params.channels[4],
                               kernel_size=(self.params.kernel_size[6], self.params.kernel_size[7]),
                               stride=self.params.stride[3]),
            GELU(),
            nn.BatchNorm2d(self.params.channels[4]),
            nn.ConvTranspose2d(in_channels=self.params.channels[4],
                               out_channels=self.params.channels[5],
                               kernel_size=(self.params.kernel_size[8], self.params.kernel_size[9]),
                               stride=self.params.stride[4]),
            GELU(),
            nn.BatchNorm2d(self.params.channels[5]),
            nn.ConvTranspose2d(in_channels=self.params.channels[5],
                               out_channels=self.params.channels[6],
                               kernel_size=(self.params.kernel_size[10], self.params.kernel_size[11]),
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

    @staticmethod
    def reparameterization(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def encode(self, x):
        mu, logvar = self.inference_network(x)
        return mu, logvar

    def decode(self, z):
        return self.generative_network(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterization(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar, z

