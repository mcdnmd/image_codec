from typing import Tuple, Any

import torch
from torch import nn, Tensor

from solution.models.base_model import BasePytorchModel


class VAE(BasePytorchModel):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__(model_name=f"vae_lat-dim-{latent_dim}")

        self.latent_dim = latent_dim

        self._encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.ReLU(),
            nn.Flatten()
        )

        # Latent space
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 512 * 4 * 4)

        self._decoder = nn.Sequential(
            nn.Unflatten(1, (512, 4, 4)),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.Sigmoid()
        )

    def encoder(self, x: torch.Tensor):
        h = self._encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z)
        return self._decoder(h)

    def forward(self, x: torch.Tensor, b_t: int | None = None) -> Tensor | tuple[Tensor, Any, Any]:
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        if self.training and b_t is not None:
            max_val = z.max() / (2 ** (b_t + 1))
            noise = torch.rand_like(z) * max_val
            z = z + noise

        return self.decoder(z), mu, logvar