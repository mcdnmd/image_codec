import torch
import torch.nn as nn


class StrideEncoder(nn.Module):
    def __init__(self):
        super(StrideEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, padding=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 32, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        x = self.layers(x)
        return x


class StrideDecoder(nn.Module):
    def __init__(self):
        super(StrideDecoder, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 128, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 3, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x


class StrideAutoEncoder(nn.Module):
    """Model without max polling"""

    def __init__(self):
        super(StrideAutoEncoder, self).__init__()

        self.encoder = StrideEncoder()
        self.decoder = StrideDecoder()

    def forward(self, x: torch.Tensor, b_t: int | None = None) -> torch.Tensor:
        x = self.encoder(x)
        if self.training and b_t is not None:
            max_val = x.max() / (2 ** (b_t + 1))
            noise = torch.rand_like(x) * max_val
            x = x + noise
        x = self.decoder(x)
        return x
