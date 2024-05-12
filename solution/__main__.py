import torch

from solution.model import AutoEncoder
from solution.pipeline import training_pipeline


if __name__ == '__main__':
    w, h = 128, 128
    b = 3
    device = 'mps' if torch.backends.mps.is_built() else 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoEncoder().to(device)

    training_pipeline(
        model=model,
        device=device,
        b=b,
        w=w,
        h=h,
    )

