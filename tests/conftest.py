import pytest
import torch


@pytest.fixture
def random_tensor():
    def wrap(w: int, h: int, c: int = 3) -> torch.Tensor:
        return torch.randn((c, w, h))

    return wrap
