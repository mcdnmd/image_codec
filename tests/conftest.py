import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from solution import config


@pytest.fixture
def random_tensor():
    def wrap(w: int, h: int, c: int = 3) -> torch.Tensor:
        return torch.randn((c, w, h))

    return wrap


@pytest.fixture(scope="session")
def first_image() -> Image:
    return Image.open(config.DATASET_DIR / "test/1.png")


@pytest.fixture(scope="session")
def dataset(first_image, to_tensor) -> Dataset:
    class DataSet(Dataset):
        items = [first_image]

        def __getitem__(self, index):
            return to_tensor(self.items[index])

        def __len__(self):
            return len(self.items)

    return DataSet()


@pytest.fixture(scope="session")
def to_tensor() -> transforms.ToTensor:
    return transforms.ToTensor()


@pytest.fixture(scope="session")
def to_pil_image() -> transforms.ToPILImage:
    return transforms.ToPILImage()


@pytest.fixture(scope="session")
def device() -> str:
    return (
        "mps"
        if torch.backends.mps.is_built()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
