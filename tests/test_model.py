from config import ROOT_DIR
from solution.dataset_loader import ImageDataset, image_transform
from solution.model import Encoder, Decoder


def test_encoder(random_tensor):
    image = random_tensor(128, 128)
    assert image.shape == (3, 128, 128)

    encoder = Encoder()
    result = encoder(image)

    assert result.shape == (16, 16, 16)


def test_decoder(random_tensor):
    image = random_tensor(w=16, h=16, c=16)

    decoder = Decoder()
    result = decoder(image)

    assert result.shape == (3, 128, 128)
