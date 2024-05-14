import pytest
from torch.utils.data import DataLoader

from solution.models.base_model import Encoder, Decoder, AutoEncoder
from solution.models.some_model import BaseAutoEncoder
from solution.utils import process_images


def test_encoder(first_image, to_tensor):
    first_image_tensor = to_tensor(first_image)
    assert first_image_tensor.shape == (3, 128, 128)

    encoder = Encoder()
    result = encoder(first_image_tensor)

    assert result.shape == (16, 16, 16)


def test_decoder(random_tensor):
    image = random_tensor(w=16, h=16, c=16)

    decoder = Decoder()
    result = decoder(image)

    assert result.shape == (3, 128, 128)


def test_auto_encoder(first_image, to_tensor):
    first_image_tensor = to_tensor(first_image)
    auto_encoder = AutoEncoder()

    result = auto_encoder(first_image_tensor)

    assert result.shape == (3, 128, 128)


@pytest.mark.skip
def test_process_image(first_image, to_tensor, device, dataset, to_pil_image):
    first_image.show()
    auto_encoder = BaseAutoEncoder().to(device)

    imgs_decoded, imgsQ_decoded, bpp = process_images(
        model=auto_encoder,
        test_dataloader=DataLoader(dataset),
        device=device,
        b=2,
        w=128,
        h=128,
    )
    print("X" * 20, bpp)

    image1 = to_pil_image(imgs_decoded[0])
    image2 = to_pil_image(imgsQ_decoded[0])
    image1.show()
    image2.show()
