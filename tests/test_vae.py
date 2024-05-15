import torch

from solution.models.vae_model import VAE


def test_vae_encoder(first_image, to_tensor):
    first_image_tensor = torch.randn((8, 3, 128, 128))

    model = VAE()
    result = model.encoder(first_image_tensor)

    assert result[0].shape == (8, 128)
    assert result[1].shape == (8, 128)


def test_vae_decoder(random_tensor):
    image = torch.randn((8, 128))

    model = VAE()
    result = model.decoder(image)

    assert len(result) == 8
    assert result[0].shape == (3, 128, 128)


def test_vae(first_image, to_tensor):
    first_image_tensor = torch.randn((8, 3, 128, 128))

    model = VAE()
    result = model(first_image_tensor)

    assert result == (8, 3, 128, 128)
