from solution.models.stride_model import StrideAutoEncoder


def test_stride_encoder(first_image, to_tensor):
    first_image_tensor = to_tensor(first_image)
    assert first_image_tensor.shape == (3, 128, 128)

    encoder = StrideAutoEncoder().encoder
    result = encoder(first_image_tensor)

    assert result.shape == (16, 16, 16)


def test_stride_decoder(random_tensor):
    image = random_tensor(w=16, h=16, c=16)

    decoder = StrideAutoEncoder().decoder
    result = decoder(image)

    assert result.shape == (3, 128, 128)


def test_stride_auto_encoder(first_image, to_tensor):
    first_image_tensor = to_tensor(first_image)
    auto_encoder = StrideAutoEncoder()

    result = auto_encoder(first_image_tensor)

    assert result.shape == (3, 128, 128)
