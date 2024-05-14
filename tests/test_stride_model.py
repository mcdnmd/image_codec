from solution.models.stride_model import StrideAutoEncoder


def test_auto_encoder(random_tensor):
    image = random_tensor(w=128, h=128, c=3)

    auto_encoder = StrideAutoEncoder()
    result = auto_encoder(image)

    assert result.shape == (3, 128, 128)
