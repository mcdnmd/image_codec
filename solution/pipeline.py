import torch
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from CNNImageCodec import EntropyEncoder, EntropyDecoder
from solution.dataset_loader import ImageDataset, image_transform


def pipeline(model: nn.Module, device: str, b, w, h):
    print("Start pipeline")
    test_dataset = ImageDataset(config.DATASET_DIR / "test", image_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    imgs_encoded = []
    imgs_decoded = []

    with torch.no_grad():
        for test_batch in tqdm(test_dataloader):
            test_batch = test_batch.to(device)
            encoded_images = model.encoder(test_batch)
            decoded_images = model.decoder(encoded_images)

            imgs_encoded.append(encoded_images.cpu().detach())
            imgs_decoded.append(decoded_images.cpu().detach())

    imgs_encoded = torch.vstack(imgs_encoded)
    imgs_decoded = torch.vstack(imgs_decoded)

    # Normalize and quantize
    max_encoded_imgs = imgs_encoded.amax(dim=1, keepdim=True)
    norm_imgs_encoded = imgs_encoded / max_encoded_imgs
    quantized_imgs_encoded = (torch.clip(norm_imgs_encoded, 0, 0.9999999) * pow(2, b)).to(
        torch.int32
    )
    quantized_imgs_encoded = quantized_imgs_encoded.numpy()

    # Encode and decode using entropy coding
    quantized_imgs_decoded = []
    bpp = []

    for i in range(quantized_imgs_encoded.shape[0]):
        size_z, size_h, size_w = quantized_imgs_encoded[i].shape
        encoded_bits = EntropyEncoder(quantized_imgs_encoded[i], size_z, size_h, size_w)
        byte_size = len(encoded_bits)
        bpp.append(byte_size * 8 / (w * h))
        quantized_imgs_decoded.append(EntropyDecoder(encoded_bits, size_z, size_h, size_w))
    quantized_imgs_decoded = torch.tensor(np.array(quantized_imgs_decoded, dtype=np.uint8))

    shift = 1.0 / pow(2, b + 1)
    dequantized_imgs_decoded = (quantized_imgs_decoded.to(torch.float32) / pow(2, b)) + shift
    dequantized_denorm_imgs_decoded = dequantized_imgs_decoded * max_encoded_imgs

    imgsQ_decoded = []

    with torch.no_grad():
        for deq_img in dequantized_denorm_imgs_decoded:
            deq_img = deq_img.to(device)
            decoded_imgQ = model.decoder(deq_img)

            imgsQ_decoded.append(decoded_imgQ.cpu().detach())

    imgsQ_decoded = torch.stack(imgsQ_decoded)

    assert imgsQ_decoded.shape == imgs_decoded.shape
    assert imgsQ_decoded.shape[0] == len(bpp)

    return imgs_decoded, imgsQ_decoded, bpp
