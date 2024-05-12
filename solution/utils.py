import os

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from solution.entropy_model import entropy_decoder, entropy_encoder

to_pil_transform = transforms.ToPILImage()


def PSNR_RGB(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def PSNR(y_true, y_pred):
    max_pixel = 1.0
    mse = torch.mean(torch.square(y_pred - y_true))
    if mse == 0:
        return float('inf')  # consistent handling of zero MSE
    return (10 * torch.log10(max_pixel ** 2 / mse)).item()


def JPEGRDSingleImage(torch_img, TargetBPP):
    image = to_pil_transform(torch_img)

    width, height = image.size
    realbpp = 0
    realpsnr = 0
    realQ = 0
    for Q in range(101):
        image.save("test.jpeg", "JPEG", quality=Q)
        image_dec = Image.open("test.jpeg")
        bytesize = os.path.getsize("test.jpeg")
        bpp = bytesize * 8 / (width * height)
        psnr = PSNR_RGB(np.array(image), np.array(image_dec))
        if abs(realbpp - TargetBPP) > abs(bpp - TargetBPP):
            realbpp = bpp
            realpsnr = psnr
            realQ = Q
    return image, realQ, realbpp, realpsnr


def display_images_and_save_pdf(test_dataset, imgs_decoded, imgsQ_decoded, bpp, filepath, NumImagesToShow=None):
    if NumImagesToShow is None:
        NumImagesToShow = len(test_dataset)
    cols = NumImagesToShow
    rows = 4

    fig = plt.figure(figsize=(2 * cols, 2 * rows))
    psnr_decoded = []
    psnr_decoded_q = []
    psnr_jpeg = []

    for i in range(NumImagesToShow):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(to_pil_transform(test_dataset[i]), interpolation="nearest")
        plt.title("", fontsize=10)
        plt.axis('off')

    for i in range(NumImagesToShow):
        psnr = PSNR(test_dataset[i], imgs_decoded[i])
        psnr_decoded.append(psnr)
        plt.subplot(rows, cols, cols + i + 1)
        plt.imshow(to_pil_transform(imgs_decoded[i]), interpolation="nearest")
        plt.title(f"{psnr:.2f}", fontsize=10)
        plt.axis('off')

    for i in range(NumImagesToShow):
        psnr = PSNR(test_dataset[i], imgsQ_decoded[i])
        psnr_decoded_q.append(psnr)
        plt.subplot(rows, cols, 2 * cols + i + 1)
        plt.imshow(to_pil_transform(imgsQ_decoded[i]), interpolation="nearest")
        plt.title(f"{psnr:.2f} {bpp[i]:.2f}", fontsize=10)
        plt.axis('off')

    for i in range(NumImagesToShow):
        jpeg_img, JPEGQP, JPEGrealbpp, JPEGrealpsnr = JPEGRDSingleImage(test_dataset[i], bpp[i])
        psnr_jpeg.append(JPEGrealpsnr)
        plt.subplot(rows, cols, 3 * cols + i + 1)
        plt.imshow(jpeg_img, interpolation="nearest")
        plt.title(f"{JPEGrealpsnr:.2f} {JPEGrealbpp:.2f}", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(filepath, format='pdf')
    return fig, np.mean(psnr_decoded), np.mean(psnr_decoded_q), np.mean(psnr_jpeg)


def process_images(model: nn.Module, test_dataloader: DataLoader, device: str, b: int, w: int, h: int):
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
        encoded_bits = entropy_encoder(quantized_imgs_encoded[i], size_z, size_h, size_w)
        byte_size = len(encoded_bits)
        bpp.append(byte_size * 8 / (w * h))
        quantized_imgs_decoded.append(entropy_decoder(encoded_bits, size_z, size_h, size_w))
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
