import numpy as np
import torch


def PSNR_RGB(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def PSNR(y_true, y_pred):
    max_pixel = 1.0
    mse = torch.mean(torch.square(y_pred - y_true))
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(max_pixel**2 / mse)).item()
