import argparse

import torch
import wandb

from solution import config
from solution.config import WANDB_PROJECT_NAME
from solution.models.base_auto_encoder_model import AutoEncoder
from solution.models.residual_model import ResAutoEncoder
from solution.models.stride_model import StrideAutoEncoder
from solution.models.vae_model import VAE
from solution.pipeline import training_pipeline


if __name__ == "__main__":
    w, h = 128, 128
    b = 2
    device = (
        "mps"
        if torch.backends.mps.is_built()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    parser = argparse.ArgumentParser(description="A iamge codec model executor tool")
    parser.add_argument(
        "model_type", help="Name of the model", choices=["base", "stride", "vae", "residual"]
    )
    args = parser.parse_args()

    match args.model_type:
        case "base":
            model = AutoEncoder().to(device)
        case "stride":
            model = StrideAutoEncoder().to(device)
        case "vae":
            model = VAE().to(device)
        case "residual":
            model = ResAutoEncoder().to(device)
        case _:
            raise ValueError(f"No such model {args.model_type}")

    run = wandb.init(
        project=WANDB_PROJECT_NAME,
        config={
            "learning_rate": config.LEARNING_RATE,
            "dataset": "itmo_image_codec_cars",
            "epochs": config.EPOCHS,
        },
        name=f"{model.model_name}-b_{b}-lr_{config.LEARNING_RATE}",
    )
    try:
        training_pipeline(model=model, device=device, run=run, b=b, w=w, h=h)
    finally:
        wandb.finish()
