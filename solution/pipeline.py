import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from loguru import logger

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from solution import config
from solution.dataset_loader import ImageDataset, image_transform
from solution.models.base_model import BasePytorchModel
from solution.utils import display_images_and_save_pdf, process_images


def is_val_epoch(epoch_num: int) -> bool:
    return (epoch_num + 1) % config.VAL_EPOCH_EACH_STEP == 0


def training_pipeline(model: BasePytorchModel, device: str, run, b: int, w: int, h: int):
    logger.info("Start training pipeline")

    logger.info("Prepare train data utils")
    train_dataset = ImageDataset(config.DATASET_DIR / "train", image_transform)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )

    logger.info("Prepare test data utils")
    test_dataset = ImageDataset(config.DATASET_DIR / "test", image_transform)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    global_step_counter = 0

    for epoch in tqdm(range(config.EPOCHS)):
        model.train()

        for step, train_batch in enumerate(train_dataloader):
            train_batch = train_batch.to(device)

            optimizer.zero_grad()

            outputs = model(train_batch, b_t=b)
            loss = nn.MSELoss()(outputs, train_batch)

            run.log(
                {"train/loss": loss, "epoch": epoch, "step": global_step_counter},
                step=global_step_counter,
            )

            loss.backward()
            optimizer.step()

            global_step_counter += 1

        if is_val_epoch(epoch):
            logger.info("Start val epoch on %s" % epoch)
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_batch in tqdm(test_dataloader):
                    val_batch = val_batch.to(device)
                    val_outputs = model(val_batch, b_t=b)
                    val_loss += nn.MSELoss()(val_outputs, val_batch).item()

            val_loss /= len(test_dataloader)

            imgs_decoded, imgsQ_decoded, bpp = process_images(
                model, test_dataloader, device, b, w, h
            )

            fig, psnr_decoded, psnr_decoded_q, psnr_jpeg = display_images_and_save_pdf(
                test_dataset, imgs_decoded, imgsQ_decoded, bpp, config.WEIGHTS_DIR / model.model_name/ "output.pdf"
            )

            run.log(
                {
                    "val/loss": val_loss,
                    "val/img": fig,
                    "val/psnr_ae": psnr_decoded,
                    "val/psnr_ae_q": psnr_decoded_q,
                    "val/psnr_jpeg": psnr_jpeg,
                    "val/bpp": np.mean(bpp),
                    "epoch": epoch,
                    "step": global_step_counter,
                },
                step=global_step_counter,
            )
            plt.close()
            output_dir = config.WEIGHTS_DIR / model.model_name
            save_path = output_dir / f"epoch_{epoch}"

            logger.info("Saved model: %s" % str(save_path))
            model.save(save_path)

    logger.info("Training pipeline completed")
