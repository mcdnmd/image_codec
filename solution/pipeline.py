import numpy as np
import torch
from loguru import logger

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from solution import config
from solution.dataset_loader import ImageDataset, image_transform
from solution.utils import display_images_and_save_pdf, process_images


def is_val_epoch(epoch_num: int) -> bool:
    return (epoch_num + 1) % config.VAL_EPOCH_EACH_STEP == 0


def training_pipeline(model: nn.Module, device: str, run, b: int, w: int, h: int):
    logger.info("Start training pipeline")
    train_dataset = ImageDataset(config.DATASET_DIR / "train", image_transform)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )

    test_dataset = ImageDataset(config.DATASET_DIR / "test", image_transform)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=True
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

            loss.backward()
            optimizer.step()

            run.log(
                {"train loss": loss, "epoch": epoch, "step": global_step_counter},
                step=global_step_counter,
            )

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
            fig, psnr_decoded, psnr_decoded_q, _ = display_images_and_save_pdf(
                test_dataset, imgs_decoded, imgsQ_decoded, bpp, "output.pdf"
            )

            run.log(
                {
                    "val loss": val_loss,
                    "val pick": fig,
                    "val psnr ae": psnr_decoded,
                    "val psnr_ae_q": psnr_decoded_q,
                    "val bpp": np.mean(bpp),
                    "epoch": epoch,
                    "step": global_step_counter,
                },
                step=global_step_counter,
            )

    logger.info("Training pipeline completed")
