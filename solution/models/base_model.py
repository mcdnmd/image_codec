import os
from pathlib import Path

import torch
from loguru import logger
from torch import nn


class BasePytorchModel(nn.Module):
    def __init__(self, model_name: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    def save(self, dir_path: Path) -> None:
        os.makedirs(dir_path, exist_ok=True)
        path = dir_path / f"{self.model_name}.pth"
        torch.save(self.state_dict(), path)
        logger.info("Model saved into %s" % str(path))

    def load(self, device: str, dir_path: Path) -> None:
        path = dir_path / f"{self.model_name}.pth"
        self.load_state_dict(torch.load(path, map_location=device))
        logger.info("Model loaded from %s" % str(path))
