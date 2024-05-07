import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import ROOT_DIR


class ImageDataset(Dataset):
    def __init__(self, absolute_dataset_path: Path, transform=None) -> None:
        self.files_paths = [
            absolute_dataset_path / f for f in os.listdir(absolute_dataset_path) if
            f.endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.files_paths)

    def __getitem__(self, index):
        img = Image.open(self.files_paths[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
