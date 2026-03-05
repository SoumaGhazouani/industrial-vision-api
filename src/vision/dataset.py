# src/vision/dataset.py

import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MVTecBinaryDataset(Dataset):
    """
    Custom Dataset for MVTec anomaly detection (binary classification).
    good = 0
    defect = 1 
    """

    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): path to mvtec category (e.g. data/mvtec/bottle)
            split (str): "train" or "test"
            transform: torchvision transforms
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        self.image_paths = []
        self.labels = []

        self._load_dataset()

    def _load_dataset(self):
        split_path = self.root_dir / self.split

        for defect_type in os.listdir(split_path):
            defect_path = split_path / defect_type

            for img_name in os.listdir(defect_path):
                img_path = defect_path / img_name

                self.image_paths.append(img_path)

                if defect_type == "good":
                    self.labels.append(0)
                else:
                    self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
