import pathlib
from typing import List, Tuple

import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class SaltSeismologyDataset(Dataset):
    """
    TGS Salt Seismology Dataset
    See: https://www.kaggle.com/c/tgs-salt-identification-challenge
    """

    def __init__(
        self,
        image_paths: List[pathlib.Path],
        mask_paths: List[pathlib.Path],
        augmentations=[],
    ):
        self.image_paths_ = image_paths
        self.mask_paths_ = mask_paths
        self.augmentations_ = augmentations

    def __len__(self):
        return min(len(self.image_paths_), len(self.mask_paths_)) * len(
            self.augmentations_
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_pixels: Image.Image = Image.open(
            self.image_paths_[idx % len(self.image_paths_)]
        ).convert("RGB")
        mask_pixels: Image.Image = Image.open(
            self.mask_paths_[idx % len(self.mask_paths_)]
        ).convert("L")
        # the ToTensor transform automatically converts to float and rescales px values
        image_tensor: torch.Tensor = torchvision.transforms.ToTensor()(image_pixels)
        mask_tensor: torch.Tensor = torchvision.transforms.ToTensor()(mask_pixels)
        if idx // len(self.image_paths_) > 0 and self.augmentations_ != []:
            augmentation_idx = (idx // len(self.image_paths_)) - 1
            image_tensor = self.augmentations_[augmentation_idx](image_tensor)
            mask_tensor = self.augmentations_[augmentation_idx](mask_tensor)
        return image_tensor, mask_tensor
