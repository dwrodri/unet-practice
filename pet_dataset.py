import logging
import pathlib
from typing import List, Any

import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class PetDataset(Dataset):
    """
    Dog Pictures Dataset
    See: https://www.robots.ox.ac.uk/~vgg/data/pets/
    """

    def __init__(
        self,
        image_paths: List[pathlib.Path],
        mask_paths: List[pathlib.Path],
        desired_shape: List[int] = [128, 128],
        augmentations=[],
    ) -> None:
        self.image_paths_ = image_paths
        self.mask_paths_ = mask_paths
        self.desired_shape_ = desired_shape
        self.augmentations_ = augmentations

    def __len__(self):
        return (
            len(self.image_paths_) * len(self.augmentations_)
            if self.augmentations_ != []
            else len(self.image_paths_)
        )

    def __getitem__(self, idx: int) -> Any:
        image_pixels: Image.Image = Image.open(
            self.image_paths_[idx % len(self.image_paths_)]
        ).convert("RGB")
        mask_pixels: Image.Image = Image.open(
            self.mask_paths_[idx % len(self.mask_paths_)]
        ).convert("L")
        # the ToTensor transform automatically converts to float and rescales px values
        image_tensor: torch.Tensor = torchvision.transforms.ToTensor()(image_pixels)
        mask_tensor: torch.Tensor = torchvision.transforms.ToTensor()(mask_pixels)
        if image_tensor.shape[1:] != mask_tensor.shape[1:]:
            logging.error(
                f" Image {self.image_paths_[idx % len(self.image_paths_)]} Shape {image_tensor.shape} does not match mask {self.mask_paths_[idx % len(self.mask_paths_)]} shape {mask_tensor.shape}"
            )

        # masks are 1D with three values, we want a simple binary mask
        mask_tensor = (mask_tensor - 0.0078 < 1e-9).to(torch.float)

        if image_tensor.shape[-2:] != torch.Size(self.desired_shape_):
            image_tensor = torchvision.transforms.Resize(self.desired_shape_)(
                image_tensor
            )
            mask_tensor = torchvision.transforms.Resize(
                self.desired_shape_,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT,
            )(mask_tensor)
        if idx // len(self.image_paths_) > 0 and self.augmentations_ != []:
            augmentation_idx = (idx // len(self.image_paths_)) - 1
            image_tensor = self.augmentations_[augmentation_idx](image_tensor)
            mask_tensor = self.augmentations_[augmentation_idx](mask_tensor)
        return image_tensor, mask_tensor
