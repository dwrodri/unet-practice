import torchvision
import itertools
import pathlib
import sys
from typing import Tuple
import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split


logging.basicConfig(level=logging.ERROR)


class SaltSeismologyDataset(Dataset):
    def __init__(
        self,
        image_paths: pathlib.Path,
        mask_paths: pathlib.Path,
        transforms: torch.nn.Module = torchvision.transforms.ToTensor(),
    ):
        self.image_paths_ = image_paths
        self.mask_paths_ = mask_paths
        self.transforms_ = transforms

    def __len__(self):
        return len(self.image_paths_)

    def __getitem__(self, idx: int):
        image_pixels = Image.open(self.image_paths_[idx]).convert("RGB")
        mask_pixels = Image.open(self.mask_paths_[idx]).convert("L")
        if self.transforms_ is not None:
            image_pixels = self.transforms_(image_pixels)
            mask_pixels = self.transforms_(mask_pixels)
        return image_pixels, mask_pixels


class UNetConvBlock(torch.nn.Module):
    """
    Simple chain of Conv2d -> ReLU -> Conv2d
    """

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.first_conv = torch.nn.Conv2d(
            in_channels=input_channels, out_channels=output_channels, kernel_size=3
        )
        self.second_conv = torch.nn.Conv2d(
            in_channels=output_channels, out_channels=output_channels, kernel_size=3
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = F.relu(self.first_conv(tensor))
        tensor = self.second_conv(tensor)
        return tensor


class UNet(torch.nn.Module):
    def __init__(
        self,
        encoder_channel_stages: Tuple[int] = (3, 16, 32, 64),
        num_classes: int = 1,
        retain_dim: bool = True,
        output_size: Tuple[int, int] = (101, 101),
    ) -> None:
        super().__init__()
        # Encoder is just a stack of these blocks
        self.encoder_blocks_ = torch.nn.ModuleList(
            [
                UNetConvBlock(in_chan, out_chan)
                for in_chan, out_chan in itertools.pairwise(encoder_channel_stages)
            ]
        )

        # The decoder has channels staged in a reverse pattern from the encoder, and has no input layer
        decoder_channel_stages = encoder_channel_stages[1:][::-1]
        self.transposed_convs_ = torch.nn.ModuleList(
            [
                torch.nn.ConvTranspose2d(
                    in_channels=in_chan, out_channels=out_chan, kernel_size=2, stride=2
                )
                for in_chan, out_chan in itertools.pairwise(decoder_channel_stages)
            ]
        )
        self.decoder_blocks_ = torch.nn.ModuleList(
            [
                UNetConvBlock(in_chan, out_chan)
                for in_chan, out_chan in itertools.pairwise(decoder_channel_stages)
            ]
        )

        # The final conv that generates the mask takes the decoder output and generates logit scores for each class
        self.head = torch.nn.Conv2d(
            decoder_channel_stages[-1], num_classes, kernel_size=1
        )
        self.retain_dim = retain_dim
        self.output_size = output_size

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        encoder_activations = []
        # Encoder pass, save layer activations which will need to be forwarded
        # to corresponding decoder layer
        for i, encoder_block in enumerate(self.encoder_blocks_):
            tensor = encoder_block(tensor)
            encoder_activations.append(tensor)
            tensor = F.max_pool2d(tensor, kernel_size=2)

        # Decoder pass
        for decoder_block, transposed_conv, encoder_activation in zip(
            self.decoder_blocks_,
            self.transposed_convs_,
            reversed(encoder_activations[:-1]),
        ):
            tensor = transposed_conv(tensor)
            # center crop + concat the encoder activation (tensor dim labels are BCHW)
            encoder_activation = torchvision.transforms.CenterCrop(tensor.shape[-2:])(
                encoder_activation
            )
            tensor = torch.cat([tensor, encoder_activation], dim=1)
            # then run through regular conv block
            tensor = decoder_block(tensor)

        # convert decoder output to classification map
        tensor = self.head(tensor)

        if self.retain_dim:
            tensor = F.interpolate(tensor, self.output_size)

        return tensor


def create_train_test_segmentation_datasets(
    images_folder: pathlib.Path, masks_folder: pathlib.Path, train_percent: float = 0.8
) -> Tuple[Dataset, Dataset]:
    """
    Collect sample images and masks from their respective folders and create train/test datasets
    """
    # validate path objects
    if not images_folder.is_dir():
        raise Exception(f"{images_folder} is not a folder")
    if not masks_folder.is_dir():
        raise Exception(f"{masks_folder} is not a folder")
    image_paths = sorted(
        [filepath for filepath in images_folder.iterdir() if ".png" == filepath.suffix]
    )
    mask_paths = sorted(
        [filepath for filepath in masks_folder.iterdir() if ".png" == filepath.suffix]
    )
    if len(image_paths) != len(mask_paths):
        raise UserWarning(
            f"number of images ({len(image_paths)}) and number of masks ({len(mask_paths)}) don't match"
        )
    train_pairs, test_pairs = random_split(
        list(zip(image_paths, mask_paths)), lengths=[train_percent, 1.0 - train_percent]
    )
    return SaltSeismologyDataset(*zip(*train_pairs)), SaltSeismologyDataset(
        *zip(*test_pairs)
    )


def fit_unet(
    model: UNet,
    train_set: Dataset,
    test_set: Dataset,
    batch_size: int,
    num_epochs: int,
    learning_rate: float = 0.001,
):
    # Pixel level binary cross-entropy, between ground-truth mask and generated mask
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    train_loader = DataLoader(
        dataset=train_set, shuffle=True, batch_size=batch_size, pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_set, shuffle=True, batch_size=batch_size, pin_memory=True
    )

    # Pretty vanilla training loop
    # NOTE: See https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#optimization-loop
    model.train()
    for _ in tqdm(range(num_epochs)):
        for image_batch, ground_truth_mask_batch in tqdm(train_loader):
            generated_mask_batch = model(image_batch.cuda())
            loss = loss_fn(generated_mask_batch, ground_truth_mask_batch.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Following epoch of training, evaluate model
        with torch.no_grad():
            model.eval()
            loss = 0.0
            for test_image_batch, test_ground_truth_mask_batch in test_loader:
                test_generated_mask_batch = model(test_image_batch.cuda())
                loss += loss_fn(
                    test_generated_mask_batch, test_ground_truth_mask_batch.cuda()
                ).cpu()
            logging.info(f"Test Loss: {loss}")


if __name__ == "__main__":
    train_set, test_set = create_train_test_segmentation_datasets(
        *map(pathlib.Path, sys.argv[1:])
    )
    fit_unet(
        model=UNet().cuda(),
        train_set=train_set,
        test_set=test_set,
        batch_size=64,
        num_epochs=40,
    )
