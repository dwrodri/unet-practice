import itertools
import logging
import pathlib
import sys
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# from salt_dataset import SaltSeismologyDataset
from pet_dataset import PetDataset

logging.basicConfig(level=logging.INFO)


class UNetConvBlock(torch.nn.Module):
    """
    Simple chain of Conv2d -> ReLU -> Conv
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
        encoder_channel_stages: List[int] = [3, 16, 32, 64],
        num_classes: int = 1,
        retain_dim: bool = True,
        output_size: Tuple[int, int] = (128, 128),
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
        [
            filepath
            for filepath in images_folder.iterdir()
            if filepath.suffix == ".png" or filepath.suffix == ".jpg"
        ]
    )
    mask_paths = sorted(
        [filepath for filepath in masks_folder.iterdir() if ".png" == filepath.suffix]
    )
    if len(image_paths) != len(mask_paths):
        raise UserWarning(
            f"number of images ({len(image_paths)}) and number of masks ({len(mask_paths)}) don't match"
        )
    train_pairs, test_pairs = random_split(
        list(zip(image_paths, mask_paths)),  # type:ignore
        lengths=[train_percent, 1.0 - train_percent],
    )
    # augmentations = [
    #     torchvision.transforms.RandomHorizontalFlip(p=1.0),
    #     torchvision.transforms.RandomVerticalFlip(p=1.0),
    # ]
    train_image_paths, train_mask_paths = zip(*train_pairs)
    train_set = PetDataset(train_image_paths, train_mask_paths)
    test_image_paths, test_mask_paths = zip(*test_pairs)
    test_set = PetDataset(test_image_paths, test_mask_paths)
    return train_set, test_set


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
        dataset=train_set,
        shuffle=True,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        dataset=test_set, shuffle=False, batch_size=batch_size, pin_memory=True
    )

    # Pretty vanilla training loop
    # NOTE: See https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#optimization-loop
    last_test_loss = sys.float_info.max
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for image_batch, ground_truth_mask_batch in tqdm(train_loader):
            generated_mask_batch = model(image_batch.cuda())
            loss = loss_fn(generated_mask_batch, ground_truth_mask_batch.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss

        # Following epoch of training, evaluate model
        with torch.no_grad():
            model = model.eval()
            loss = 0.0
            for test_image_batch, test_ground_truth_mask_batch in test_loader:
                test_generated_mask_batch = model(test_image_batch.cuda())
                loss += loss_fn(
                    test_generated_mask_batch, test_ground_truth_mask_batch.cuda()
                ).cpu()
            logging.info(f"Epoch {epoch} Train Loss:{train_loss} Test Loss: {loss}")
            if loss < last_test_loss:
                last_test_loss = loss
    visualize_sample(test_set, model, limit=32)
    torch.save(model, "unet_pets.pkl")


@torch.no_grad()
def visualize_sample(dataset: Dataset, model: UNet, limit: int = 64):
    demo_loader = DataLoader(
        dataset=dataset, shuffle=False, batch_size=1, pin_memory=True
    )
    for i, (demo_image, demo_ground_truth_mask) in enumerate(tqdm(demo_loader)):
        demo_generated_mask = model(demo_image.cuda()).cpu().squeeze()
        demo_generated_mask = (torch.sigmoid(demo_generated_mask) * 255).to(torch.uint8)
        side_by_side = torch.cat(
            [
                demo_generated_mask.squeeze(),
                demo_ground_truth_mask.squeeze(),
            ],
            dim=1,
        )
        comparison_image: Image.Image = torchvision.transforms.ToPILImage()(
            side_by_side
        )
        demo_input: Image.Image = torchvision.transforms.ToPILImage()(
            demo_image.squeeze()
        )
        demo_input.save(f"demo/input_{i}.png")
        comparison_image.save(f"demo/result_{i}.png")
        i += 1
        if i >= limit:
            break


if __name__ == "__main__":
    images_folder, masks_folder = map(pathlib.Path, sys.argv[1:3])
    train_set, test_set = create_train_test_segmentation_datasets(
        images_folder, masks_folder, 0.90
    )
    # model: UNet = torch.load("unet_augmented.pkl")
    # visualize_sample(train_set, model)
    fit_unet(
        model=UNet(encoder_channel_stages=[3, 64, 128, 256, 512]).cuda(),
        train_set=train_set,
        test_set=test_set,
        batch_size=64,
        num_epochs=80,
    )
