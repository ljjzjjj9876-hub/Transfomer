import torch

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGE_SIZE = 128
BATCH_SIZE = 4


# =========================
# Image transform
# =========================

image_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor()
])


# =========================
# Mask transform
# =========================

mask_transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=InterpolationMode.NEAREST
    ),
    transforms.PILToTensor()
])


def process_mask(mask):

    mask = mask_transform(mask)

    # [1,H,W] -> [H,W]
    mask = mask.squeeze(0)

    # Oxford Pet 原标签 1,2,3
    # 改成 0,1,2
    mask = mask.long() - 1

    return mask


def get_dataloaders():

    dataset = OxfordIIITPet(
        root=r"C:\postgraduate\python\cnn_learing\UNet\data",
        split="trainval",
        target_types="segmentation",
        download=True,
        transform=image_transform,
        target_transform=process_mask
    )


    train_size = int(
        0.8 * len(dataset)
    )

    val_size = (
        len(dataset) - train_size
    )


    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    return (
        train_loader,
        val_loader,
        val_dataset
    )