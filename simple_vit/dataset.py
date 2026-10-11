from torch.utils.data import DataLoader
from torchvision import datasets, transforms


BATCH_SIZE = 64


transform = transforms.Compose([
    transforms.ToTensor(),
])


def get_dataloaders():

    train_dataset = datasets.CIFAR10(
        root=r"C:\postgraduate\python\cnn_learing\ResNet18\data",
        train=True,
        download=False,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=r"C:\postgraduate\python\cnn_learing\ResNet18\data",
        train=False,
        download=False,
        transform=transform
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    return train_loader, test_loader