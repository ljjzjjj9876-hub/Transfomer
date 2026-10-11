from torch.utils.data import DataLoader
from torchvision import datasets, transforms


BATCH_SIZE = 32
IMAGE_SIZE = 64


transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


transform_test = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


def get_dataloaders():

    train_dataset = datasets.CIFAR10(
        root=r"C:\postgraduate\python\cnn_learing\ResNet18\data",
        train=True,
        download=False,
        transform=transform_train
    )


    test_dataset = datasets.CIFAR10(
        root=r"C:\postgraduate\python\cnn_learing\ResNet18\data",
        train=False,
        download=False,
        transform=transform_test
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