import torch.nn as nn
from torchvision.models import swin_t


class SimpleSwin(nn.Module):

    def __init__(self, num_classes=10):

        super().__init__()


        # Official Swin-T
        # weights=None means:
        # train from scratch
        self.model = swin_t(
            weights=None
        )


        # Replace ImageNet 1000-class head
        # with CIFAR-10 10-class head

        in_features = self.model.head.in_features

        self.model.head = nn.Linear(
            in_features,
            num_classes
        )


    def forward(self, x):

        return self.model(x)