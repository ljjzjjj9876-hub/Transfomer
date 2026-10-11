import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import SimpleSwin


# ============================================================
# Settings
# ============================================================

EPOCHS = 10
LEARNING_RATE = 0.0003


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("Device:", device)


# ============================================================
# Dataset
# ============================================================

train_loader, test_loader = (
    get_dataloaders()
)


# ============================================================
# Model
# ============================================================

model = SimpleSwin(
    num_classes=10
).to(device)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# ============================================================
# Training
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        # 1. Forward
        outputs = model(images)


        # 2. Loss
        loss = criterion(
            outputs,
            labels
        )


        # 3. Clear gradients
        optimizer.zero_grad()


        # 4. Backward
        loss.backward()


        # 5. Update
        optimizer.step()


        total_loss += loss.item()


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: "
        f"{total_loss / len(train_loader):.4f}"
    )


# ============================================================
# Test
# ============================================================

model.eval()


correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        labels = labels.to(device)


        outputs = model(images)


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


accuracy = (
    100 * correct / total
)


print(
    f"Test Accuracy: "
    f"{accuracy:.2f}%"
)