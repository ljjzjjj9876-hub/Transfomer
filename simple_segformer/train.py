import torch
import torch.nn as nn

from dataset import get_dataloaders
from model import SimpleSegFormer


# ============================================================
# Settings
# ============================================================

EPOCHS = 10
LEARNING_RATE = 0.0003
NUM_CLASSES = 3


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Device:",
    device
)


# ============================================================
# Dataset
# ============================================================

train_loader, val_loader, val_dataset = (
    get_dataloaders()
)


# ============================================================
# Model
# ============================================================

model = SimpleSegFormer(
    num_classes=NUM_CLASSES
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
# mIoU
# ============================================================

def calculate_miou(
    predictions,
    masks,
    num_classes
):

    ious = []


    for cls in range(num_classes):

        pred_cls = (
            predictions == cls
        )

        mask_cls = (
            masks == cls
        )


        intersection = (
            pred_cls & mask_cls
        ).sum().item()


        union = (
            pred_cls | mask_cls
        ).sum().item()


        if union > 0:

            iou = (
                intersection / union
            )

            ious.append(iou)


    return (
        sum(ious) / len(ious)
    )


# ============================================================
# Training
# ============================================================

for epoch in range(EPOCHS):


    # ========================================================
    # Train
    # ========================================================

    model.train()

    total_loss = 0


    for images, masks in train_loader:

        images = images.to(device)

        masks = masks.to(device)


        # -----------------------------
        # 1. Forward
        # -----------------------------

        outputs = model(images)


        # -----------------------------
        # 2. Loss
        # -----------------------------

        loss = criterion(
            outputs,
            masks
        )


        # -----------------------------
        # 3. Clear gradients
        # -----------------------------

        optimizer.zero_grad()


        # -----------------------------
        # 4. Backward
        # -----------------------------

        loss.backward()


        # -----------------------------
        # 5. Update parameters
        # -----------------------------

        optimizer.step()


        total_loss += loss.item()


    # ========================================================
    # Validation
    # ========================================================

    model.eval()


    correct_pixels = 0
    total_pixels = 0

    total_miou = 0

    batches = 0


    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)

            masks = masks.to(device)


            outputs = model(images)


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            # =========================
            # Pixel Accuracy
            # =========================

            correct_pixels += (
                predictions == masks
            ).sum().item()


            total_pixels += (
                masks.numel()
            )


            # =========================
            # mIoU
            # =========================

            total_miou += calculate_miou(
                predictions,
                masks,
                NUM_CLASSES
            )


            batches += 1


    pixel_accuracy = (
        correct_pixels
        /
        total_pixels
    )


    miou = (
        total_miou
        /
        batches
    )


    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: "
        f"{total_loss / len(train_loader):.4f} | "
        f"Pixel Acc: "
        f"{pixel_accuracy * 100:.2f}% | "
        f"mIoU: "
        f"{miou * 100:.2f}%"
    )


# ============================================================
# Save Model
# ============================================================

torch.save(
    model.state_dict(),
    "segformer.pth"
)


print(
    "Training finished."
)

print(
    "Model saved: segformer.pth"
)