import torch

from torch.utils.data import DataLoader, random_split

from dataset import FetalHeadDataset
from model import UNet
from losses import BCEDiceLoss
from tqdm import tqdm



# SETTINGS


IMAGE_DIR = "data/training_set"
MASK_DIR = "data/filled_masks"

BATCH_SIZE = 10
EPOCHS = 20
LEARNING_RATE = 0.001



# DEVICE


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# DATASET


dataset = FetalHeadDataset(
    IMAGE_DIR,
    MASK_DIR
)

total_size = len(dataset)

train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)

test_size = (
    total_size
    - train_size
    - val_size
)

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

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



# MODEL

model = UNet().to(device)



# LOSS


criterion = BCEDiceLoss()



# OPTIMIZER


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)



# METRICS


def dice_score(logits, targets):

    smooth = 1e-6

    probs = torch.sigmoid(logits)

    preds = (
        probs > 0.5
    ).float()

    intersection = (
        preds * targets
    ).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        preds.sum()
        + targets.sum()
        + smooth
    )

    return dice.item()


def iou_score(logits, targets):

    smooth = 1e-6

    probs = torch.sigmoid(logits)

    preds = (
        probs > 0.5
    ).float()

    intersection = (
        preds * targets
    ).sum()

    union = (
        preds.sum()
        + targets.sum()
        - intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.item()


# TRAIN
# ----------------------------

best_dice = 0.0

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0

    for images, masks in tqdm(
    train_loader,
    desc=f"Epoch {epoch + 1}/{EPOCHS}"
):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(
            logits,
            masks
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = (
        total_train_loss
        / len(train_loader)
    )


  
    # VALIDATION
  

    model.eval()

    total_val_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            loss = criterion(
                logits,
                masks
            )

            total_val_loss += loss.item()

            total_dice += dice_score(
                logits,
                masks
            )

            total_iou += iou_score(
                logits,
                masks
            )

    avg_val_loss = (
        total_val_loss
        / len(val_loader)
    )

    avg_dice = (
        total_dice
        / len(val_loader)
    )

    avg_iou = (
        total_iou
        / len(val_loader)
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Dice: {avg_dice:.4f} | "
        f"IoU: {avg_iou:.4f}"
    )

    if avg_dice > best_dice:

        best_dice = avg_dice

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print(
            f"Best model saved! Dice: {best_dice:.4f}"
        )


print("Training finished.")