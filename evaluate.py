import torch
from torch.utils.data import DataLoader, random_split

from dataset import FetalHeadDataset
from model import UNet
from losses import BCEDiceLoss


IMAGE_DIR = "data/training_set"
MASK_DIR = "data/filled_masks"

MODEL_PATH = "best_model.pth"



# DEVICE


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# LOAD COMPLETE DATASET


dataset = FetalHeadDataset(
    IMAGE_DIR,
    MASK_DIR
)


# CREATE SAME 70 / 15 / 15 SPLIT


total_size = len(dataset)

train_size = int(
    0.70 * total_size
)

val_size = int(
    0.15 * total_size
)

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


# TEST DATA LOADER


test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)


# LOAD MODEL


model = UNet().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print("Best model loaded successfully.")



# LOSS


criterion = BCEDiceLoss()



# DICE METRIC


def dice_score(logits, targets):

    smooth = 1e-6

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities > 0.5
    ).float()

    predictions = predictions.view(-1)

    targets = targets.view(-1)


    intersection = (
        predictions * targets
    ).sum()


    dice = (
        2 * intersection + smooth
    ) / (
        predictions.sum()
        + targets.sum()
        + smooth
    )


    return dice.item()



# IOU METRIC


def iou_score(logits, targets):

    smooth = 1e-6

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities > 0.5
    ).float()

    predictions = predictions.view(-1)

    targets = targets.view(-1)


    intersection = (
        predictions * targets
    ).sum()


    union = (
        predictions.sum()
        + targets.sum()
        - intersection
    )


    iou = (intersection + smooth) / (union + smooth)


    return iou.item()


# TESTING


total_loss = 0.0
total_dice = 0.0
total_iou = 0.0

number_of_samples = 0


with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(device)

        masks = masks.to(device)


        # Forward propagation
        logits = model(
            images
        )


        # Test loss
        loss = criterion(
            logits,
            masks
        )


        # Metrics
        dice = dice_score(
            logits,
            masks
        )

        iou = iou_score(
            logits,
            masks
        )


        total_loss += loss.item()

        total_dice += dice

        total_iou += iou

        number_of_samples += 1



# FINAL RESULTS


average_loss = (
    total_loss
    / number_of_samples
)

average_dice = (
    total_dice
    / number_of_samples
)

average_iou = (
    total_iou
    / number_of_samples
)


print("\n-----------------------------")
print("FINAL TEST RESULTS")
print("-----------------------------")

print(
    f"Test Loss : {average_loss:.4f}"
)

print(
    f"Test Dice : {average_dice:.4f}"
)

print(
    f"Test IoU  : {average_iou:.4f}"
)

print("-----------------------------")