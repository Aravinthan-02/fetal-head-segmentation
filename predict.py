import random

import matplotlib.pyplot as plt

import torch
from torch.utils.data import random_split

from dataset import FetalHeadDataset
from model import UNet


IMAGE_DIR = "data/training_set"
MASK_DIR = "data/filled_masks"


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ----------------------------
# DATASET
# ----------------------------

dataset = FetalHeadDataset(
    IMAGE_DIR,
    MASK_DIR
)

total_size = len(dataset)

train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

_, _, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)


# Pick random TEST image
random_index = random.randint(
    0,
    len(test_dataset) - 1
)

image, true_mask = test_dataset[
    random_index
]


print(
    "Random test index:",
    random_index
)


# ----------------------------
# MODEL
# ----------------------------

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

model.eval()


# ----------------------------
# PREDICTION
# ----------------------------

input_tensor = (
    image.unsqueeze(0).to(device)
)


with torch.no_grad():

    logits = model(
        input_tensor
    )

    probability = torch.sigmoid(
        logits
    )

    predicted_mask = (probability > 0.5).float()


predicted_mask = (
    predicted_mask
    .squeeze()
    .cpu()
    .numpy()
)


# ----------------------------
# DISPLAY
# ----------------------------

plt.figure(figsize=(12, 4))


plt.subplot(1, 3, 1)

plt.imshow(
    image.squeeze(),
    cmap="gray"
)

plt.title("Ultrasound")

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    true_mask.squeeze(),
    cmap="gray"
)

plt.title("Ground Truth")

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    predicted_mask,
    cmap="gray"
)

plt.title("Prediction")

plt.axis("off")


plt.show()