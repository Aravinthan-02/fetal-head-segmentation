from dataset import FetalHeadDataset
import matplotlib.pyplot as plt


dataset = FetalHeadDataset(
    image_dir="data/training_set",
    mask_dir="data/filled_masks"
)


print("Number of samples:", len(dataset))


image, mask = dataset[0]


print("Image shape:", image.shape)
print("Mask shape:", mask.shape)
print("Mask values:", mask.unique())


plt.figure(figsize=(10, 4))


plt.subplot(1, 2, 1)

plt.imshow(
    image.squeeze(),
    cmap="gray"
)

plt.title("Ultrasound Image")

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    mask.squeeze(),
    cmap="gray"
)

plt.title("Filled Ground Truth")

plt.axis("off")


plt.show()