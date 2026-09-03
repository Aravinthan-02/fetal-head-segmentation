import os

from PIL import Image
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.transforms import InterpolationMode


class FetalHeadDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.image_files = sorted([
            file for file in os.listdir(image_dir)
            if file.endswith(".png")
            and "_Annotation" not in file
        ])

        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (256, 256),
                interpolation=InterpolationMode.NEAREST
            ),
            transforms.ToTensor()
        ])

        print(
            f"Found {len(self.image_files)} ultrasound images"
        )

    def __len__(self):

        return len(self.image_files)

    def __getitem__(self, index):

        image_filename = self.image_files[index]

        annotation_filename = image_filename.replace(
            ".png",
            "_Annotation.png"
        )

        image_path = os.path.join(
            self.image_dir,
            image_filename
        )

        mask_path = os.path.join(
            self.mask_dir,
            annotation_filename
        )

        if not os.path.exists(mask_path):

            raise FileNotFoundError(
                f"Mask not found: {mask_path}"
            )

        image = Image.open(
            image_path
        ).convert("L")

        mask = Image.open(
            mask_path
        ).convert("L")

        image = self.image_transform(image)

        mask = self.mask_transform(mask)

        mask = (mask > 0.5).float()

        return image, mask