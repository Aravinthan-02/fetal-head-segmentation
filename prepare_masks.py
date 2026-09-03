import os
import cv2
import numpy as np


DATA_DIR = "data/training_set"
OUTPUT_DIR = "data/filled_masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)


annotation_files = [
    file for file in os.listdir(DATA_DIR)
    if file.endswith("_Annotation.png")
]


print(f"Found {len(annotation_files)} annotations")


for filename in annotation_files:

    input_path = os.path.join(
        DATA_DIR,
        filename
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    # Read annotation as grayscale
    mask = cv2.imread(
        input_path,
        cv2.IMREAD_GRAYSCALE
    )

    # Convert everything non-black to white
    _, binary = cv2.threshold(
        mask,
        1,
        255,
        cv2.THRESH_BINARY
    )

    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Empty black image
    filled_mask = np.zeros_like(binary)

    if len(contours) > 0:

        # Take largest contour
        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        # Fill inside contour
        cv2.drawContours(
            filled_mask,
            [largest_contour],
            contourIdx=-1,
            color=255,
            thickness=cv2.FILLED
        )

    cv2.imwrite(
        output_path,
        filled_mask
    )


print("Finished creating filled masks!")
print("Saved to:", OUTPUT_DIR)