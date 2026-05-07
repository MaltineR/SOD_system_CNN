import os
import cv2
import torch
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split



IMAGE_DIR = "data/images"
MASK_DIR = "data/masks"

IMAGE_SIZE = 128
BATCH_SIZE = 8



class SODDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list, image_size=128, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = image_list
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        base_name = os.path.splitext(image_name)[0]

        image_path = os.path.join(self.image_dir, image_name)

        mask_path = None
        for file in os.listdir(self.mask_dir):
            if os.path.splitext(file)[0] == base_name:
                mask_path = os.path.join(self.mask_dir, file)
                break

        if mask_path is None:
            raise FileNotFoundError(f"No matching mask found for {image_name}")

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      
        if self.augment:
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)

            brightness_factor = random.uniform(0.8, 1.2)
            image = np.clip(image * brightness_factor, 0, 255).astype(np.uint8)

        # Normalize image to 0-1
        image = image.astype(np.float32) / 255.0

        # Convert mask to 0 or 1
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        # Convert to tensor format
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask



def create_loaders():
    all_images = sorted(os.listdir(IMAGE_DIR))

    print("Total images:", len(all_images))

    train_size = int(0.70 * len(all_images))
    val_size = int(0.15 * len(all_images))
    test_size = len(all_images) - train_size - val_size

    train_images, val_images, test_images = random_split(
        all_images,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_images = list(train_images)
    val_images = list(val_images)
    test_images = list(test_images)

    train_dataset = SODDataset(
        IMAGE_DIR,
        MASK_DIR,
        train_images,
        image_size=IMAGE_SIZE,
        augment=True
    )

    val_dataset = SODDataset(
        IMAGE_DIR,
        MASK_DIR,
        val_images,
        image_size=IMAGE_SIZE,
        augment=False
    )

    test_dataset = SODDataset(
        IMAGE_DIR,
        MASK_DIR,
        test_images,
        image_size=IMAGE_SIZE,
        augment=False
    )

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader



if __name__ == "__main__":
    train_loader, val_loader, test_loader = create_loaders()

    images, masks = next(iter(train_loader))

    print("Train batch image shape:", images.shape)
    print("Train batch mask shape:", masks.shape)

    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    print("DataLoader test successful.")
