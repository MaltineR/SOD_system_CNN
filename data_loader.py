import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SaliencyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=128, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.transform = transform

        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        # paths
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace(".jpg", ".png"))

        # read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))

        # read mask (grayscale)
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        # normalize image
        image = image / 255.0

        # normalize mask (binary)
        mask = mask / 255.0
        mask = np.where(mask > 0.5, 1.0, 0.0)

        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)

        return image, mask
    
def get_loaders(train_img, train_mask, val_img, val_mask, test_img, test_mask, batch_size=8):
    
    train_dataset = SaliencyDataset(train_img, train_mask)
    val_dataset = SaliencyDataset(val_img, val_mask)
    test_dataset = SaliencyDataset(test_img, test_mask)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader