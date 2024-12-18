import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

class ImageFeatureDataset(Dataset):
    def __init__(self, base_path, features, filenames, labels, transform=None):
        self.base_path = base_path
        self.features = features
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        features = self.features[idx]
        filename = self.filenames[idx]
        label = self.labels[idx]

        folder_name = filename.rsplit('_', 1)[0]
        image_path = os.path.join(self.base_path, folder_name, filename)
        mask_filename = filename.replace('.tif', '_mask.tif')
        mask_path = os.path.join(self.base_path, folder_name, mask_filename)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')
        image = image.resize((256, 256))
        mask = mask.resize((256, 256))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        image = torch.tensor(np.array(image), dtype=torch.float32).unsqueeze(0) / 255.0
        mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0) / 255.0
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return image, mask, features, label