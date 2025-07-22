import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class LandmarkDataset(Dataset):
    def __init__(self, images_dir, csv_path, transform=None):
        self.images_dir = images_dir
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = row['id']
        label = row['landmark_id']
        # Images are stored as a/b/c/0123456789abcdef.jpg
        img_path = os.path.join(
            self.images_dir,
            img_id[0], img_id[1], img_id[2], f"{img_id}.jpg"
        )
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
