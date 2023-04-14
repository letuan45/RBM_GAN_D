import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage import io
import os
from torchvision import transforms

class IotDataset(Dataset):
    """IOT dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform= transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Grayscale()
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])

        image = io.imread(img_path)
        y_label =torch.tensor(int(self.annotations.iloc[idx,1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)