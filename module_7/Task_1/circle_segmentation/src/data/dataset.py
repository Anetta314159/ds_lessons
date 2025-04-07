
import torch
from torch.utils.data import Dataset
from .provider import create_image_and_label

class CircleSegmentationDataset(Dataset):
    def __init__(self, count, size=128):
        self.count = count
        self.size = size

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        img, msk = create_image_and_label(self.size, self.size)
        x = img.transpose(2, 0, 1)
        y = msk[None, ...]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
