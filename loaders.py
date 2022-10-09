import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

class BinaryImage:
    def __call__(self, image: Image):
        image = np.array(image, dtype="float32")
        image = torch.from_numpy(image).unsqueeze(0)
        return image

class ToCategorical:
    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes

    def __call__(self, sample: Image):
        image = np.array(sample)
        categories = np.unique(image).tolist()[1:]
        one_hot_image = torch.zeros(self.n_classes, *image.shape[:-1])
        for category in categories:
            rows, cols, _ = np.where(image == category)
            one_hot_image[category, rows, cols] = 1
        return one_hot_image


class SegmentationDataset(Dataset):
    def __init__(self, data_path: str, transform=None, target_transform=None, img_exts=[".png", ".jpg", ".jpeg", ".tif"]) -> None:
        root = Path(data_path)
        if not root.exists():
            raise FileNotFoundError(f"No file found at {data_path}. Please check the path provided")
        self.img_exts = img_exts
        self.samples = [sample for sample in (root / "images").iterdir() if sample.suffix in self.img_exts]
        self.masks = [mask for mask in (root / "masks").iterdir() if mask.suffix in self.img_exts]
        if len(self.samples) != len(self.masks):
            raise IndexError("Samples and masks do not contain the same number of image files, unable to create a paired dataset")
        self.transform, self.target_transform = transform, target_transform
    
    def __len__(self):
        return len(self.samples) or len(self.masks)
    
    def __getitem__(self, index):
        sample, mask = cv2.imread(str(self.samples[index])), cv2.imread(str(self.masks[index]))
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            mask = self.target_transform(mask)
        return sample, mask