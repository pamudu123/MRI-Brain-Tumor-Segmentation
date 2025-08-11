import numpy as np
import torch
import cv2

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class Resize:
    def __init__(self, size: int):
        self.size = size
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)

class ToTensorImage:
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        image = np.expand_dims(image, axis=0)
        return torch.from_numpy(image)

class ToTensorMask:
    def __call__(self, mask: np.ndarray) -> torch.Tensor:
        mask = mask.astype(np.float32)
        mask = (mask > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)
        return torch.from_numpy(mask)

class Normalize:
    def __init__(self, mean: float = 0.5, std: float = 0.5):
        self.mean = mean
        self.std = std
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std


def build_image_transforms(size: int):
    return Compose([Resize(size), ToTensorImage(), Normalize(0.5, 0.5)])


def build_mask_transforms(size: int):
    return Compose([Resize(size), ToTensorMask()])
