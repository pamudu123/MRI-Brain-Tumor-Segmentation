import numpy as np
import torch
import cv2

class ToTensor:
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        # Expect grayscale uint8/float32 image HxW
        if image.ndim == 3:
            # If HxWxC, convert to grayscale if needed
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image / 255.0
        # Add channel dimension -> 1xHxW
        image = np.expand_dims(image, axis=0)
        return torch.from_numpy(image)

class Resize:
    def __init__(self, size: int):
        self.size = size
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)

class Normalize:
    def __init__(self, mean: float = 0.5, std: float = 0.5):
        self.mean = mean
        self.std = std
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # tensor: 1xHxW
        return (tensor - self.mean) / self.std

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        x = image
        for t in self.transforms:
            x = t(x)
        return x


def build_classifier_transforms(input_size: int):
    return Compose([
        Resize(input_size),
        ToTensor(),
        Normalize(mean=0.5, std=0.5),
    ])


def build_detector_transforms(input_size: int):
    # Placeholder: use same as classifier for now
    return build_classifier_transforms(input_size)
