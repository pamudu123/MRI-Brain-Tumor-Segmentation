import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class MRISegmentationDataset(Dataset):
    """
    Dataset for MRI tumor localization/segmentation from .mat files.
    Expects keys: cjdata/image, cjdata/tumorMask. Ignores class label.
    """
    def __init__(self, filepaths, image_transform=None, mask_transform=None):
        self.filepaths = filepaths
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        try:
            with h5py.File(path, 'r') as f:
                cjdata = f['cjdata']
                image = np.array(cjdata['image'], dtype=np.float32)
                mask = np.array(cjdata['tumorMask'], dtype=np.uint8)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

        img_t = self.image_transform(image) if self.image_transform else torch.from_numpy(image)
        if img_t.ndim == 2:
            img_t = img_t.unsqueeze(0)
        m_t = self.mask_transform(mask) if self.mask_transform else torch.from_numpy(mask).unsqueeze(0).float()
        return {"image": img_t, "mask": m_t, "path": path}
