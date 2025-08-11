import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class BrainTumorDataset(Dataset):
    """
    Custom PyTorch Dataset for loading brain tumor data from .mat files.
    It extracts the image, class label, tumor mask, and tumor border.
    """
    def __init__(self, filepaths, transform_dict):
        self.filepaths = filepaths
        self.classifier_transform = transform_dict.get('classifier')
        self.detector_transform = transform_dict.get('detector')

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        try:
            with h5py.File(filepath, 'r') as f:
                cjdata = f['cjdata']
                
                # Load data and convert to numpy arrays
                image = np.array(cjdata['image'], dtype=np.float32)
                # Labels are 1-4 in the dataset (1=Meningioma, 2=Glioma, 3=Pituitary, 4=No Tumor)
                # For this research, we only use Meningioma (1) and Glioma (2)
                original_label = int(np.array(cjdata['label'])[0, 0])
                
                # Filter out pituitary tumors (label 3)
                if int(original_label) == 3:
                    return None
                
                # Convert to binary classification: Meningioma=0, Glioma=1
                label = 0 if original_label == 1 else 1
                tumor_mask = np.array(cjdata['tumorMask'], dtype=np.uint8)
                tumor_border = np.array(cjdata['tumorBorder'])[0] 
                tumor_mask = np.array(cjdata['tumorMask'], dtype=np.uint8)
                tumor_border = np.array(cjdata['tumorBorder'])[0]
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            return None

        # Prepare data for classifier
        classifier_img = self.classifier_transform(image) if self.classifier_transform else torch.from_numpy(image)
        
        # Prepare data for detector
        detector_img = self.detector_transform(image) if self.detector_transform else torch.from_numpy(image)
        
        # Create bounding box from tumor border coordinates
        x_coords = tumor_border[::2]
        y_coords = tumor_border[1::2]
        
        # Ensure there are coordinates to process
        if len(x_coords) > 0 and len(y_coords) > 0:
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
        else:
            # Handle cases with no border data
            x_min, y_min, x_max, y_max = 0, 0, 0, 0

        # Ensure box coordinates are valid (width and height must be > 0)
        if x_min >= x_max or y_min >= y_max:
            x_min, y_min, x_max, y_max = 0, 0, 1, 1 # Create a 1x1 pixel box

        boxes = torch.as_tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)
        # For detection, label '1' is the tumor class. Label '0' is background.
        labels = torch.as_tensor([1], dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((1,), dtype=torch.int64)
        }

        return {
            "classifier_img": classifier_img,
            "classifier_label": torch.tensor(label, dtype=torch.long),
            "detector_img": detector_img,
            "detector_target": target,
            "original_image": image,
            "ground_truth_mask": torch.from_numpy(tumor_mask)
        }
