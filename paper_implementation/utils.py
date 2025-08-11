import os
import random
from typing import List, Optional, Dict, Any

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_batch(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    # Filter out None samples
    samples = [b for b in batch if b is not None]
    if len(samples) == 0:
        return None

    # Stack classifier inputs
    cls_imgs = torch.stack([s["classifier_img"] for s in samples], dim=0)
    cls_labels = torch.stack([s["classifier_label"] for s in samples], dim=0)

    # Detector inputs (keep targets as list as per detection practice)
    det_imgs = torch.stack([s["detector_img"] for s in samples], dim=0)
    det_targets = [s["detector_target"] for s in samples]

    original_images = [s["original_image"] for s in samples]
    gt_masks = [s["ground_truth_mask"] for s in samples]

    return {
        "classifier_img": cls_imgs,
        "classifier_label": cls_labels,
        "detector_img": det_imgs,
        "detector_target": det_targets,
        "original_image": original_images,
        "ground_truth_mask": gt_masks,
    }


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
