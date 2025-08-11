import glob
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from .config import DATA_DIR, IMAGE_SIZE, BATCH_SIZE, EPOCHS, LR, NUM_FOLDS, MODEL_SAVE_DIR, MODEL_NAME
from .dataset import MRISegmentationDataset
from .transforms import build_image_transforms, build_mask_transforms
from .models import UNetSmall
from .engine import train_one_epoch, evaluate
from .utils import set_seed, ensure_dir


def collect_mat_files(root: str):
    files = glob.glob(os.path.join(root, "**", "*.mat"), recursive=True)
    return sorted(list(set(files)))


def main():
    """
    Segmentation/localization runner scaffold (training loop commented by default).
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[segmentation] Device: {device}")

    paths = collect_mat_files(DATA_DIR)
    if not paths:
        raise FileNotFoundError(f"No .mat files under {DATA_DIR}")

    img_t = build_image_transforms(IMAGE_SIZE)
    msk_t = build_mask_transforms(IMAGE_SIZE)
    dataset = MRISegmentationDataset(paths, image_transform=img_t, mask_transform=msk_t)

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    ensure_dir(MODEL_SAVE_DIR)

    # for fold, (tr_idx, va_idx) in enumerate(kf.split(range(len(dataset))), start=1):
    #     train_ds = Subset(dataset, tr_idx)
    #     val_ds = Subset(dataset, va_idx)
    #     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    #     val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    #     model = UNetSmall().to(device)
    #     optim = torch.optim.Adam(model.parameters(), lr=LR)
    #     best_dice = 0.0
    #     for epoch in range(1, EPOCHS + 1):
    #         tr_loss, tr_dice = train_one_epoch(model, train_loader, optim, device)
    #         va_loss, va_dice = evaluate(model, val_loader, device)
    #         if va_dice > best_dice:
    #             best_dice = va_dice
    #             torch.save({
    #                 'model_state_dict': model.state_dict(),
    #                 'best_val_dice': float(best_dice)
    #             }, os.path.join(MODEL_SAVE_DIR, f"{Path(MODEL_NAME).stem}_fold{fold}.pth"))
    #     print(f"Fold {fold}: best Dice = {best_dice:.4f}")

if __name__ == "__main__":
    main()
