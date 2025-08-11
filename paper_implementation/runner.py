import glob
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from config import (
    DATA_DIR,
    NUM_FOLDS,
    BATCH_SIZE,
    CLASSIFIER_INPUT_SIZE,
    MODEL_SAVE_DIR,
    CLASSIFIER_MODEL_NAME,
    CLASSIFIER_LR,
    CLASSIFIER_EPOCHS,
    NUM_CLASSES_CLASSIFIER,
)
from dataset import BrainTumorDataset
from models import TumourClassifier
from train import train_classifier_one_epoch
from transforms import build_classifier_transforms, build_detector_transforms
from utils import set_seed, collate_batch, ensure_dir


def collect_mat_files(data_dir: str):
    patterns = ["*.mat", "**/*.mat"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(data_dir, p), recursive=True))
    files = sorted(list(set(files)))
    return files


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_files = collect_mat_files(DATA_DIR)
    if len(all_files) == 0:
        raise FileNotFoundError(f"No .mat files found under {DATA_DIR}")

    transform_dict = {
        "classifier": build_classifier_transforms(CLASSIFIER_INPUT_SIZE),
        "detector": build_detector_transforms(CLASSIFIER_INPUT_SIZE),
    }

    dataset = BrainTumorDataset(filepaths=all_files, transform_dict=transform_dict)

    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    ensure_dir(MODEL_SAVE_DIR)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(dataset))), start=1):
        print(f"\n===== Fold {fold}/{NUM_FOLDS} =====")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_batch,
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_batch,
        )

        model = TumourClassifier(num_classes=NUM_CLASSES_CLASSIFIER).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=CLASSIFIER_LR)
        criterion = torch.nn.CrossEntropyLoss()

        best_val_acc = 0.0
        fold_model_path = os.path.join(
            MODEL_SAVE_DIR, f"{Path(CLASSIFIER_MODEL_NAME).stem}_fold{fold}.pth"
        )

        for epoch in range(1, CLASSIFIER_EPOCHS + 1):
            print(f"\n--- Epoch {epoch}/{CLASSIFIER_EPOCHS} (Fold {fold}) ---")
            _, _, _, val_acc, best_val_acc = train_classifier_one_epoch(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                best_val_acc=best_val_acc,
                model_save_path=fold_model_path,
            )

        fold_results.append((fold, best_val_acc))

    # Summary
    print("\n===== Cross-Validation Summary =====")
    for fold, acc in fold_results:
        print(f"Fold {fold}: Best Val Acc = {acc:.4f}")
    if fold_results:
        mean_acc = sum(a for _, a in fold_results) / len(fold_results)
        print(f"Mean Best Val Acc: {mean_acc:.4f}")


if __name__ == "__main__":
    main()
