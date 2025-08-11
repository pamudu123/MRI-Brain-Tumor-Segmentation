"""
Config for Ranasinghe MRI Brain Tumor Localization and Segmentation experiment.
Module-level variables for unified structure.
"""

# Data
DATA_DIR: str = "../Datasets/Raw/TumorData"  # TODO: adjust if this experiment uses different data
IMAGE_SIZE: int = 256  # placeholder input size for segmentation/localization

# Training
LR: float = 1e-3
EPOCHS: int = 50
BATCH_SIZE: int = 4
NUM_FOLDS: int = 5

# Saving
MODEL_SAVE_DIR: str = "saved_models/segmentation"
MODEL_NAME: str = "best_model.pth"
