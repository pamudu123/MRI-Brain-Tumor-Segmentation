"""
Unified configuration module with module-level variables (no class).
Adjust paths/hyperparameters here and import directly, e.g.:
from config import DATA_DIR, CLASSIFIER_INPUT_SIZE
"""

# --- Data and Model Parameters ---
DATA_DIR: str = "../Datasets/Raw/TumorData"  # Directory containing the .mat files
NUM_CLASSES_CLASSIFIER: int = 2  # Binary classification: Meningioma (0) vs Glioma (1)
CLASSIFIER_INPUT_SIZE: int = 128
ORIGINAL_IMG_SIZE: int = 512

# --- Training Hyperparameters ---
CLASSIFIER_LR: float = 0.001       # Learning rate for the classifier
CLASSIFIER_EPOCHS: int = 25        # Number of epochs for classifier training
BATCH_SIZE: int = 4                # Batch size for training
NUM_FOLDS: int = 5                 # Number of folds for K-Fold Cross-Validation

# --- Model Saving Parameters ---
MODEL_SAVE_DIR: str = "saved_models"               # Directory to save best models
CLASSIFIER_MODEL_NAME: str = "best_classifier.pth"  # Name for best classifier model