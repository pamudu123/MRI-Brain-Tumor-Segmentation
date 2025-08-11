# MRI Brain Tumor Segmentation (Paper Implementations)

Minimal, unified repo to reproduce paper-based experiments on MRI brain tumor classification/localization/segmentation.

## Repo Structure
- `paper_implementation/` — Implementation and K-Fold runner for arXiv:2102.03532v1 (binary classifier). Uses module-level config.
- `experiments/` — Segmentation/localization scaffold for "[Pamudu Ranasinghe] MRI Brain Tumor Localization and Segmentation". Training code is commented out by default.
- `docs/` — Minimal documentation. `implementation_log.md` is the single source of logic and progress; `workflow.md` contains high-level task notes.
- `Datasets/`, `brainTumorDataPublic_1766/`, `RCNN/` — Data and legacy content (ignored by Git).

## Python Requirements
Install scientific dependencies. (PyTorch is environment/GPU specific; see below.)

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:
- numpy
- h5py
- opencv-python
- scikit-learn

## Installing PyTorch (pick one)
- CPU only:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
- CUDA 12.1 (most common on recent NVIDIA drivers):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
- CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Running (disabled by default)
- Classifier (arXiv:2102.03532v1):
  - Entry: `paper_implementation/runner.py`
  - Do not run unless you explicitly want to train.
- Segmentation (Ranasinghe):
  - Entry: `experiments/runner.py`
  - K-Fold training loop is commented to avoid accidental runs. Un-comment once specs are confirmed.

## Notes
- Update `DATA_DIR` in the respective `config.py` to point to your `.mat` files.
- See `docs/implementation_log.md` for progress and decisions.
