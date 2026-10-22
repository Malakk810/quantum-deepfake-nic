# quantum-deepfake-nic

DSAI 403 (Nature-Inspired Computation) course project, idea 41.
Nature-inspired architecture search for quantum deepfake image classifiers.

## Structure

| Folder | Contents | In git? |
|---|---|---|
| `data/raw/` | Downloaded datasets (140k faces, FF++, Celeb-DF) | No |
| `data/splits/` | Fixed train/val/test index files | Yes |
| `features/` | Cached CLIP embeddings and PCA outputs (.npy) | No |
| `src/` | All code | Yes |
| `experiments/` | One config per experiment | Yes |
| `results/` | Run logs (CSV), figures, tables | Yes, except checkpoints |
| `docs/` | Research plan, hypotheses, notes | Yes |

Large files (`data/raw/`, `features/*.npy`, checkpoints) stay out of git.
Keep the cached features in one shared place (a private Kaggle dataset or a shared Drive folder)
so every machine loads identical files.

## Setup

```
pip install -r requirements.txt
```

Kaggle and Colab come with torch, numpy and scikit-learn preinstalled, so those are not pinned.
Every run logs their versions instead (see `src/utils.py`).

## Rules for every experiment

- Call `set_seed(seed)` at the start of every script. Final runs use seeds 0, 1, 2, 3, 4.
- Splits are created once and committed to `data/splits/`. Never regenerate them.
- Fit PCA and any scaler on the training split only.
- The test split is used once per final model, never during search or tuning.
- Set `MACHINE_NAME` (for example `kaggle-malak`, `laptop-malak`) so every logged row records where it ran.
- Every run appends one row to a CSV in `results/` with config, seed, machine, metrics and wall-clock time.

## Datasets

- 140k Real and Fake Faces (Kaggle), for Phase 1
- FaceForensics++ (c23), main benchmark, access by request
- Celeb-DF v2, cross-dataset test, access by request
