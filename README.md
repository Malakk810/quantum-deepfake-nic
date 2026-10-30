# quantum-deepfake-nic

DSAI 403 (Nature-Inspired Computation) course project, idea 41.
Nature-inspired architecture search for quantum deepfake image classifiers.

## Structure

| Folder | Contents | In git? |
|---|---|---|
| `data/raw/` | Downloaded datasets (FF++, Celeb-DF) | No |
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

## Step 1: face extraction

Samples frames from each video, detects the largest face with RetinaFace (batch-face, ResNet-50 backbone),
crops a square box 1.3x the face size, resizes to 224x224 and saves PNG.
Runs on a GPU when one is available, otherwise on the CPU (much slower).
For the full run on Kaggle, follow `docs/kaggle_face_extraction.md`.

Frames per video: train uses 32 per real video and 8 per fake video (FF++ has 4 fakes per real,
so this balances the classes). Val and test use 32 evenly spaced frames from every video.
Frame choice is deterministic (evenly spaced), so reruns give the same frames.

```
# quick check on 10 videos first
python src/extract_faces.py --dataset ffpp --root /path/to/FaceForensics++ --out /path/to/faces --split train --limit 10

# full runs
python src/extract_faces.py --dataset ffpp --root /path/to/FaceForensics++ --out /path/to/faces --split train
python src/extract_faces.py --dataset ffpp --root /path/to/FaceForensics++ --out /path/to/faces --split val
python src/extract_faces.py --dataset ffpp --root /path/to/FaceForensics++ --out /path/to/faces --split test
python src/extract_faces.py --dataset celebdf --root /path/to/Celeb-DF-v2 --out /path/to/faces
```

Output, per split: `faces/<dataset>/<split>/<method>/<video_id>/<frame>.png`, plus
`manifest.csv` (one row per crop), `failed_videos.csv`, and `runs.jsonl`
(one line per run: settings, machine, versions, time).
Labels everywhere: 0 = real, 1 = fake.

If a session dies, rerun the same command. Videos that already have their `rows.csv` are skipped.

Use the same `--network` for every split. `--workers` sets how many videos are decoded in parallel
while the detector runs (default 4).

Expected size: about 107,000 crops (FF++ train 46,080, val 22,400, test 22,400; Celeb-DF test 16,576),
roughly 10 GB of PNG.

## Phase 1 notebooks

Every notebook in `notebooks/` is the executed copy from Kaggle, with its outputs kept as the record of the run.
The small result files each one wrote (CSV, JSON, figures) are in `results/<notebook name>/`.
Large outputs (videos, face crops, features) stay in private Kaggle datasets.
Run them in this order:

| # | Notebook | Kaggle hardware, run time | Reads | Writes |
|---|---|---|---|---|
| 1 | `download_faceforensics.ipynb` | CPU, 2 h 31 min | the FF++ download script | FF++ c23 videos |
| 2 | `face_crops.ipynb` | T4, 1 h 38 min | FF++ videos, Celeb-DF test videos | 224x224 face crops |
| 3 | `clip_features.ipynb` | T4, 40 min | face crops | 512-d CLIP features per crop |
| 4 | `feature_prep.ipynb` | 39 s | CLIP features | PCA to 32, angles, inner fitness split, bottleneck check |
| 5 | `classical_baselines.ipynb` | 1.5 min | CLIP features, prepared features | classical baselines |
| 6 | `hand_designed_grid.ipynb` | 47 min | prepared features | 9-circuit grid, ranking check, `recipe.json` |
| 7 | `pls_preview.ipynb` | 18 min | CLIP features, prepared features | grid circuits on 8 PLS components |
| 8 | `fairness_check.ipynb` | 15 min | CLIP features, prepared features | entanglement fairness check |
| 9 | `qcnn_baseline.ipynb` | 17.5 min | CLIP features, prepared features | QCNN baseline |
| 10 | `more_search.ipynb` | 2 h 20 min | prepared features | PSO, ACO and GWO, 2 runs of 300 circuits each |

Still to come: the GA and random search rerun. Its result files and those of `more_search.ipynb` stay on Kaggle for now.
The Phase 1 searches use short-training learning rate 0.03 (tied with 0.1 in the ranking check); Phase 2 uses 0.1.
GWO stopped early in both runs (260 and 240 circuits), when no new circuits came up for 20 rounds.

The ranking check in notebook 6 picked learning rate 0.1 with 6 epochs for short training.
It is tied with 0.03 at 6 epochs (mean within-input Spearman 0.692 against 0.683), and the tie went
to the faster setting by 0.07 s per evaluation; an earlier run picked 0.03 the same way.

## Datasets

- FaceForensics++ (c23), main benchmark, access by request. Official splits in `data/splits/`.
- Celeb-DF v2, cross-dataset test only, using its official test list (518 videos), access by request.
- Neither dataset may be redistributed: keep any copies (for example Kaggle datasets) private.
