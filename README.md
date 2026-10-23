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

## Datasets

- FaceForensics++ (c23), main benchmark, access by request. Official splits in `data/splits/`.
- Celeb-DF v2, cross-dataset test only, using its official test list (518 videos), access by request.
- Neither dataset may be redistributed: keep any copies (for example Kaggle datasets) private.
