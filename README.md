# quantum-deepfake-nic

Nature-inspired architecture search for quantum deepfake image classifiers.

## Structure

| Folder | Contents |
|---|---|
| `data/splits/` | Official FF++ train/val/test lists |
| `notebooks/` | Every notebook as run on Kaggle, with its outputs |
| `results/` | Small files each notebook wrote, one folder per notebook |
| `src/` | Face extraction script, seed and logging helpers |
| `docs/` | Kaggle face extraction checklist |

Videos, face crops and features are not on git. They are in private Kaggle datasets.

## Setup

```
pip install -r requirements.txt
```

## Rules for every experiment

- Call `set_seed(seed)` at the start of every script. Final runs use seeds 0, 1, 2, 3, 4.
- Splits are created once and committed to `data/splits/`. Don't regenerate them.
- Fit PCA and any scaler on the training split only.
- The test split is used once per final model, not during search or tuning.
- Set `MACHINE_NAME` so every logged row records where it ran.
- Every run appends one row to a CSV in `results/` with config, seed, machine, metrics and wall-clock time.

## Step 1: face extraction

Samples frames from each video, detects the largest face with RetinaFace (batch-face, ResNet-50 backbone),
crops a square box 1.3x the face size, resizes to 224x224 and saves PNG.
Runs on a GPU when one is available.
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

Use the same `--network` for every split. `--workers` sets how many videos are decoded in parallel
while the detector runs (default 4).

Expected size: about 107,000 crops (FF++ train 46,080, val 22,400, test 22,400; Celeb-DF test 16,576),
roughly 10 GB of PNG.

## Notebooks

Every notebook in `notebooks/` is the executed copy from Kaggle, with its outputs kept as the record of the run.
The small result files each one wrote (CSV, JSON, figures) are in `results/<notebook name>/`.
Large outputs (videos, face crops, features) are in private Kaggle datasets.
Run them in this order:

| # | Notebook | Kaggle hardware, run time | Reads | Writes |
|---|---|---|---|---|
| 1 | `download_faceforensics.ipynb` | CPU, 2 h 31 min | the FF++ download script | FF++ c23 videos |
| 2 | `face_crops.ipynb` | T4, 1 h 38 min | FF++ videos, Celeb-DF test videos | 224x224 face crops |
| 3 | `clip_features.ipynb` | T4, 40 min | face crops | 512-d CLIP features per crop |
| 4 | `feature_prep.ipynb` | 39 s | CLIP features | PCA to 32, angles, inner fitness split, bottleneck check |
| 5 | `classical_baselines.ipynb` | 1.5 min | CLIP features, prepared features | classical baselines |
| 6 | `first_quantum_circuit.ipynb` | 7 min | prepared features | first 8-qubit circuit: ring vs no entangler, learning rates, time per depth |
| 7 | `hand_designed_grid.ipynb` | 47 min | prepared features | 9-circuit grid, ranking check, `recipe.json` |
| 8 | `pls_preview.ipynb` | 18 min | CLIP features, prepared features | grid circuits on 8 PLS components |
| 9 | `fairness_check.ipynb` | 15 min | CLIP features, prepared features | entanglement fairness check |
| 10 | `qcnn_baseline.ipynb` | 17.5 min | CLIP features, prepared features | QCNN baseline |
| 11 | `ga_vs_random_pilot.ipynb` | not recorded | prepared features | pilot GA and random search, 1 run of 100 circuits each |
| 12 | `GA_vs_RandomSearch.ipynb` | 1 h 40 min | prepared features | GA and random search, 2 runs of 300 circuits each |
| 13 | `PSO_ACO_GWO.ipynb` | 2 h 20 min | prepared features | PSO, ACO and GWO, 2 runs of 300 circuits each |
| 14 | `clip_features_l14.ipynb` | T4 x2, 1 h 33 min | face crops | 768-d CLIP ViT-L/14 features per crop |
| 15 | `feature_prep_l14.ipynb` | 44 s | ViT-L/14 features | same as notebook 4 on the ViT-L/14 features |
| 16 | `precheck_b16.ipynb` | 5 min | CLIP features, prepared features | classical pre-check: PLS 8/16/32, 16 inputs on 8 qubits, 8 of 512 raw CLIP dims picked by ACO, greedy and random search |
| 17 | `precheck_l14.ipynb` | 9 min | ViT-L/14 features, prepared ViT-L/14 features | the same pre-check on ViT-L/14 (768 raw dims) |
| 18 | `interaction_check.ipynb` | 52 min | CLIP features, prepared features | additive vs interaction models, all methods and NeuralTextures held out |
| 19 | `effnet_b4_features.ipynb` | T4, 1 h | face crops | 1792-d frozen EfficientNet-B4 features per crop, same rows as CLIP |
| 20 | `pool_prep.ipynb` | 20 min | B/16, L/14 and EfficientNet-B4 features | input pool for three backbones: 128 PCA + 16 PLS as angles, column strength, bottleneck table on val and with one manipulation held out (PLS refit per fold, C tuned on the inner split) |
| 21 | `ranking_check.ipynb` | 5 min | input pool | 60 genomes on the L/14 pool, real circuits (3 seeds) against the classical twin: Spearman 0.996, seed-to-seed 0.997 |
| 22 | `genome_search.ipynb` | 15.6 min | input pool | stage 1 search on all three backbones: random search, GA, PSO and memetic GA (10 runs each), DE and GWO (5 runs each), 1,000 twin evaluations per run |
| 23 | `search_analysis.ipynb` | 5 min | input pool, genome search output | baselines (first 8 PCA, strongest 8, PLS 8, MI top 8, greedy), val video AUC of each run's top genome with the twin, Kruskal-Wallis, Holm-corrected Mann-Whitney and A12, top 3 genomes per run |
| 24 | `genome_search_ablation.ipynb` | 4.9 min | input pool, ViT-L/14 features | exploratory ablation: GA on L/14 with a held-out-manipulation term in the fitness (PLS refit per fold), 10 runs |
| 25 | `final_circuits.ipynb` | 2 h 41 min | input pool, search analysis output | real 8-qubit circuits: the top genome of every run and the baselines (seed 0), then 7 finalists per backbone with 5 seeds, with and without a CZ ring; twin against circuit on val |
| 26 | `context_table.ipynb` | 15 min | input pool, raw features, search outputs | twin, logistic regression and a 4-unit MLP on the same 8 inputs, on val and with one manipulation held out |
| 27 | `single_test.ipynb` | about 25 min | final circuits output, input pool, raw features | the single test: frozen finalists and the linear probe on FF++ test and Celeb-DF |
| 28 | `paper_figures.ipynb` | under 1 min | single test, context table and final circuits outputs | the paper's tables (CSV, LaTeX, Markdown) and figures |

The two searches (notebooks 12 and 13) use short-training learning rate 0.03, tied with 0.1 in the ranking check.
GWO stopped early in both runs (260 and 240 circuits), when no new circuits came up for 20 rounds.

The ranking check in notebook 7 picked learning rate 0.1 with 6 epochs for short training.
It is tied with 0.03 at 6 epochs (mean within-input Spearman 0.692 against 0.683), and the tie went
to the faster setting by 0.07 s per evaluation; an earlier run picked 0.03 the same way.

Notebooks 16 and 17 are one notebook run twice, with `encoder` set to `"b16"` and `"l14"`.
Their raw search logs (`raw_search_log.csv`, 2.5 and 3.9 MB) are left out of `results/` and stay in the Kaggle output.

Notebook 24 is `genome_search.ipynb` run with `mode = "ablation"`.
Notebook 22 scores genomes with the classical twin only; validation, test and real circuits come later.
Its fitness stores (13 MB per backbone) and full search log (33 MB) stay in the Kaggle output.

## Datasets

- FaceForensics++ (c23), main benchmark, access by request. Official splits in `data/splits/`.
- Celeb-DF v2, cross-dataset test only, using its official test list (518 videos), access by request.
