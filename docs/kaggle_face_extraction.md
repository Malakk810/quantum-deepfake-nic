# Face extraction on Kaggle

Goal: face crops for FF++ c23 (train, val, test) and the Celeb-DF test list, saved as a private Kaggle dataset.

Two notebooks:
- **A, CPU:** downloads FF++ c23 and saves it as a private dataset, so the GPU is never idle during the download and you never download twice.
- **B, GPU:** cuts out the faces and saves them as a second private dataset.

Expected result: about 107,000 crops, roughly 10 GB. Time: in the 10-video test (T4, 23 Oct 2026) a training video took about 1 s and an evaluation video about 1.7 s, so the full notebook B run takes about 2 to 2.5 hours.

Neither dataset may be redistributed. Keep every Kaggle dataset **private** and share it only with your teammate.

## Part 1: once, on your Mac

- [ ] **Copy the 518 Celeb-DF test videos into their own folder and zip it.** In the VS Code terminal, from `NIC_Quantum_Project`:

  ```bash
  cd Celeb-DF-v2
  mkdir -p ../celebdf_test
  awk '{print $2}' List_of_testing_videos.txt | while read p; do mkdir -p "../celebdf_test/$(dirname "$p")"; cp "$p" "../celebdf_test/$p"; done
  cp List_of_testing_videos.txt ../celebdf_test/
  cd ..
  find celebdf_test -name "*.mp4" | wc -l
  zip -r -q celebdf_test.zip celebdf_test
  ```

  The `find` line must print **518**.

- [ ] **Put the code files in one place.** You need `extract_faces.py` and the three split files `ffpp_train.json`, `ffpp_val.json`, `ffpp_test.json` (from `data/splits/` in the repo). Zip them together as `nic_code.zip`.

- [ ] **Turn on GPU and internet on Kaggle.** Your account must be phone-verified for both.

- [ ] **Create the input dataset.** On kaggle.com: Datasets, New Dataset, title `nic-inputs`, **Private**. Upload three files:
  - `celebdf_test.zip`
  - `nic_code.zip`
  - `FaceForensics_Download.py` (your copy from the FF++ email)

  Kaggle unpacks the zips by itself.

## Part 2: notebook A, download FF++ (CPU)

- [ ] **Set up the notebook.** New notebook named `nic-download-ffpp`. Settings: Accelerator **None**, Internet **On**. Add Input: `nic-inputs`.

- [ ] **Cell 1: download.**

  ```python
  import glob, os, subprocess
  CODE = os.path.dirname(glob.glob("/kaggle/input/**/FaceForensics_Download.py", recursive=True)[0])
  for d in ["original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
      !echo "" | python {CODE}/FaceForensics_Download.py /tmp/ffpp -d {d} -c c23 -t videos --server EU2
  ```

  The `echo ""` answers the "press any key" terms-of-use prompt.

- [ ] **Cell 2: check the counts.** Every line must show 1000.

  ```python
  folders = glob.glob("/tmp/ffpp/original_sequences/youtube/c23/videos") + glob.glob("/tmp/ffpp/manipulated_sequences/*/c23/videos")
  for d in sorted(folders):
      print(d, len(glob.glob(d + "/*.mp4")))
  ```

- [ ] **Cell 3: pack into five archives.** Kaggle handles a few big files better than 5,000 small ones.

  ```python
  os.makedirs("/kaggle/working/ffpp_c23", exist_ok=True)
  for d in ["original_sequences/youtube", "manipulated_sequences/Deepfakes", "manipulated_sequences/Face2Face",
            "manipulated_sequences/FaceSwap", "manipulated_sequences/NeuralTextures"]:
      subprocess.run(["tar", "-cf", "/kaggle/working/ffpp_c23/" + os.path.basename(d) + ".tar", "-C", "/tmp/ffpp", d], check=True)
  !ls -lh /kaggle/working/ffpp_c23
  ```

- [ ] **Run it.** Save Version, then **Save & Run All**. It runs in the background, so you can close the tab.

- [ ] **Save the result as a dataset.** When the run finishes, open the version and check that cell 2 printed 1000 five times. Then, from its **Output** tab, create a new **private** dataset named `nic-ffpp-c23`. Kaggle unpacks the five archives into folders by itself (about 9 GB, 5,000 videos); that is expected, and notebook B works with the folders directly.

## Part 3: notebook B, face crops (GPU)

- [ ] **Set up the notebook.** New notebook named `nic-face-crops`. Settings: Accelerator **GPU** (T4 or P100), Internet **On** (needed for pip and the detector weights). Add Inputs: `nic-inputs` and `nic-ffpp-c23`.

- [ ] **Cell 1: setup.** Replace `<your-name>` with your own name.

  ```python
  import glob, os, subprocess, shutil
  os.environ["MACHINE_NAME"] = "kaggle-<your-name>"
  CODE = os.path.dirname(glob.glob("/kaggle/input/**/extract_faces.py", recursive=True)[0])
  CELEB = os.path.dirname(glob.glob("/kaggle/input/**/List_of_testing_videos.txt", recursive=True)[0])
  print(CODE, CELEB, sep="\n")
  !pip install -q batch-face==1.5.4
  !nvidia-smi --query-gpu=name --format=csv,noheader
  ```

  If this cell stops with an `IndexError`, Kaggle did not unpack one of the zips, or an input is missing. Run `!find /kaggle/input -maxdepth 3` and look at what is there.

- [ ] **Cell 2: link the FF++ folders and check the counts.** This links the unpacked video folders into the layout the script expects, without copying anything. Every line must show 1000.

  ```python
  for v in glob.glob("/kaggle/input/**/c23/videos", recursive=True):
      name = v.split("/")[-3]
      if name == "youtube":
          dst = "/tmp/ffpp/original_sequences/youtube/c23/videos"
      else:
          dst = "/tmp/ffpp/manipulated_sequences/" + name + "/c23/videos"
      os.makedirs(os.path.dirname(dst), exist_ok=True)
      if not os.path.exists(dst):
          os.symlink(v, dst)

  folders = glob.glob("/tmp/ffpp/original_sequences/youtube/c23/videos") + glob.glob("/tmp/ffpp/manipulated_sequences/*/c23/videos")
  for d in sorted(folders):
      print(d, len(glob.glob(d + "/*.mp4")))
  ```

- [ ] **Cell 3: cut out the faces.** Keep `LIMIT = 10` for the first, interactive test.

  ```python
  LIMIT = 10   # 10 for the test, 0 for the full run
  for split in ["train", "val", "test"]:
      !python {CODE}/extract_faces.py --dataset ffpp --root /tmp/ffpp --out /tmp/faces --split {split} --splits_dir {CODE} --limit {LIMIT}
  !python {CODE}/extract_faces.py --dataset celebdf --root {CELEB} --out /tmp/faces --limit {LIMIT}
  ```

- [ ] **Cell 4: summary and a look at some crops.**

  ```python
  import pandas as pd
  import matplotlib.pyplot as plt
  from PIL import Image

  for part in ["ffpp/train", "ffpp/val", "ffpp/test", "celebdf/test"]:
      m = pd.read_csv("/tmp/faces/" + part + "/manifest.csv")
      f = pd.read_csv("/tmp/faces/" + part + "/failed_videos.csv")
      n_videos = m.groupby(["method", "video_id"]).ngroups
      print(part, "videos", n_videos, "crops", len(m), "failed", len(f))
      print(m.groupby("label").size().to_dict())

  m = pd.read_csv("/tmp/faces/ffpp/val/manifest.csv")
  sample = m.sample(12, random_state=0)
  fig, axes = plt.subplots(2, 6, figsize=(12, 4.5))
  for ax, (_, r) in zip(axes.ravel(), sample.iterrows()):
      ax.imshow(Image.open("/tmp/faces/" + r["path"]))
      ax.set_title(r["method"] + " " + str(r["label"]), fontsize=8)
      ax.axis("off")
  plt.show()
  ```

- [ ] **Cell 5: pack the crops for saving.**

  ```python
  os.makedirs("/kaggle/working/faces", exist_ok=True)
  for part in ["ffpp/train", "ffpp/val", "ffpp/test", "celebdf/test"]:
      name = part.replace("/", "_")
      subprocess.run(["tar", "-cf", "/kaggle/working/faces/" + name + ".tar", "-C", "/tmp/faces", part], check=True)
      for f in ["manifest.csv", "failed_videos.csv", "runs.jsonl"]:
          shutil.copy("/tmp/faces/" + part + "/" + f, "/kaggle/working/faces/" + name + "_" + f)
  !ls -lh /kaggle/working/faces
  ```

- [ ] **Test run with `LIMIT = 10`** (interactive: Run All). Check these:
  - Cell 1 prints a GPU name.
  - Each extraction run prints `device cuda:0, detector resnet50`.
  - Cell 4 shows faces, centred and not stretched, with no empty frames.
  - Note the minutes printed per split, to estimate the full run.

- [ ] **Full run.** Set `LIMIT = 0`, then Save Version, **Save & Run All**. You can close the tab.

- [ ] **Check the finished run.** Cell 4 should show about:

  | Split | Videos | Crops |
  |---|---|---|
  | ffpp/train | 3,600 | up to 46,080 |
  | ffpp/val | 700 | up to 22,400 |
  | ffpp/test | 700 | up to 22,400 |
  | celebdf/test | 518 | up to 16,576 |

  A few failed videos or missing crops are normal, since frames with no detected face are skipped. More than about 1% failed means something is wrong, so stop and check.

- [ ] **Save the crops.** From the version's **Output** tab, create a new **private** dataset named `nic-faces-v1`. Share it with your teammate (dataset Settings, Sharing).

- [ ] **Send the summary.** Copy cell 4's printed lines and the failed counts into the chat. The next step is CLIP features from these crops.

## Fallback: your Mac

The same `extract_faces.py` commands run on the Mac's CPU, with no flags to change. Budget a whole night or more, and use the same `--network` for every split.
