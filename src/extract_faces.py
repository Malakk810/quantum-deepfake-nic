import os
import csv
import json
import time
import socket
import argparse
import platform
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version

import cv2
import numpy as np
import torch
from tqdm import tqdm
from batch_face import RetinaFace


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FFPP_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
ROW_FIELDS = ["dataset", "split", "method", "video_id", "label", "frame_idx", "path", "det_score", "x1", "y1", "x2", "y2"]


# ---------- video lists ----------
# label convention for the whole project: 0 = real, 1 = fake

def list_ffpp_videos(root, split, compression, methods, splits_dir):
    with open(os.path.join(splits_dir, "ffpp_" + split + ".json")) as f:
        pairs = json.load(f)

    videos = []
    real_dir = os.path.join(root, "original_sequences", "youtube", compression, "videos")
    for a, b in pairs:
        for vid in [a, b]:
            videos.append({
                "path": os.path.join(real_dir, vid + ".mp4"),
                "label": 0,
                "method": "original",
                "video_id": vid,
            })
        for method in methods:
            fake_dir = os.path.join(root, "manipulated_sequences", method, compression, "videos")
            for vid in [a + "_" + b, b + "_" + a]:
                videos.append({
                    "path": os.path.join(fake_dir, vid + ".mp4"),
                    "label": 1,
                    "method": method,
                    "video_id": vid,
                })
    return videos


def list_celebdf_test_videos(root):
    # each line of the official test list looks like: "1 YouTube-real/00170.mp4"
    # the label is taken from the folder name so we do not depend on the file's label convention
    videos = []
    with open(os.path.join(root, "List_of_testing_videos.txt")) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            rel_path = parts[1]
            folder = rel_path.split("/")[0]
            stem = os.path.splitext(os.path.basename(rel_path))[0]
            videos.append({
                "path": os.path.join(root, rel_path),
                "label": 1 if "synthesis" in folder.lower() else 0,
                "method": folder,
                "video_id": folder + "_" + stem,
            })
    return videos


# ---------- frames ----------

def frames_for_video(label, split, args):
    if split == "train":
        return args.train_real_frames if label == 0 else args.train_fake_frames
    return args.eval_frames


def read_frames(video_path, n_frames):
    # evenly spaced frame indices, read sequentially because seeking in h264 is unreliable
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return [], []

    wanted = np.unique(np.linspace(0, total - 1, min(n_frames, total)).round().astype(int))
    wanted_set = set(wanted.tolist())
    last = int(wanted[-1])

    frames = []
    indices = []
    for i in range(last + 1):
        ok = cap.grab()
        if not ok:
            break
        if i in wanted_set:
            ok, frame = cap.retrieve()
            if ok:
                frames.append(frame)
                indices.append(i)
    cap.release()
    return frames, indices


# ---------- faces ----------

def square_crop_box(box, img_w, img_h, margin):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * margin
    side = min(side, img_w, img_h)

    # shift the box inside the image instead of clipping it, so the crop stays square
    left = cx - side / 2.0
    top = cy - side / 2.0
    left = min(max(left, 0), img_w - side)
    top = min(max(top, 0), img_h - side)
    return int(round(left)), int(round(top)), int(round(left + side)), int(round(top + side))


def largest_face(faces):
    best = None
    best_area = 0
    for face in faces:
        x1, y1, x2, y2 = face["box"]
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best = face
            best_area = area
    return best


def process_video(video, frames, indices, split, dataset, detector, args):
    if len(frames) == 0:
        return None, "no frames read"

    img_h, img_w = frames[0].shape[:2]
    # batch-face expects RGB, as in its own examples
    rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    # max_size no larger than the frame, otherwise batch-face prints a warning on every call
    max_size = min(args.det_size, max(img_h, img_w))
    all_faces = detector(rgb_frames, threshold=args.threshold, max_size=max_size,
                         batch_size=args.batch_size, return_dict=True)

    rel_dir = os.path.join(dataset, split, video["method"], video["video_id"])
    out_dir = os.path.join(args.out, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for frame, frame_idx, faces in zip(frames, indices, all_faces):
        face = largest_face(faces)
        if face is None:
            continue
        left, top, right, bottom = square_crop_box(face["box"], img_w, img_h, args.margin)
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            continue
        interp = cv2.INTER_AREA if crop.shape[0] > args.size else cv2.INTER_LINEAR
        crop = cv2.resize(crop, (args.size, args.size), interpolation=interp)

        name = "%04d.png" % frame_idx
        cv2.imwrite(os.path.join(out_dir, name), crop)
        rows.append({
            "dataset": dataset,
            "split": split,
            "method": video["method"],
            "video_id": video["video_id"],
            "label": video["label"],
            "frame_idx": frame_idx,
            "path": os.path.join(rel_dir, name),
            "det_score": round(float(face["score"]), 4),
            "x1": left, "y1": top, "x2": right, "y2": bottom,
        })

    if len(rows) == 0:
        return None, "no face detected"

    # rows.csv is written last, so a video with rows.csv is complete and can be skipped on resume
    with open(os.path.join(out_dir, "rows.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows, None


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ffpp", "celebdf"], required=True)
    parser.add_argument("--root", required=True, help="dataset root folder")
    parser.add_argument("--out", required=True, help="folder where face crops are saved")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--compression", default="c23")
    parser.add_argument("--methods", nargs="+", default=FFPP_METHODS)
    parser.add_argument("--splits_dir", default=os.path.join(REPO_ROOT, "data", "splits"))
    parser.add_argument("--train_real_frames", type=int, default=32)
    parser.add_argument("--train_fake_frames", type=int, default=8)
    parser.add_argument("--eval_frames", type=int, default=32)
    parser.add_argument("--network", choices=["resnet50", "mobilenet"], default="resnet50",
                        help="detector backbone; use the same one for every split")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--margin", type=float, default=1.3)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--det_size", type=int, default=640, help="frames are resized to this max side for detection only")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4, help="videos decoded in parallel while the detector runs")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="only process the first N videos, for testing")
    args = parser.parse_args()

    if args.dataset == "ffpp":
        split = args.split
        videos = list_ffpp_videos(args.root, split, args.compression, args.methods, args.splits_dir)
    else:
        split = "test"
        videos = list_celebdf_test_videos(args.root)
    if args.limit > 0:
        videos = videos[:args.limit]

    gpu_id = 0 if (torch.cuda.is_available() and not args.cpu) else -1
    device = "cuda:0" if gpu_id == 0 else "cpu"
    detector = RetinaFace(gpu_id=gpu_id, network=args.network)

    split_dir = os.path.join(args.out, args.dataset, split)
    os.makedirs(split_dir, exist_ok=True)
    failed = []
    n_skipped = 0

    todo = []
    for video in videos:
        rows_file = os.path.join(split_dir, video["method"], video["video_id"], "rows.csv")
        if os.path.exists(rows_file):
            n_skipped += 1
        elif not os.path.exists(video["path"]):
            failed.append({"path": video["path"], "reason": "missing file"})
        else:
            todo.append(video)

    print("%s %s: %d videos, %d already done, %d to process, device %s, detector %s"
          % (args.dataset, split, len(videos), n_skipped, len(todo), device, args.network))

    # decoding is the slow part, so the next few videos are read in background threads
    # while the detector works on the current one; results are still used in list order
    start = time.time()
    n_done = 0
    lookahead = max(1, args.workers * 2)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        pending = {}
        next_i = 0
        for i in tqdm(range(len(todo)), desc=args.dataset + " " + split):
            while next_i < len(todo) and next_i < i + lookahead:
                video = todo[next_i]
                n_frames = frames_for_video(video["label"], split, args)
                pending[next_i] = pool.submit(read_frames, video["path"], n_frames)
                next_i += 1
            video = todo[i]
            try:
                frames, indices = pending.pop(i).result()
                rows, error = process_video(video, frames, indices, split, args.dataset, detector, args)
            except Exception as e:
                rows, error = None, "error: " + str(e)[:200]
            if error is not None:
                failed.append({"path": video["path"], "reason": error})
            else:
                n_done += 1
    seconds = time.time() - start

    # one manifest per split, rebuilt from the per-video files so it is always consistent
    n_frames = 0
    n_videos = 0
    with open(os.path.join(split_dir, "manifest.csv"), "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for video in videos:
            rows_file = os.path.join(split_dir, video["method"], video["video_id"], "rows.csv")
            if not os.path.exists(rows_file):
                continue
            n_videos += 1
            with open(rows_file) as f_in:
                for row in csv.DictReader(f_in):
                    writer.writerow(row)
                    n_frames += 1

    with open(os.path.join(split_dir, "failed_videos.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "reason"])
        writer.writeheader()
        writer.writerows(failed)

    # one line per run, appended, so resumed runs keep the history of earlier ones
    info = {
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "machine": os.environ.get("MACHINE_NAME", socket.gethostname()),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if gpu_id == 0 else "",
        "seconds": round(seconds, 1),
        "processed": n_done,
        "skipped": n_skipped,
        "failed": len(failed),
        "videos_in_manifest": n_videos,
        "crops_in_manifest": n_frames,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "batch_face": version("batch-face"),
        "args": vars(args),
    }
    with open(os.path.join(split_dir, "runs.jsonl"), "a") as f:
        f.write(json.dumps(info) + "\n")

    print("done %d, skipped %d, failed %d, videos in manifest %d, crops %d, %.1f min"
          % (n_done, n_skipped, len(failed), n_videos, n_frames, seconds / 60))


if __name__ == "__main__":
    main()
