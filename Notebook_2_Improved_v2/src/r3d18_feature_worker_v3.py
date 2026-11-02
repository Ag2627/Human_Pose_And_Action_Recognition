# Worker starts in a fresh process. Preload PyTorch's bundled OpenMP runtime
# before importing torch/cv2/torchvision. No package or system files are changed.
import os, sys, ctypes, json, time, hashlib, traceback, gc
from pathlib import Path

TORCH_LIB = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
_dll_handles = []
_openmp_handle = None
if os.name == "nt":
    torch_iomp = TORCH_LIB / "libiomp5md.dll"
    if torch_iomp.is_file():
        try:
            _openmp_handle = ctypes.WinDLL(str(torch_iomp))
            print("OPENMP_PRELOADED_FROM", torch_iomp, flush=True)
        except OSError as exc:
            print("OPENMP_PRELOAD_WARNING", repr(exc), flush=True)
if TORCH_LIB.is_dir():
    os.environ["PATH"] = str(TORCH_LIB) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            _dll_handles.append(os.add_dll_directory(str(TORCH_LIB)))
        except OSError:
            pass
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import cv2
import torch
from torchvision.models.video import r3d_18, R3D_18_Weights


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


def output_for(row, out_dir):
    key = str(Path(row["video_path"]).resolve()).casefold()
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return Path(out_dir) / f"video_{int(row['video_id']):06d}_{digest}.npz"


def validate_existing(path, row):
    try:
        with np.load(path, allow_pickle=False) as d:
            return (
                d["feature"].shape == (512,)
                and int(d["video_id"]) == int(row["video_id"])
                and str(d["video_path"].item()) == str(row["video_path"])
                and str(d["split"].item()) == str(row["split"])
                and np.isfinite(d["feature"]).all()
            )
    except Exception:
        return False


def resize_center_crop(rgb, short_side=128, crop=112):
    h, w = rgb.shape[:2]
    if h < 1 or w < 1:
        raise ValueError("Invalid decoded frame dimensions")
    scale = float(short_side) / float(min(h, w))
    nw = max(crop, int(round(w * scale)))
    nh = max(crop, int(round(h * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x0 = max(0, (nw - crop) // 2)
    y0 = max(0, (nh - crop) // 2)
    return resized[y0:y0 + crop, x0:x0 + crop]


def extract_clip(video_path, num_frames):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count < 1:
            raise RuntimeError("Video reports zero frames")
        indices = np.linspace(0, max(frame_count - 1, 0), int(num_frames)).round().astype(np.int64)
        frames = []
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, bgr = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(resize_center_crop(rgb))
        if len(frames) < max(8, int(num_frames) // 2):
            raise RuntimeError(f"Too few readable sampled frames: {len(frames)}")
        return np.asarray(frames, dtype=np.uint8), frame_count
    finally:
        cap.release()


def main(cfg_path):
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["rgb_dir"])
    log_dir = Path(cfg["log_dir"])
    per_video_dir = Path(cfg["per_video_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    per_video_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(cfg["selected_csv"])
    total = len(selected)
    state_path = Path(cfg["state_path"])
    start = time.time()
    success = skipped = failed = 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment. GPU RGB extraction was not started.")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA device was detected.")
    device = torch.device(cfg.get("device", "cuda:0"))
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cv2.setNumThreads(1)
    print("RGB_WORKER_START", "videos", total, "device", device, flush=True)
    print("GPU", torch.cuda.get_device_name(device), flush=True)
    print("Loading pretrained R3D-18 Kinetics-400 weights...", flush=True)
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    print("R3D18_READY", "device", device, "feature_dim", 512, flush=True)

    atomic_json(state_path, {"status": "running", "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
                             "total_videos": total, "completed": 0, "success": 0, "skipped_existing": 0,
                             "failed": 0, "current_video": None, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")})

    for pos, row in enumerate(selected.to_dict("records"), 1):
        video_path = Path(str(row["video_path"]))
        out_path = output_for(row, out_dir)
        result_path = per_video_dir / f"result_{int(row['video_id']):06d}.json"
        common = {"video_id": int(row["video_id"]), "video_path": str(video_path),
                  "label": str(row["label"]), "split": str(row["split"]), "output_path": str(out_path)}
        try:
            if out_path.is_file() and validate_existing(out_path, row):
                skipped += 1
                obj = {**common, "status": "skipped_existing", "feature_dim": 512, "device": str(device)}
            else:
                if not video_path.is_file():
                    raise FileNotFoundError(f"Source video not found: {video_path}")
                frames, frame_count = extract_clip(video_path, int(cfg["num_frames"]))
                # [T,H,W,C] -> [1,C,T,H,W], matching torchvision R3D-18 input.
                x = torch.from_numpy(frames).to(device=device, dtype=torch.float32)
                x = x.permute(3, 0, 1, 2).unsqueeze(0).div_(255.0)
                mean = torch.tensor([0.43216, 0.394666, 0.37645], device=device).view(1, 3, 1, 1, 1)
                std = torch.tensor([0.22803, 0.22145, 0.216989], device=device).view(1, 3, 1, 1, 1)
                x = (x - mean) / std
                with torch.inference_mode():
                    feat = model(x).detach().float().cpu().numpy().reshape(-1).astype(np.float32)
                if feat.shape != (512,) or not np.isfinite(feat).all():
                    raise RuntimeError(f"Invalid R3D feature output: shape={feat.shape}")
                tmp = out_path.with_name(out_path.stem + f".{os.getpid()}.tmp.npz")
                np.savez_compressed(tmp, feature=feat,
                                    video_path=np.asarray(str(video_path)),
                                    label=np.asarray(str(row["label"])),
                                    video_id=np.asarray(int(row["video_id"]), dtype=np.int64),
                                    split=np.asarray(str(row["split"])),
                                    sampled_frames=np.asarray(len(frames), dtype=np.int32),
                                    decoded_frames=np.asarray(frame_count, dtype=np.int32),
                                    backbone=np.asarray("torchvision_r3d_18_kinetics400"),
                                    device=np.asarray(str(device)))
                os.replace(str(tmp), str(out_path))
                success += 1
                obj = {**common, "status": "success", "feature_dim": 512,
                       "sampled_frames": int(len(frames)), "decoded_frames": int(frame_count), "device": str(device)}
                del x, frames, feat
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            atomic_json(result_path, obj)
        except Exception as exc:
            failed += 1
            obj = {**common, "status": "failed", "error": repr(exc), "device": str(device)}
            atomic_json(result_path, obj)
            print("VIDEO_FAILED", json.dumps(obj), flush=True)

        completed = success + skipped + failed
        state = {"status": "running", "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
                 "total_videos": total, "completed": completed, "success": success,
                 "skipped_existing": skipped, "failed": failed,
                 "current_video": video_path.name, "current_label": str(row["label"]),
                 "elapsed_seconds": round(time.time() - start, 1),
                 "last_update": time.strftime("%Y-%m-%d %H:%M:%S")}
        atomic_json(state_path, state)
        if completed % 25 == 0 or completed == total:
            print("RGB_PROGRESS", json.dumps(state), flush=True)

    rows = []
    for p in sorted(per_video_dir.glob("result_*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    if rows:
        pd.DataFrame(rows).sort_values("video_id").to_csv(Path(cfg["report_path"]), index=False)
    final_status = "completed_with_errors" if failed else "completed"
    final = {"status": final_status, "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
             "total_videos": total, "completed": success + skipped + failed, "success": success,
             "skipped_existing": skipped, "failed": failed,
             "elapsed_seconds": round(time.time() - start, 1),
             "report_path": str(cfg["report_path"])}
    atomic_json(state_path, final)
    print("RGB_FEATURE_EXTRACTION_FINISHED", json.dumps(final), flush=True)


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except BaseException as exc:
        cfg = {}
        try:
            cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        except Exception:
            pass
        state_path = Path(cfg["state_path"]) if cfg.get("state_path") else None
        error = {"status": "worker_failed", "error": repr(exc), "traceback": traceback.format_exc(),
                 "failed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        print("RGB_WORKER_FATAL", json.dumps(error), flush=True)
        if state_path:
            try:
                atomic_json(state_path, error)
            except Exception:
                pass
        traceback.print_exc()
        sys.exit(2)
