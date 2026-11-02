import os, sys, json, time, gc, traceback, ctypes
from pathlib import Path

# Windows: prefer the OpenMP DLL bundled with this Python's PyTorch before
# importing numerical packages. This is process-local and does not change Anaconda.
torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
_dll_handles = []
_openmp_handle = None
if os.name == "nt":
    torch_iomp = torch_lib / "libiomp5md.dll"
    if torch_iomp.is_file():
        try:
            _openmp_handle = ctypes.WinDLL(str(torch_iomp))
            print("OPENMP_PRELOADED_FROM", str(torch_iomp), flush=True)
        except OSError as exc:
            print("OPENMP_PRELOAD_WARNING", repr(exc), flush=True)
if torch_lib.is_dir():
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            _dll_handles.append(os.add_dll_directory(str(torch_lib)))
        except OSError:
            pass
for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[name] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import torch
import cv2
from ultralytics import YOLO

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
cv2.setNumThreads(1)

project = Path(cfg["project_dir"])
sys.path.insert(0, str(project))
from src.vitpose_model import ViTPose, get_affine_transform, decode_heatmaps, IMG_H, IMG_W, IMG_MEAN, IMG_STD

device = torch.device(cfg.get("device", "cuda:0"))
if device.type != "cuda":
    raise RuntimeError(f"GPU-improved worker requires CUDA, but config device is {device}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the worker process")
torch.cuda.set_device(device)
props = torch.cuda.get_device_properties(device)
free_bytes, total_bytes = torch.cuda.mem_get_info(device)
print("GPU_DEVICE", torch.cuda.get_device_name(device), flush=True)
print("GPU_MEMORY_GIB", round(free_bytes / 2**30, 2), "/", round(total_bytes / 2**30, 2), flush=True)
print("BATCH_WORKER_START", "videos", len(cfg["videos"]), "device", device, flush=True)
print("YOLO_WEIGHTS", cfg["yolo_weights"], flush=True)
print("VITPOSE_CHECKPOINT", cfg["vitpose_checkpoint"], flush=True)

# Load each network once per batch, rather than once per video.
yolo = YOLO(cfg["yolo_weights"])
pose = ViTPose().to(device)
try:
    checkpoint = torch.load(cfg["vitpose_checkpoint"], map_location=device, weights_only=True)
except TypeError:
    checkpoint = torch.load(cfg["vitpose_checkpoint"], map_location=device)
if "model_state" not in checkpoint:
    raise KeyError("ViTPose checkpoint has no 'model_state' key")
pose.load_state_dict(checkpoint["model_state"], strict=True)
pose.eval()
print("BATCH_MODELS_LOADED", flush=True)


def box_iou(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(area_a + area_b - inter, 1e-8)


def estimate_pose(rgb, box):
    x1, y1, x2, y2 = map(float, box)
    height, width = rgb.shape[:2]
    x1 = float(np.clip(x1, 0, width - 1))
    x2 = float(np.clip(x2, 0, width))
    y1 = float(np.clip(y1, 0, height - 1))
    y2 = float(np.clip(y2, 0, height))
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return None
    center = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    scale = max(bw, bh * IMG_W / IMG_H) * 1.25
    trans = get_affine_transform(center, scale, 0.0, (IMG_W, IMG_H))
    crop = cv2.warpAffine(rgb, trans, (IMG_W, IMG_H)).astype(np.float32) / 255.0
    crop = (crop - IMG_MEAN) / IMG_STD
    tensor = torch.from_numpy(crop.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)
    with torch.inference_mode():
        heatmaps = pose(tensor)[0].float().cpu().numpy()
    keypoints = decode_heatmaps(heatmaps, center, scale).astype(np.float32)
    del tensor, heatmaps, crop
    return keypoints if keypoints.shape == (17, 3) else None


def atomic_npz_save(path, arrays, retries=6):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    last_error = None
    for attempt in range(retries):
        try:
            os.replace(str(tmp), str(path))
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(attempt + 1, 5))
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    raise last_error


def write_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_name(path.stem + f".{os.getpid()}.tmp.json")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


def process_video(rec):
    video_path = Path(str(rec["video_path"]))
    label = str(rec["label"])
    out_path = Path(str(rec["output_path"]))
    result_path = Path(str(rec["result_path"]))
    started = time.time()
    cap = None
    try:
        if not video_path.is_file():
            raise FileNotFoundError(f"Video does not exist: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open source video")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        frame_step = int(cfg["frame_step"])
        max_decoded = int(cfg["max_decoded_frames"])
        max_people = int(cfg["max_people_per_frame"])
        iou_threshold = float(cfg["iou_threshold"])
        max_track_age = int(cfg["max_track_age"])
        frame_index = 0
        reached_eof = False
        active_tracks = {}
        next_track_id = 1
        track_pose_counts = {}
        track_area_sums = {}
        sampled_frame_indices = []
        sampled_observations = []

        def assign_ids(detections, current_frame):
            nonlocal next_track_id
            stale_ids = [tid for tid, state in active_tracks.items()
                         if current_frame - state["last_frame"] > max_track_age]
            for tid in stale_ids:
                active_tracks.pop(tid, None)
            ids = np.full(len(detections), -1, dtype=np.int64)
            candidates = []
            for det_i, det_box in enumerate(detections):
                for track_id, state in active_tracks.items():
                    overlap = box_iou(det_box, state["box"])
                    if overlap >= iou_threshold:
                        candidates.append((overlap, det_i, track_id))
            candidates.sort(key=lambda item: item[0], reverse=True)
            used_detections, used_tracks = set(), set()
            for overlap, det_i, track_id in candidates:
                if det_i in used_detections or track_id in used_tracks:
                    continue
                ids[det_i] = int(track_id)
                used_detections.add(det_i)
                used_tracks.add(track_id)
            for det_i in range(len(detections)):
                if ids[det_i] < 0:
                    ids[det_i] = next_track_id
                    next_track_id += 1
            for det_i, det_box in enumerate(detections):
                track_id = int(ids[det_i])
                previous = active_tracks.get(track_id, {})
                active_tracks[track_id] = {
                    "box": np.asarray(det_box, dtype=np.float32).copy(),
                    "last_frame": int(current_frame),
                    "hits": int(previous.get("hits", 0)) + 1,
                }
            return ids

        print("VIDEO_START", label, video_path.name, flush=True)
        while True:
            if max_decoded > 0 and frame_index >= max_decoded:
                break
            ok, frame = cap.read()
            if not ok:
                reached_eof = True
                break

            # YOLO runs on each decoded frame to maintain temporal association.
            prediction = yolo.predict(
                frame, classes=[0], conf=float(cfg["conf"]), imgsz=int(cfg["imgsz"]),
                device=str(device), max_det=int(cfg["max_det"]), verbose=False
            )[0]
            if prediction.boxes is not None and len(prediction.boxes) > 0:
                boxes = prediction.boxes.xyxy.cpu().numpy().astype(np.float32)
                confidences = prediction.boxes.conf.cpu().numpy().astype(np.float32)
                ids = assign_ids(boxes, frame_index)
            else:
                boxes = np.empty((0, 4), dtype=np.float32)
                confidences = np.empty((0,), dtype=np.float32)
                ids = assign_ids(boxes, frame_index)

            if frame_index % frame_step == 0:
                frame_observations = []
                if len(boxes) > 0:
                    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
                    keep_indices = np.argsort(areas)[::-1][:max_people]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    for det_i in keep_indices:
                        track_id = int(ids[det_i])
                        keypoints = estimate_pose(rgb, boxes[det_i])
                        if keypoints is None:
                            continue
                        finite_joints = np.isfinite(keypoints[:, :2]).all(axis=1)
                        pose_ok = int(finite_joints.sum()) >= 8
                        frame_observations.append({
                            "track_id": track_id,
                            "keypoints": keypoints,
                            "box": boxes[det_i].astype(np.float32),
                            "detection_score": float(confidences[det_i]),
                            "area": float(areas[det_i]),
                            "pose_ok": bool(pose_ok),
                        })
                        if pose_ok:
                            track_pose_counts[track_id] = track_pose_counts.get(track_id, 0) + 1
                            track_area_sums[track_id] = track_area_sums.get(track_id, 0.0) + float(areas[det_i])
                    del rgb
                sampled_frame_indices.append(int(frame_index))
                sampled_observations.append(frame_observations)

            frame_index += 1
            del prediction
            if frame_index % 500 == 0:
                print("FRAME_PROGRESS", label, "decoded", frame_index, "sampled", len(sampled_frame_indices), flush=True)
                gc.collect()

        cap.release()
        cap = None

        # Select the most consistently pose-observed person; retain all-person data too.
        target_track_id = max(
            track_pose_counts,
            key=lambda tid: (track_pose_counts[tid], track_area_sums.get(tid, 0.0))
        ) if track_pose_counts else -1
        n_samples = len(sampled_frame_indices)
        keypoints_selected = np.full((n_samples, 17, 3), np.nan, dtype=np.float32)
        boxes_selected = np.full((n_samples, 4), np.nan, dtype=np.float32)
        has_person = np.zeros(n_samples, dtype=bool)
        selected_track_ids = np.full(n_samples, -1, dtype=np.int64)
        detection_scores = np.zeros(n_samples, dtype=np.float32)
        all_keypoints, all_boxes, all_frame_indices, all_track_ids, all_detection_scores = [], [], [], [], []

        for sample_i, observations in enumerate(sampled_observations):
            for obs in observations:
                all_keypoints.append(obs["keypoints"])
                all_boxes.append(obs["box"])
                all_frame_indices.append(sampled_frame_indices[sample_i])
                all_track_ids.append(obs["track_id"])
                all_detection_scores.append(obs["detection_score"])
            selected_obs = next((obs for obs in observations
                                 if obs["track_id"] == target_track_id and target_track_id >= 0 and obs["pose_ok"]), None)
            if selected_obs is not None:
                keypoints_selected[sample_i] = selected_obs["keypoints"]
                boxes_selected[sample_i] = selected_obs["box"]
                has_person[sample_i] = True
                selected_track_ids[sample_i] = target_track_id
                detection_scores[sample_i] = selected_obs["detection_score"]

        all_keypoints = np.asarray(all_keypoints, dtype=np.float32).reshape(-1, 17, 3)
        all_boxes = np.asarray(all_boxes, dtype=np.float32).reshape(-1, 4)
        all_frame_indices = np.asarray(all_frame_indices, dtype=np.int32)
        all_track_ids = np.asarray(all_track_ids, dtype=np.int64)
        all_detection_scores = np.asarray(all_detection_scores, dtype=np.float32)

        arrays = {
            "keypoints": keypoints_selected,
            "boxes": boxes_selected,
            "frame_indices": np.asarray(sampled_frame_indices, dtype=np.int32),
            "has_person": has_person,
            "track_ids": selected_track_ids,
            "detection_scores": detection_scores,
            "all_keypoints": all_keypoints,
            "all_boxes": all_boxes,
            "all_frame_indices": all_frame_indices,
            "all_track_ids": all_track_ids,
            "all_detection_scores": all_detection_scores,
            "label": np.asarray(label),
            "video_path": np.asarray(str(video_path)),
            "target_track_id": np.asarray(target_track_id, dtype=np.int64),
            "fps": np.asarray(fps, dtype=np.float32),
            "total_frames": np.asarray(total_frames, dtype=np.int32),
            "frame_step": np.asarray(frame_step, dtype=np.int32),
            "person_selection": np.asarray("most_pose_observations_by_greedy_iou_track"),
            "complete_video": np.asarray(reached_eof, dtype=np.bool_),
            "max_decoded_frames": np.asarray(max_decoded, dtype=np.int32),
            "decoded_frames": np.asarray(frame_index, dtype=np.int32),
            "max_people_per_frame": np.asarray(max_people, dtype=np.int32),
            "inference_device": np.asarray(str(device)),
        }
        atomic_npz_save(out_path, arrays)
        result_obj = {
            "status": "success" if reached_eof else "partial",
            "label": label,
            "video_path": str(video_path),
            "output_path": str(out_path),
            "decoded_frames": int(frame_index),
            "sampled_frames": int(n_samples),
            "detections_with_pose": int(has_person.sum()),
            "pose_success_rate": float(has_person.mean()) if n_samples else 0.0,
            "tracked_ids": int(len(track_pose_counts)),
            "target_track_id": int(target_track_id),
            "target_track_pose_observations": int(track_pose_counts.get(target_track_id, 0)),
            "all_person_observations": int(len(all_track_ids)),
            "inference_device": str(device),
            "complete_video": bool(reached_eof),
            "seconds": round(time.time() - started, 2),
        }
        write_json_atomic(result_path, result_obj)
        print("VIDEO_DONE", json.dumps(result_obj, ensure_ascii=False), flush=True)
        return result_obj
    except Exception as exc:
        if cap is not None:
            cap.release()
        error_obj = {
            "status": "video_error",
            "label": label,
            "video_path": str(video_path),
            "output_path": str(out_path),
            "complete_video": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=8),
            "seconds": round(time.time() - started, 2),
        }
        write_json_atomic(result_path, error_obj)
        print("VIDEO_ERROR", json.dumps(error_obj, ensure_ascii=False), flush=True)
        return error_obj
    finally:
        if cap is not None:
            cap.release()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


batch_results = []
for video_record in cfg["videos"]:
    batch_results.append(process_video(video_record))
print("BATCH_DONE", json.dumps({"videos": len(batch_results), "success": sum(r.get("status") == "success" for r in batch_results), "errors": sum(r.get("status") != "success" for r in batch_results)}), flush=True)
