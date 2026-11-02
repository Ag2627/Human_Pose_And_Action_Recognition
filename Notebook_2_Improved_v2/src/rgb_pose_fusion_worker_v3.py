import os, sys, ctypes, json, time, random, gc, traceback
from pathlib import Path

# Load the PyTorch-bundled OpenMP runtime in this child process only.
torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
_dll_handles = []
_openmp_handle = None
if os.name == "nt":
    iomp = torch_lib / "libiomp5md.dll"
    if iomp.is_file():
        try:
            _openmp_handle = ctypes.WinDLL(str(iomp))
            print("OPENMP_PRELOADED_FROM", iomp, flush=True)
        except OSError as exc:
            print("OPENMP_PRELOAD_WARNING", repr(exc), flush=True)
if torch_lib.is_dir():
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            _dll_handles.append(os.add_dll_directory(str(torch_lib)))
        except OSError:
            pass

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)

def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))

def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(obj, tmp)
    os.replace(str(tmp), str(path))

def main(cfg):
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    seq_dir = Path(cfg["sequence_dir"])
    rgb_dir = Path(cfg["rgb_dir"])
    run_dir = Path(cfg["run_dir"])
    index_csv = Path(cfg["feature_index_csv"])
    run_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Anaconda worker. Training was not started.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    print(f"WORKER_PYTHON={sys.executable}", flush=True)
    print(f"DEVICE={device}; GPU={gpu_name}; FREE_GB={free_bytes / (1024**3):.2f}; TOTAL_GB={total_bytes / (1024**3):.2f}", flush=True)
    if free_bytes < int(cfg.get("minimum_free_gpu_gb", 3.0) * (1024**3)):
        raise RuntimeError(
            f"Only {free_bytes / (1024**3):.2f} GB GPU memory is free. "
            "Close other GPU workloads and relaunch; do not lower the safety check blindly."
        )

    metadata = json.loads((seq_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
    if not metadata.get("is_full_dataset", False):
        raise RuntimeError("The sequence dataset is marked as partial.")
    label_map_raw = json.loads((seq_dir / "label_map.json").read_text(encoding="utf-8"))
    label_map = {int(k): str(v) for k, v in label_map_raw.items()}

    arrays = {}
    for split in ("train", "val", "test"):
        x = np.load(seq_dir / f"X_{split}.npy", mmap_mode="r", allow_pickle=False)
        y = np.load(seq_dir / f"y_{split}.npy", mmap_mode="r", allow_pickle=False)
        vids = np.load(seq_dir / f"videos_{split}.npy", mmap_mode="r", allow_pickle=False)
        if x.ndim != 3 or x.shape[1:] != (16, 51):
            raise ValueError(f"{split}: expected X shape [N,16,51], got {x.shape}")
        if len(x) != len(y) or len(y) != len(vids):
            raise ValueError(f"{split}: X/y/videos length mismatch")
        if len(x) == 0:
            raise ValueError(f"{split}: no sequences available")
        arrays[split] = {"X": x, "y": y, "videos": vids}

    # Output classes are restricted to classes with actual training windows.
    train_original_ids = sorted(int(x) for x in np.unique(arrays["train"]["y"]))
    if not train_original_ids:
        raise RuntimeError("No classes have training windows.")
    original_to_model = {orig: i for i, orig in enumerate(train_original_ids)}
    model_to_original = list(train_original_ids)
    n_classes = len(model_to_original)

    val_original_ids = sorted(int(x) for x in np.unique(arrays["val"]["y"]))
    test_original_ids = sorted(int(x) for x in np.unique(arrays["test"]["y"]))
    val_eval_original_ids = sorted(set(train_original_ids).intersection(val_original_ids))
    test_eval_original_ids = sorted(set(train_original_ids).intersection(test_original_ids))
    if not val_eval_original_ids:
        raise RuntimeError("No validation classes overlap with training classes.")
    if not test_eval_original_ids:
        raise RuntimeError("No test classes overlap with training classes.")

    # Load only the validated per-video RGB descriptors required by sequence windows.
    index_df = pd.read_csv(index_csv)
    required_cols = {"video_id", "feature_path", "video_path", "label", "split"}
    if not required_cols.issubset(index_df.columns):
        raise ValueError(f"Feature index is missing columns: {sorted(required_cols - set(index_df.columns))}")
    if index_df["video_id"].duplicated().any():
        raise ValueError("Duplicate video_id entries in validated RGB feature index.")
    rgb_by_video = {}
    for row in index_df.itertuples(index=False):
        with np.load(row.feature_path, allow_pickle=False) as data:
            feature = np.asarray(data["feature"], dtype=np.float32)
            if feature.shape != (512,) or not np.isfinite(feature).all():
                raise ValueError(f"Invalid 512-D RGB feature for video_id={row.video_id}")
            stored_id = int(np.asarray(data["video_id"]).item())
            stored_path = str(np.asarray(data["video_path"]).item())
            stored_label = str(np.asarray(data["label"]).item())
            stored_split = str(np.asarray(data["split"]).item()).lower()
            if stored_id != int(row.video_id):
                raise ValueError(f"RGB video_id mismatch in {row.feature_path}")
            if stored_path != str(row.video_path) or stored_label != str(row.label) or stored_split != str(row.split).lower():
                raise ValueError(f"RGB metadata mismatch for video_id={row.video_id}")
            rgb_by_video[stored_id] = feature.copy()

    for split in ("train", "val", "test"):
        missing = sorted(set(int(v) for v in np.unique(arrays[split]["videos"])) - set(rgb_by_video))
        if missing:
            raise RuntimeError(f"Missing RGB features for {len(missing)} {split} source videos. Example IDs: {missing[:10]}")

    class DatasetRGBPose(Dataset):
        def __init__(self, split):
            self.split = split
            self.x = arrays[split]["X"]
            self.y = arrays[split]["y"]
            self.videos = arrays[split]["videos"]
            # A class with no training windows cannot be learned. Exclude it from
            # model loss/metrics but record its support in the final report.
            self.indices = np.flatnonzero(np.isin(self.y, train_original_ids))
            if len(self.indices) == 0:
                raise ValueError(f"{split}: no windows belong to a class represented in training.")
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, index):
            source_index = int(self.indices[index])
            pose = np.array(self.x[source_index], dtype=np.float32, copy=True)
            original_label = int(self.y[source_index])
            video_id = int(self.videos[source_index])
            rgb = rgb_by_video[video_id]
            return (
                torch.from_numpy(pose),
                torch.from_numpy(rgb.copy()),
                torch.tensor(original_to_model[original_label], dtype=torch.long),
                torch.tensor(video_id, dtype=torch.long),
            )

    batch_size = int(cfg["batch_size"])
    loaders = {
        split: DataLoader(
            DatasetRGBPose(split),
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=True,
            drop_last=False,
        )
        for split in ("train", "val", "test")
    }

    # COCO-17 parent links; one 9-channel feature vector per joint.
    PARENTS = [0, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]

    class FusionModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.pose_norm = nn.LayerNorm(17 * 9)
            self.pose_lstm = nn.LSTM(
                input_size=17 * 9, hidden_size=96, num_layers=1,
                batch_first=True, bidirectional=True
            )
            self.attention = nn.Sequential(nn.Linear(192, 96), nn.Tanh(), nn.Linear(96, 1))
            self.rgb_branch = nn.Sequential(
                nn.LayerNorm(512), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.30)
            )
            self.classifier = nn.Sequential(
                nn.Linear(192 + 256, 256), nn.ReLU(),
                nn.Dropout(0.40), nn.Linear(256, num_classes)
            )

        def enrich_pose(self, pose):
            batch, timesteps, _ = pose.shape
            q = pose.reshape(batch, timesteps, 17, 3)
            xy = torch.nan_to_num(q[..., :2], nan=0.0, posinf=0.0, neginf=0.0)
            confidence = torch.nan_to_num(q[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
            valid = confidence > 0
            velocity = torch.zeros_like(xy)
            acceleration = torch.zeros_like(xy)
            bones = torch.zeros_like(xy)

            if timesteps > 1:
                pair_valid = valid[:, 1:] & valid[:, :-1]
                delta = xy[:, 1:] - xy[:, :-1]
                velocity[:, 1:] = torch.where(pair_valid[..., None], delta, torch.zeros_like(delta))
            if timesteps > 2:
                triple_valid = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
                delta_velocity = velocity[:, 2:] - velocity[:, 1:-1]
                acceleration[:, 2:] = torch.where(
                    triple_valid[..., None], delta_velocity, torch.zeros_like(delta_velocity)
                )
            for joint, parent in enumerate(PARENTS):
                if joint == parent:
                    continue
                bone_valid = valid[:, :, joint] & valid[:, :, parent]
                vector = xy[:, :, joint] - xy[:, :, parent]
                bones[:, :, joint] = torch.where(
                    bone_valid[..., None], vector, torch.zeros_like(vector)
                )

            enriched = torch.cat([q, velocity, acceleration, bones], dim=-1)
            return torch.nan_to_num(enriched, nan=0.0, posinf=0.0, neginf=0.0).reshape(batch, timesteps, 17 * 9)

        def forward(self, pose, rgb):
            pose_features = self.enrich_pose(pose)
            temporal, _ = self.pose_lstm(self.pose_norm(pose_features))
            weights = torch.softmax(self.attention(temporal).squeeze(-1), dim=1)
            pose_embedding = (temporal * weights.unsqueeze(-1)).sum(dim=1)
            rgb_embedding = self.rgb_branch(rgb)
            return self.classifier(torch.cat([pose_embedding, rgb_embedding], dim=1))

    # Calculate mild inverse-frequency weights to reduce dominance by frequent classes.
    train_y_orig = np.asarray(arrays["train"]["y"], dtype=np.int64)
    train_y_model = np.asarray([original_to_model[int(v)] for v in train_y_orig], dtype=np.int64)
    counts = np.bincount(train_y_model, minlength=n_classes).astype(np.float64)
    if np.any(counts == 0):
        raise RuntimeError("Internal class mapping error: a train class has zero samples.")
    class_weights = np.sqrt(counts.mean() / counts)
    class_weights = np.clip(class_weights, 0.5, 3.0)
    class_weights = class_weights / class_weights.mean()
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    model = FusionModel(n_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=float(cfg["label_smoothing"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    run_dir = Path(cfg["run_dir"])
    best_path = run_dir / "rgb_pose_fusion_best.pt"
    latest_path = run_dir / "latest_checkpoint.pt"
    history_path = run_dir / "training_history.csv"
    state_path = run_dir / "progress_state.json"
    max_epochs = int(cfg["epochs"])
    patience = int(cfg["patience"])
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0
    start_epoch = 1
    history_rows = []

    if history_path.is_file():
        try:
            history_rows = pd.read_csv(history_path).to_dict("records")
        except Exception:
            history_rows = []

    if latest_path.is_file():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_score = float(ckpt.get("best_validation_video_macro_f1", -1.0))
        best_epoch = int(ckpt.get("best_epoch", 0))
        stale_epochs = int(ckpt.get("stale_epochs", 0))
        print(f"RESUMING_FROM_EPOCH={start_epoch}; BEST_EPOCH={best_epoch}; BEST_VAL_VIDEO_MACRO_F1={best_score:.4f}", flush=True)

    if (run_dir / "test_metrics.json").is_file() and not cfg.get("inference_only", False):
        print("FUSION_ALREADY_COMPLETE; existing test_metrics.json found. No retraining performed.", flush=True)
        return

    if start_epoch > max_epochs:
        if not best_path.is_file():
            raise RuntimeError("Maximum epochs reached but best checkpoint is missing.")
        print("Training checkpoints reached maximum epochs; running final evaluation.", flush=True)

    def mapped_metrics(y_true, y_pred, class_indices):
        labels = list(class_indices)
        if not labels:
            raise RuntimeError("No supported classes available for metric calculation.")
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1_supported_classes": float(
                f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
            ),
            "classes_in_macro_f1": int(len(labels)),
        }

    def aggregate_by_video(probabilities, y_true, video_ids):
        video_ids = np.asarray(video_ids, dtype=np.int64)
        y_true = np.asarray(y_true, dtype=np.int64)
        unique_videos, inverse = np.unique(video_ids, return_inverse=True)
        sums = np.zeros((len(unique_videos), probabilities.shape[1]), dtype=np.float64)
        np.add.at(sums, inverse, probabilities)
        counts_per_video = np.bincount(inverse, minlength=len(unique_videos))
        mean_probabilities = sums / counts_per_video[:, None]
        video_labels = np.empty(len(unique_videos), dtype=np.int64)
        for i in range(len(unique_videos)):
            labels_here = np.unique(y_true[inverse == i])
            if len(labels_here) != 1:
                raise ValueError(f"Source video {unique_videos[i]} has multiple labels within one split.")
            video_labels[i] = labels_here[0]
        return unique_videos, video_labels, mean_probabilities.argmax(axis=1), mean_probabilities

    def evaluate(loader, split_name):
        model.eval()
        y_true_parts, probability_parts, video_parts = [], [], []
        loss_total = 0.0
        n_total = 0
        with torch.inference_mode():
            for pose, rgb, target, video_id in loader:
                pose = pose.to(device, non_blocking=True)
                rgb = rgb.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                logits = model(pose, rgb)
                loss = criterion(logits, target)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                y_true_parts.append(target.cpu().numpy())
                probability_parts.append(probs)
                video_parts.append(video_id.numpy())
                loss_total += float(loss.item()) * len(target)
                n_total += len(target)
        y_true = np.concatenate(y_true_parts)
        probabilities = np.concatenate(probability_parts)
        video_ids = np.concatenate(video_parts)
        y_pred = probabilities.argmax(axis=1)
        supported_model_ids = sorted(set(int(v) for v in np.unique(y_true)))
        window_metrics = mapped_metrics(y_true, y_pred, supported_model_ids)
        unique_videos, video_y, video_pred, video_probs = aggregate_by_video(probabilities, y_true, video_ids)
        video_metrics = mapped_metrics(video_y, video_pred, supported_model_ids)
        return {
            "loss": loss_total / max(n_total, 1),
            "window_metrics": window_metrics,
            "video_metrics": video_metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            "probabilities": probabilities,
            "video_ids": video_ids,
            "unique_videos": unique_videos,
            "video_y": video_y,
            "video_pred": video_pred,
            "video_probs": video_probs,
        }

    def train_one_epoch(loader):
        model.train()
        y_true_parts, y_pred_parts = [], []
        loss_total = 0.0
        n_total = 0
        for pose, rgb, target, _video_id in loader:
            pose = pose.to(device, non_blocking=True)
            rgb = rgb.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(pose, rgb)
            loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += float(loss.item()) * len(target)
            n_total += len(target)
            y_true_parts.append(target.detach().cpu().numpy())
            y_pred_parts.append(logits.detach().argmax(dim=1).cpu().numpy())
        y_true = np.concatenate(y_true_parts)
        y_pred = np.concatenate(y_pred_parts)
        train_classes = sorted(set(int(v) for v in np.unique(y_true)))
        m = mapped_metrics(y_true, y_pred, train_classes)
        m["loss"] = loss_total / max(n_total, 1)
        return m

    val_model_class_ids = sorted(original_to_model[v] for v in val_eval_original_ids)

    if start_epoch <= max_epochs:
        for epoch in range(start_epoch, max_epochs + 1):
            t0 = time.time()
            train_metrics = train_one_epoch(loaders["train"])
            val_metrics = evaluate(loaders["val"], "val")
            val_score = float(val_metrics["video_metrics"]["macro_f1_supported_classes"])
            scheduler.step(val_score)

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                "train_macro_f1_supported_classes": train_metrics["macro_f1_supported_classes"],
                "val_loss": val_metrics["loss"],
                "val_window_accuracy": val_metrics["window_metrics"]["accuracy"],
                "val_window_macro_f1": val_metrics["window_metrics"]["macro_f1_supported_classes"],
                "val_video_accuracy": val_metrics["video_metrics"]["accuracy"],
                "val_video_macro_f1": val_score,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "seconds": round(time.time() - t0, 2),
            }
            history_rows.append(row)
            pd.DataFrame(history_rows).to_csv(history_path, index=False)

            improved = val_score > best_score + 1e-6
            if improved:
                best_score = val_score
                best_epoch = epoch
                stale_epochs = 0
                atomic_torch_save({
                    "model_state": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_video_macro_f1": best_score,
                    "model_to_original": model_to_original,
                    "label_map": label_map,
                    "architecture": "RGB R3D-18 clip descriptor + enriched pose BiLSTM attention late fusion",
                }, best_path)
            else:
                stale_epochs += 1

            checkpoint_obj = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_validation_video_macro_f1": best_score,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "model_to_original": model_to_original,
                "label_map": label_map,
            }
            atomic_torch_save(checkpoint_obj, latest_path)
            if epoch % 5 == 0:
                atomic_torch_save(checkpoint_obj, run_dir / f"recovery_epoch_{epoch:03d}.pt")

            state = {
                "status": "training",
                "epoch": epoch,
                "max_epochs": max_epochs,
                "best_epoch": best_epoch,
                "best_validation_video_macro_f1": best_score,
                "stale_epochs": stale_epochs,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1_supported_classes"],
                "val_window_accuracy": val_metrics["window_metrics"]["accuracy"],
                "val_video_accuracy": val_metrics["video_metrics"]["accuracy"],
                "val_video_macro_f1": val_score,
                "gpu_name": gpu_name,
                "device": str(device),
                "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            atomic_json(state_path, state)

            print(
                f"Epoch {epoch:02d}/{max_epochs} | "
                f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
                f"train_F1={train_metrics['macro_f1_supported_classes']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} val_window_acc={val_metrics['window_metrics']['accuracy']:.4f} "
                f"val_video_acc={val_metrics['video_metrics']['accuracy']:.4f} "
                f"val_video_macroF1={val_score:.4f} | best={best_score:.4f}@{best_epoch} "
                f"| seconds={row['seconds']}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            if stale_epochs >= patience:
                print(f"EARLY_STOPPING at epoch {epoch}; patience={patience}", flush=True)
                break

    if not best_path.is_file():
        raise RuntimeError("No best validation checkpoint was saved.")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state"])
    test_result = evaluate(loaders["test"], "test")
        # Save video-level probabilities for ROC and Precision-Recall analysis
    np.save(run_dir / "video_probabilities.npy", test_result["video_probs"])
    np.save(run_dir / "video_true_labels.npy", test_result["video_y"])
    np.save(run_dir / "video_class_ids.npy",np.asarray(model_to_original, dtype=np.int64))

    print("VIDEO PROBABILITIES SAVED")
    print("Probability matrix:", test_result["video_probs"].shape)
    print("True labels:", test_result["video_y"].shape)

    # Convert model-index predictions back to original class IDs for interpretable reports.
    model_to_original_array = np.asarray(model_to_original, dtype=np.int64)
    test_true_orig = model_to_original_array[test_result["y_true"]]
    test_pred_orig = model_to_original_array[test_result["y_pred"]]
    video_true_orig = model_to_original_array[test_result["video_y"]]
    video_pred_orig = model_to_original_array[test_result["video_pred"]]

    test_model_classes = sorted(set(int(v) for v in np.unique(test_result["y_true"])))
    test_original_classes = [model_to_original[i] for i in test_model_classes]
    report_names = [label_map[i] for i in test_original_classes]
    class_report = classification_report(
        video_true_orig, video_pred_orig,
        labels=test_original_classes, target_names=report_names,
        output_dict=True, zero_division=0
    )
    pd.DataFrame(class_report).T.to_csv(run_dir / "video_classification_report.csv")

    pd.DataFrame(
        confusion_matrix(video_true_orig, video_pred_orig, labels=test_original_classes),
        index=report_names, columns=report_names
    ).to_csv(run_dir / "video_confusion_matrix.csv")

    window_report = classification_report(
        test_true_orig, test_pred_orig,
        labels=test_original_classes, target_names=report_names,
        output_dict=True, zero_division=0
    )
    pd.DataFrame(window_report).T.to_csv(run_dir / "window_classification_report.csv")

    video_predictions = pd.DataFrame({
        "video_id": test_result["unique_videos"],
        "true_label_id": video_true_orig,
        "true_label": [label_map[int(x)] for x in video_true_orig],
        "predicted_label_id": video_pred_orig,
        "predicted_label": [label_map[int(x)] for x in video_pred_orig],
        "num_windows": [
            int(np.sum(test_result["video_ids"] == vid)) for vid in test_result["unique_videos"]
        ],
    })
    video_predictions.to_csv(run_dir / "video_predictions.csv", index=False)

    window_predictions = pd.DataFrame({
        "video_id": test_result["video_ids"],
        "true_label_id": test_true_orig,
        "true_label": [label_map[int(x)] for x in test_true_orig],
        "predicted_label_id": test_pred_orig,
        "predicted_label": [label_map[int(x)] for x in test_pred_orig],
    })
    window_predictions.to_csv(run_dir / "window_predictions.csv", index=False)

    all_test_original_ids = sorted(int(x) for x in np.unique(arrays["test"]["y"]))
    test_without_train = sorted(set(all_test_original_ids) - set(train_original_ids))
    val_without_train = sorted(set(val_original_ids) - set(train_original_ids))

    final_metrics = {
        "experiment": "late RGB + enriched tracked-pose fusion",
        "description": "512-D pretrained R3D-18 video descriptor plus pose BiLSTM attention with confidence-masked velocity, acceleration and COCO-17 bone vectors; video-level mean-probability aggregation.",
        "best_epoch": int(best_checkpoint["best_epoch"]),
        "best_validation_video_macro_f1_supported_classes": float(best_checkpoint["best_validation_video_macro_f1"]),
        "train_windows": int(len(arrays["train"]["y"])),
        "validation_windows": int(len(arrays["val"]["y"])),
        "test_windows": int(len(arrays["test"]["y"])),
        "test_videos": int(len(test_result["unique_videos"])),
        "classes_in_model": int(n_classes),
        "test_classes_evaluated": int(len(test_original_classes)),
        "test_classes_without_training_support": [label_map[x] for x in test_without_train],
        "validation_classes_without_training_support": [label_map[x] for x in val_without_train],
        "window_metrics": mapped_metrics(
            test_result["y_true"], test_result["y_pred"], test_model_classes
        ),
        "video_metrics": mapped_metrics(
            test_result["video_y"], test_result["video_pred"], test_model_classes
        ),
        "checkpoint": str(best_path),
        "device": str(device),
        "gpu_name": gpu_name,
        "rgb_feature_dimension": 512,
        "pose_sequence_shape": [16, 51],
        "model_class_ids_original": model_to_original,
    }
    atomic_json(run_dir / "test_metrics.json", final_metrics)
    atomic_json(state_path, {
        "status": "completed",
        "epoch": int(best_checkpoint["best_epoch"]),
        "max_epochs": max_epochs,
        "best_epoch": int(best_checkpoint["best_epoch"]),
        "best_validation_video_macro_f1": best_score,
        "test_video_accuracy": final_metrics["video_metrics"]["accuracy"],
        "test_video_macro_f1": final_metrics["video_metrics"]["macro_f1_supported_classes"],
        "gpu_name": gpu_name,
        "device": str(device),
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print("RGB_POSE_FUSION_COMPLETE", json.dumps(final_metrics, indent=2), flush=True)

if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise RuntimeError("Usage: rgb_pose_fusion_worker_v3.py <config.json>")
        cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        main(cfg)
    except BaseException as exc:
        print("FUSION_WORKER_EXCEPTION", repr(exc), flush=True)
        traceback.print_exc()
        sys.exit(2)
