# Notebook 2 Improved v2 — isolated, non-destructive improvement workspace

## Important: Notebook 1 and Anaconda are left alone

This package does **not** install, uninstall, upgrade or downgrade packages. It does not change Anaconda, the system PATH, global environment variables, DLL files, Notebook 1 code, keypoints, sequences, or model checkpoints. It uses the existing Python 3 (ipykernel) already selected in Jupyter. All new outputs are written under `Notebook_2_Improved_v2`.

Heavy YOLO/ViTPose inference, ST-GCN training, R3D feature extraction, and RGB-pose fusion execute in separate child processes. The notebook kernel launches the worker and reads its log files. If a worker crashes or times out, the child process can fail without directly terminating Jupyter. This is a crash-containment measure—not an absolute guarantee against Windows, driver, or native-library failures.

In the child process only, the scripts set thread counts to 1, prioritize PyTorch's bundled DLL folder, and attempt to preload PyTorch's bundled `libiomp5md.dll`. They do **not** set the unsafe `KMP_DUPLICATE_LIB_OK` workaround. If the OpenMP error persists, the worker will return a failure/log for diagnosis; no package changes are required by these notebooks.

## Install

Extract the folder `Notebook_2_Improved_v2` directly inside:

`C:\Users\admin\ViTPose_Action_Recognition`

Open notebooks from `Notebook_2_Improved_v2\notebooks` in your existing Jupyter Python 3 kernel. Do not switch environments. Do not delete or overwrite your previous `Notebook_2_Improved` folder.

## The nine notebooks

1. `00_setup_and_preflight.ipynb` — verifies the Notebook 1 source assets and recording-group split; creates only v2 folders.
2. `01_pose_quality_audit.ipynb` — read-only audit of original keypoint NPZs; saves per-video/per-class CSVs and skeleton previews.
3. `02_tracking_pilot.ipynb` — isolated one-video YOLO + ByteTrack diagnostic (CPU, 80 decoded frames).
4. `03_tracked_pose_extraction.ipynb` — resumable per-video YOLO + ByteTrack + ViTPose extraction; five-video pilot by default. Saves persistent track IDs and all retained person observations, plus the longest-observed target track.
5. `04_build_tracked_sequences_group_split.ipynb` — rebuilds new windows from tracked NPZs using the exact existing Notebook 1 recording-group assignment; never reshuffles videos. It only marks the dataset full when every expected source video has a valid NPZ processed to EOF.
6. `05_train_stgcn.ipynb` — compact ST-GCN-inspired graph model. Its 9 channels per joint are x/y/confidence, confidence-masked velocity, acceleration, and bone vectors. Batch size 8; child process; per-epoch history and recovery checkpoints every 5 epochs.
7. `06_extract_rgb_features.ipynb` — optional R3D-18 pretrained RGB clip descriptors; five-video pilot by default. The first run may need internet to download torchvision's pretrained weights if not already cached.
8. `07_train_rgb_pose_fusion.ipynb` — optional late fusion of the R3D-18 descriptor and the enriched (9-channel-per-joint) BiLSTM pose branch.
9. `08_compare_experiments.ipynb` — compares metrics reports from Notebook 1 and Notebook 2 v2.

## Required Notebook 1 inputs

- `data\processed\manifest.csv`
- `data\processed\sequences_group_split_v1\video_split_manifest.csv`
- `data\processed\sequences_group_split_v1\label_map.json`
- `data\processed\keypoints_batch_all\*.npz`
- `src\vitpose_model.py`
- `checkpoints\epoch_209.pth`
- `notebooks\yolo11n.pt` or project-root `yolo11n.pt`

## Run order and safe defaults

1. Run Notebook 00, then 01, then 02. Notebook 02 must return code 0 and its previews should show detections with reasonably stable IDs before continuing.
2. Run Notebook 03 with the defaults first: five videos, max 120 decoded frames, CPU. Inspect the resulting NPZs and preview/report files.
3. Only after accepting the pilot, set `RUN_FULL_MANIFEST=True` **and** `MAX_DECODED_FRAMES=0` in Notebook 03 to process full videos. Its report is checkpointed after each video and valid completed NPZs are skipped on rerun.
4. Run Notebook 04 after enough tracked files exist. The resulting metadata distinguishes complete full-video coverage from a pilot or capped run.
5. Notebook 05 refuses final training while the sequence metadata says the tracked dataset is partial.
6. Notebook 06 starts with a five-video RGB pilot. Full extraction must be explicitly enabled only after the pilot works. Notebook 07 requires complete RGB features for the same split's videos.

## Improvements implemented

- **A — tracking and pose audit:** persistent person tracking IDs, per-person pose observations, dominant-track sequence, and quality reports. Numeric IDs are association metadata; they are not fed as arbitrary scalar inputs.
- **B — pose representation:** body-box-relative x/y and confidence are built in Notebook 04; Notebook 05 derives confidence-masked velocity, acceleration, and bone vectors and learns over the COCO joint graph.
- **C — RGB + pose:** Notebook 06 extracts an RGB clip descriptor; Notebook 07 fuses that descriptor with an enriched temporal pose branch.

## Limitations that must be understood

- No software package changes are performed. The notebooks cannot promise that a worker, Windows native library, hardware, or GPU driver will never fail. They isolate heavy processes so a child crash should not directly kill the Jupyter kernel and save logs for diagnosis.
- Tracking selects the longest-observed persistent track as the primary actor. Other retained person observations are saved, but a complete multi-person interaction classifier is not yet implemented.
- Notebook 05 is a compact ST-GCN-inspired implementation, not an exact reproduction of the published ST-GCN, CTR-GCN, or PoseConv3D architecture.
- Notebook 07 is late fusion with one clip-level RGB descriptor per source video, repeated across its pose windows; it is not a fully synchronized two-stream video model.
- The source split is the existing randomized recording-group split, not the official UCF101 split-list benchmark. Report this accurately; use official UCF101 splits for a final published benchmark.
