# UCF101 RGB + Pose Fusion Frontend (matched to Notebook 07)

## Placement
Extract this folder as `C:\Users\admin\ViTPose_Action_Recognition\frontend`.
The existing project root must retain its `src\vitpose_model.py`, `checkpoints\epoch_209.pth`, and YOLO weights.

## Fusion checkpoint
The screenshot shows the best checkpoint in:
`Notebook_2_Improved_v2\outputs\models\rgb_pose_fusion_v3_20261031_035536\rgb_pose_fusion_best.pt`

If the timestamp folder differs, set the environment variable `UCF_FUSION_CHECKPOINT` to the exact full path before starting Streamlit.

## Run from Jupyter
```python
%pip install -r r"C:\Users\admin\ViTPose_Action_Recognition\frontend\requirements.txt"
```
Then:
```python
import os
os.chdir(r"C:\Users\admin\ViTPose_Action_Recognition\frontend")
!streamlit run app.py
```
Open `http://localhost:8501`.

## Important model scope
This matches the Notebook 07 architecture and inference preprocessing: R3D-18 Kinetics-400 512-D descriptor, COCO-17 pose sequence normalized to person box, enriched pose (velocity, acceleration, bone vectors), BiLSTM with attention, late fusion, and mean-window-probability video aggregation. Notebook 04 trains one dominant tracked person per video. The frontend therefore reports the dominant-person/video prediction; it does not claim independently trained per-person predictions for all people in the scene.

First run may download torchvision R3D-18 pretrained weights. The frontend is provided as integration code and still needs a local smoke test with your exact environment and checkpoints.
