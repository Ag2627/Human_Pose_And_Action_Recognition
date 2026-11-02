from pathlib import Path
import json, os
import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights
from ultralytics import YOLO

# These are the exact parent links and model dimensions used in Notebook 07.
PARENTS = [0, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14]

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.pose_norm = nn.LayerNorm(17 * 9)
        self.pose_lstm = nn.LSTM(input_size=17 * 9, hidden_size=96, num_layers=1,
                                 batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(192, 96), nn.Tanh(), nn.Linear(96, 1))
        self.rgb_branch = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 256),
                                        nn.ReLU(), nn.Dropout(0.30))
        self.classifier = nn.Sequential(nn.Linear(192 + 256, 256), nn.ReLU(),
                                        nn.Dropout(0.40), nn.Linear(256, num_classes))

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
            acceleration[:, 2:] = torch.where(triple_valid[..., None], delta_velocity, torch.zeros_like(delta_velocity))
        for joint, parent in enumerate(PARENTS):
            if joint == parent: continue
            bone_valid = valid[:, :, joint] & valid[:, :, parent]
            vector = xy[:, :, joint] - xy[:, :, parent]
            bones[:, :, joint] = torch.where(bone_valid[..., None], vector, torch.zeros_like(vector))
        enriched = torch.cat([q, velocity, acceleration, bones], dim=-1)
        return torch.nan_to_num(enriched, nan=0.0, posinf=0.0, neginf=0.0).reshape(batch, timesteps, 17 * 9)

    def forward(self, pose, rgb):
        pose_features = self.enrich_pose(pose)
        temporal, _ = self.pose_lstm(self.pose_norm(pose_features))
        weights = torch.softmax(self.attention(temporal).squeeze(-1), dim=1)
        pose_embedding = (temporal * weights.unsqueeze(-1)).sum(dim=1)
        rgb_embedding = self.rgb_branch(rgb)
        return self.classifier(torch.cat([pose_embedding, rgb_embedding], dim=1))

class FusionVideoPredictor:
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.n2 = self.project_dir / 'Notebook_2_Improved_v2'
        self.checkpoint_path = Path(os.environ.get(
            'UCF_FUSION_CHECKPOINT', str(self.n2 / 'outputs' / 'models' / 'rgb_pose_fusion_v3_20261031_035536' / 'rgb_pose_fusion_best.pt')))
        self.pose_checkpoint_path = Path(os.environ.get('UCF_POSE_CHECKPOINT', str(self.project_dir / 'checkpoints' / 'epoch_209.pth')))
        self.yolo_path = Path(os.environ.get('UCF_YOLO_WEIGHTS', str(self.project_dir / 'yolo11n.pt')))
        if not self.yolo_path.is_file():
            candidate = self.project_dir / 'notebooks' / 'yolo11n.pt'
            if candidate.is_file(): self.yolo_path = candidate
        pose_src = self.project_dir / 'src'
        if not (pose_src / 'vitpose_model.py').is_file():
            raise FileNotFoundError(f"Missing {pose_src / 'vitpose_model.py'}. Keep your existing project src folder in place.")
        if not self.pose_checkpoint_path.is_file(): raise FileNotFoundError(f"ViTPose checkpoint not found: {self.pose_checkpoint_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Fusion checkpoint not found: {self.checkpoint_path}. Set UCF_FUSION_CHECKPOINT to its actual full path.")
        if not self.yolo_path.is_file(): raise FileNotFoundError(f"YOLO weights not found: {self.yolo_path}")

        import sys
        sys.path.insert(0, str(self.project_dir))
        from src.vitpose_model import ViTPose, get_affine_transform, decode_heatmaps, IMG_H, IMG_W, IMG_MEAN, IMG_STD
        self.get_affine_transform = get_affine_transform
        self.decode_heatmaps = decode_heatmaps
        self.IMG_H, self.IMG_W = IMG_H, IMG_W
        self.IMG_MEAN, self.IMG_STD = IMG_MEAN, IMG_STD
        pose_ckpt = torch.load(self.pose_checkpoint_path, map_location=self.device, weights_only=True)
        self.pose_model = ViTPose().to(self.device)
        self.pose_model.load_state_dict(pose_ckpt['model_state'], strict=True)
        self.pose_model.eval()
        self.detector = YOLO(str(self.yolo_path))

        weights = R3D_18_Weights.DEFAULT
        self.rgb_model = r3d_18(weights=weights)
        self.rgb_model.fc = nn.Identity()
        self.rgb_model.to(self.device).eval()

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        if 'model_state' not in ckpt or 'model_to_original' not in ckpt or 'label_map' not in ckpt:
            raise ValueError('Fusion checkpoint does not contain model_state, model_to_original and label_map as saved by Notebook 07.')
        self.model_to_original = [int(x) for x in ckpt['model_to_original']]
        raw_map = ckpt['label_map']
        self.label_map = {int(k): str(v) for k, v in raw_map.items()}
        self.fusion_model = FusionModel(len(self.model_to_original)).to(self.device)
        self.fusion_model.load_state_dict(ckpt['model_state'], strict=True)
        self.fusion_model.eval()

    def _pose_for_box(self, frame_rgb, bbox):
        x1,y1,x2,y2 = map(float,bbox)
        bw,bh=max(x2-x1,1.0),max(y2-y1,1.0)
        center=np.array([(x1+x2)/2,(y1+y2)/2],dtype=np.float32)
        scale=max(bw,bh*self.IMG_W/self.IMG_H)*1.25
        trans=self.get_affine_transform(center,scale,0.0,(self.IMG_W,self.IMG_H))
        crop=cv2.warpAffine(frame_rgb,trans,(self.IMG_W,self.IMG_H))
        crop=crop.astype(np.float32)/255.0
        crop=(crop-self.IMG_MEAN)/self.IMG_STD
        tensor=torch.from_numpy(crop.transpose(2,0,1).copy()).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            heatmaps=self.pose_model(tensor)[0].float().cpu().numpy()
        return self.decode_heatmaps(heatmaps,center,scale).astype(np.float32)

    @staticmethod
    def _normalize_pose(keypoints, boxes, valid_flags):
        n=len(keypoints); out=np.zeros((n,17,3),dtype=np.float32); frame_valid=np.zeros(n,dtype=bool)
        for t in range(n):
            if not valid_flags[t]: continue
            kp=np.asarray(keypoints[t],dtype=np.float32); box=np.asarray(boxes[t],dtype=np.float32)
            if not np.isfinite(box).all(): continue
            x1,y1,x2,y2=map(float,box); bw,bh=x2-x1,y2-y1
            if bw<2 or bh<2: continue
            finite_xy=np.isfinite(kp[:,:2]).all(axis=1)
            if int(finite_xy.sum())<8: continue
            out[t,finite_xy,0]=(kp[finite_xy,0]-x1)/bw
            out[t,finite_xy,1]=(kp[finite_xy,1]-y1)/bh
            scores=np.nan_to_num(kp[:,2],nan=0.0,posinf=0.0,neginf=0.0)
            out[t,:,2]=np.clip(scores,0.0,1.0)
            out[t,~finite_xy,:2]=0.0; out[t,~finite_xy,2]=0.0
            frame_valid[t]=True
        return out,frame_valid

    @staticmethod
    def _resize_center_crop(rgb, short_side=128, crop=112):
        h,w=rgb.shape[:2]; scale=float(short_side)/float(min(h,w))
        nw=max(crop,int(round(w*scale))); nh=max(crop,int(round(h*scale)))
        resized=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_LINEAR)
        x0=max(0,(nw-crop)//2); y0=max(0,(nh-crop)//2)
        return resized[y0:y0+crop,x0:x0+crop]

    def _rgb_feature(self, video_path, num_frames=16):
        """Decode sequentially, then uniformly sample frames.

        Some Windows/OpenCV video backends fail repeated random seeks on uploaded
        MP4 files (e.g. only 1-2 successful reads). Sequential decoding avoids
        that issue while preserving the training resize/crop and normalization.
        """
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open video: {video_path}')
        try:
            reported_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            decoded_frames = []
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if bgr is None or bgr.size == 0:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                decoded_frames.append(rgb)
            if not decoded_frames:
                raise RuntimeError(
                    'OpenCV opened the file but decoded zero frames. '
                    'Try converting the video to MP4 (H.264) and upload it again.'
                )
            # Use actual decoded frame count, not potentially inaccurate metadata.
            actual_count = len(decoded_frames)
            indices = np.linspace(0, actual_count - 1, int(num_frames)).round().astype(np.int64)
            frames = [self._resize_center_crop(decoded_frames[int(i)]) for i in indices]
            if len(frames) != int(num_frames):
                raise RuntimeError(f'Expected {num_frames} sampled RGB frames, got {len(frames)}')
        finally:
            cap.release()

        x = torch.from_numpy(np.asarray(frames, dtype=np.uint8)).to(self.device, dtype=torch.float32)
        x = x.permute(3, 0, 1, 2).unsqueeze(0).div_(255.0)
        mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(1, 3, 1, 1, 1)
        with torch.inference_mode():
            feat = self.rgb_model((x - mean) / std).float().reshape(1, 512)
        return feat, actual_count

    def _extract_pose_windows(self, video_path):
        cap=cv2.VideoCapture(str(video_path))
        if not cap.isOpened(): raise RuntimeError(f'Cannot open video for pose: {video_path}')
        sample_frames=[]; observations=[]; track_counts={}; track_areas={}; frame_i=0
        # Reset tracker between uploaded videos, where supported by Ultralytics.
        pred=getattr(self.detector,'predictor',None)
        if pred is not None and hasattr(pred,'trackers'):
            for tracker in pred.trackers:
                if hasattr(tracker,'reset'): tracker.reset()
        try:
            while True:
                ok,frame_bgr=cap.read()
                if not ok: break
                sampled=(frame_i % 4 == 0)
                rows=[]
                result=self.detector.track(frame_bgr,persist=True,tracker='bytetrack.yaml',classes=[0],
                                           conf=0.20,imgsz=640,device=0 if self.device.type=='cuda' else 'cpu',verbose=False)[0]
                if result.boxes is not None and len(result.boxes):
                    boxes=result.boxes.xyxy.cpu().numpy(); confs=result.boxes.conf.cpu().numpy()
                    ids=(result.boxes.id.cpu().numpy().astype(np.int64) if result.boxes.id is not None else np.full(len(boxes),-1,dtype=np.int64))
                    areas=np.maximum(0,boxes[:,2]-boxes[:,0])*np.maximum(0,boxes[:,3]-boxes[:,1])
                    for tid,area in zip(ids,areas):
                        if tid>=0:
                            track_counts[int(tid)]=track_counts.get(int(tid),0)+1
                            track_areas[int(tid)]=track_areas.get(int(tid),0.0)+float(area)
                    if sampled:
                        keep=np.argsort(areas)[::-1][:3]
                        rgb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
                        for idx in keep:
                            try:
                                kp=self._pose_for_box(rgb,boxes[idx])
                                rows.append({'track_id':int(ids[idx]),'keypoints':kp,'box':boxes[idx].astype(np.float32),'area':float(areas[idx])})
                            except Exception:
                                continue
                if sampled:
                    sample_frames.append(frame_i); observations.append(rows)
                frame_i+=1
        finally: cap.release()
        assigned=[o for group in observations for o in group if o['track_id']>=0]
        if not assigned: raise RuntimeError('No tracked person with a valid ByteTrack ID and pose was found.')
        pose_counts={}; pose_areas={}
        for o in assigned:
            tid=o['track_id']; pose_counts[tid]=pose_counts.get(tid,0)+1
            pose_areas[tid]=pose_areas.get(tid,0.0)+o['area']
        target_id=max(pose_counts,key=lambda tid:(pose_counts[tid],track_counts.get(tid,0),pose_areas[tid]))
        n=len(sample_frames); kps=np.full((n,17,3),np.nan,np.float32); boxes=np.full((n,4),np.nan,np.float32)
        has=np.zeros(n,bool); tids=np.full(n,-1,np.int64)
        for j,group in enumerate(observations):
            candidates=[o for o in group if o['track_id']==target_id]
            if not candidates: continue
            chosen=max(candidates,key=lambda o:o['area'])
            kps[j]=chosen['keypoints']; boxes[j]=chosen['box']; has[j]=True; tids[j]=target_id
        pose,valid=self._normalize_pose(kps,boxes,has & (tids==target_id))
        windows=[]; length=16; stride=4
        if len(pose)>=length:
            starts=list(range(0,len(pose)-length+1,stride)); final_start=len(pose)-length
            if starts and starts[-1]!=final_start: starts.append(final_start)
            for start in starts:
                end=start+length
                if float(valid[start:end].mean())>=0.80:
                    windows.append(pose[start:end].reshape(length,51))
        if not windows: raise RuntimeError('No valid 16-frame pose windows (requires at least 80% valid frames per window).')
        return np.stack(windows).astype(np.float32), int(target_id), int(frame_i), int(valid.sum())

    def predict(self, video_path):
        rgb_feature, decoded=self._rgb_feature(video_path)
        pose_windows,track_id,frames,valid_frames=self._extract_pose_windows(video_path)
        rgb_batch=rgb_feature.expand(len(pose_windows),-1)
        pose_tensor=torch.from_numpy(pose_windows).to(self.device)
        with torch.inference_mode():
            logits=self.fusion_model(pose_tensor,rgb_batch)
            probs=torch.softmax(logits,dim=1).cpu().numpy()
        # Notebook 07 selects video predictions by mean window probabilities.
        video_probs=probs.mean(axis=0)
        model_idx=int(video_probs.argmax())
        original_id=self.model_to_original[model_idx]
        label=self.label_map[original_id]
        order=np.argsort(video_probs)[::-1][:5]
        import pandas as pd
        top5=pd.DataFrame({'Action':[self.label_map[self.model_to_original[int(i)]] for i in order],
                           'Probability':[float(video_probs[i]) for i in order]})
        return {'label':label,'confidence':float(video_probs[model_idx]),'top5':top5,
                'track_id':track_id,'num_windows':len(pose_windows),'decoded_frames':decoded,
                'valid_pose_frames':valid_frames,'device':str(self.device)}
