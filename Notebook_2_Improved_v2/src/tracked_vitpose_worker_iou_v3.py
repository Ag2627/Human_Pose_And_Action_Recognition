
import os, sys
from pathlib import Path
# Prefer PyTorch's bundled OpenMP DLL in the child process without modifying
# the parent Jupyter environment or deleting any DLL/package.
torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
import ctypes
_openmp_handle = None
if os.name == "nt":
    _torch_iomp = torch_lib / "libiomp5md.dll"
    if _torch_iomp.is_file():
        try:
            _openmp_handle = ctypes.WinDLL(str(_torch_iomp))
            print("OPENMP_PRELOADED_FROM", _torch_iomp, flush=True)
        except OSError as _dll_error:
            print("OPENMP_PRELOAD_WARNING", repr(_dll_error), flush=True)
_dll_handles = []
if torch_lib.is_dir():
    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            _dll_handles.append(os.add_dll_directory(str(torch_lib)))
        except OSError:
            pass
for _name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_name] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"

import json, time, gc, traceback, hashlib, re
from pathlib import Path

cfg=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
cap=None
try:
    import torch
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    import numpy as np
    from ultralytics import YOLO
    project=Path(cfg["project_dir"])
    sys.path.insert(0,str(project))
    from src.vitpose_model import ViTPose, get_affine_transform, decode_heatmaps, IMG_H, IMG_W, IMG_MEAN, IMG_STD
    device=torch.device(cfg.get("device","cpu"))
    print("WORKER_START",flush=True)
    print("device",device,"weights",cfg["yolo_weights"],flush=True)
    # Load models once for this video; the process exits when the video finishes.
    yolo=YOLO(cfg["yolo_weights"])
    pose=ViTPose().to(device)
    try: ckpt=torch.load(cfg["vitpose_checkpoint"],map_location=device,weights_only=True)
    except TypeError: ckpt=torch.load(cfg["vitpose_checkpoint"],map_location=device)
    if "model_state" not in ckpt: raise KeyError("ViTPose checkpoint does not contain model_state")
    pose.load_state_dict(ckpt["model_state"],strict=True); pose.eval()
    print("MODELS_LOADED",flush=True)

    def estimate_pose(rgb,box):
        x1,y1,x2,y2=map(float,box); h,w=rgb.shape[:2]
        x1=float(np.clip(x1,0,w-1)); x2=float(np.clip(x2,0,w))
        y1=float(np.clip(y1,0,h-1)); y2=float(np.clip(y2,0,h))
        bw,bh=x2-x1,y2-y1
        if bw<2 or bh<2: return None
        center=np.array([(x1+x2)/2,(y1+y2)/2],dtype=np.float32)
        scale=max(bw,bh*IMG_W/IMG_H)*1.25
        trans=get_affine_transform(center,scale,0.0,(IMG_W,IMG_H))
        crop=cv2.warpAffine(rgb,trans,(IMG_W,IMG_H)).astype(np.float32)/255.0
        crop=(crop-IMG_MEAN)/IMG_STD
        tensor=torch.from_numpy(crop.transpose(2,0,1).copy()).float().unsqueeze(0).to(device)
        with torch.inference_mode(): hm=pose(tensor)[0].float().cpu().numpy()
        kp=decode_heatmaps(hm,center,scale).astype(np.float32)
        del tensor,hm,crop
        return kp if kp.shape==(17,3) else None

    video=Path(cfg["video_path"])
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError("Could not open video")
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    # Lightweight greedy IoU tracker. Uses YOLO predict(), not built-in tracking.
    # This follows the successful Notebook 02C pilot and avoids the failing
    # Ultralytics tracking integration observed in this Windows environment.
    sample_frames=[]; sample_observations=[]; track_counts={}; frame_i=0; started=time.time(); reached_eof=False
    active_tracks={}; next_track_id=1
    IOU_THRESHOLD=0.25
    MAX_TRACK_AGE=12

    def box_iou(a,b):
        xx1=max(float(a[0]),float(b[0])); yy1=max(float(a[1]),float(b[1]))
        xx2=min(float(a[2]),float(b[2])); yy2=min(float(a[3]),float(b[3]))
        inter=max(0.0,xx2-xx1)*max(0.0,yy2-yy1)
        area_a=max(0.0,float(a[2]-a[0]))*max(0.0,float(a[3]-a[1]))
        area_b=max(0.0,float(b[2]-b[0]))*max(0.0,float(b[3]-b[1]))
        return inter/max(area_a+area_b-inter,1e-8)

    def assign_ids(detections, current_frame):
        global next_track_id
        stale=[tid for tid,state in active_tracks.items()
               if current_frame-state["last_frame"]>MAX_TRACK_AGE]
        for tid in stale:
            active_tracks.pop(tid,None)
        det_ids=np.full(len(detections),-1,dtype=np.int64)
        candidates=[]
        for di,det_box in enumerate(detections):
            for tid,state in active_tracks.items():
                overlap=box_iou(det_box,state["box"])
                if overlap>=IOU_THRESHOLD:
                    candidates.append((overlap,di,tid))
        candidates.sort(reverse=True,key=lambda item:item[0])
        used_det=set(); used_track=set()
        for overlap,di,tid in candidates:
            if di in used_det or tid in used_track:
                continue
            det_ids[di]=tid
            used_det.add(di); used_track.add(tid)
        for di in range(len(detections)):
            if det_ids[di]<0:
                det_ids[di]=next_track_id
                next_track_id+=1
        for di,det_box in enumerate(detections):
            tid=int(det_ids[di])
            previous=active_tracks.get(tid,{"hits":0})
            active_tracks[tid]={"box":np.asarray(det_box,dtype=np.float32).copy(),
                                "last_frame":int(current_frame),
                                "hits":int(previous.get("hits",0))+1}
        return det_ids

    while True:
        if int(cfg.get("max_decoded_frames",0))>0 and frame_i>=int(cfg["max_decoded_frames"]):
            break
        ok,frame=cap.read()
        if not ok:
            reached_eof=True
            break
        result=yolo.predict(frame,classes=[0],conf=float(cfg["conf"]),
                            imgsz=int(cfg["imgsz"]),device=cfg.get("device","cpu"),
                            max_det=int(cfg["max_det"]),verbose=False)[0]
        if result.boxes is not None and len(result.boxes)>0:
            boxes=result.boxes.xyxy.cpu().numpy().astype(np.float32)
            confs=result.boxes.conf.cpu().numpy().astype(np.float32)
            ids=assign_ids(boxes,frame_i)
        else:
            boxes=np.empty((0,4),dtype=np.float32)
            confs=np.empty((0,),dtype=np.float32)
            ids=np.empty((0,),dtype=np.int64)
            assign_ids(boxes,frame_i)

        if frame_i % int(cfg["frame_step"]) == 0:
            observations=[]
            if len(boxes)>0:
                areas=np.maximum(0,boxes[:,2]-boxes[:,0])*np.maximum(0,boxes[:,3]-boxes[:,1])
                keep=np.argsort(areas)[::-1][:int(cfg["max_people_per_frame"])]
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                for ix in keep:
                    tid=int(ids[ix])
                    kp=estimate_pose(rgb,boxes[ix])
                    if kp is None:
                        continue
                    observations.append({"track_id":tid,"keypoints":kp,"box":boxes[ix].astype(np.float32),
                                         "score":float(confs[ix]),"area":float(areas[ix])})
                    track_counts[tid]=track_counts.get(tid,0)+1
                del rgb
            sample_frames.append(frame_i)
            sample_observations.append(observations)
        frame_i+=1
        del result
        if frame_i%50==0: print(f"PROGRESS frames={frame_i} samples={len(sample_frames)} tracks={len(track_counts)}",flush=True)
        if frame_i%100==0: gc.collect()
    cap.release(); cap=None

    target_id=max(track_counts,key=track_counts.get) if track_counts else -1
    n=len(sample_frames)
    keypoints=np.full((n,17,3),np.nan,dtype=np.float32)
    boxes=np.full((n,4),np.nan,dtype=np.float32)
    has_person=np.zeros(n,dtype=bool)
    target_ids=np.full(n,-1,dtype=np.int64)
    det_scores=np.zeros(n,dtype=np.float32)
    all_kps=[]; all_boxes=[]; all_frames=[]; all_ids=[]; all_scores=[]
    for j,obs_list in enumerate(sample_observations):
        for obs in obs_list:
            all_kps.append(obs["keypoints"]); all_boxes.append(obs["box"])
            all_frames.append(sample_frames[j]); all_ids.append(obs["track_id"]); all_scores.append(obs["score"])
        match=next((o for o in obs_list if o["track_id"]==target_id and target_id>=0),None)
        if match is not None:
            keypoints[j]=match["keypoints"]; boxes[j]=match["box"]
            has_person[j]=bool(np.isfinite(match["keypoints"][:,:2]).all(axis=1).sum()>=8)
            target_ids[j]=target_id; det_scores[j]=match["score"]
    all_kps=np.asarray(all_kps,dtype=np.float32).reshape(-1,17,3)
    all_boxes=np.asarray(all_boxes,dtype=np.float32).reshape(-1,4)
    all_frames=np.asarray(all_frames,dtype=np.int32); all_ids=np.asarray(all_ids,dtype=np.int64); all_scores=np.asarray(all_scores,dtype=np.float32)
    out=Path(cfg["output_path"]); out.parent.mkdir(parents=True,exist_ok=True)
    temp=out.with_name(out.stem+f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temp,keypoints=keypoints,boxes=boxes,frame_indices=np.asarray(sample_frames,dtype=np.int32),
        has_person=has_person,track_ids=target_ids,detection_scores=det_scores,
        all_keypoints=all_kps,all_boxes=all_boxes,all_frame_indices=all_frames,all_track_ids=all_ids,all_detection_scores=all_scores,
        label=np.asarray(cfg["label"]),video_path=np.asarray(cfg["video_path"]),target_track_id=np.asarray(target_id),
        fps=np.asarray(fps,dtype=np.float32),total_frames=np.asarray(total,dtype=np.int32),frame_step=np.asarray(cfg["frame_step"],dtype=np.int32),
        person_selection=np.asarray("most_pose_observations_by_persistent_track_id"),
        complete_video=np.asarray(reached_eof,dtype=np.bool_),max_decoded_frames=np.asarray(cfg.get("max_decoded_frames",0),dtype=np.int32),
        decoded_frames=np.asarray(frame_i,dtype=np.int32),max_people_per_frame=np.asarray(cfg["max_people_per_frame"],dtype=np.int32))
    os.replace(temp,out)
    result_obj={"status":"success","label":cfg["label"],"video_path":cfg["video_path"],"output_path":str(out),
        "decoded_frames":frame_i,"sampled_frames":n,"pose_successes":int(has_person.sum()),
        "pose_success_rate":float(has_person.mean()) if n else 0.0,"tracked_ids":len(track_counts),
        "target_track_id":int(target_id),"all_person_observations":len(all_ids),"complete_video":bool(reached_eof),
        "seconds":round(time.time()-started,2)}
    Path(cfg["result_path"]).write_text(json.dumps(result_obj,indent=2),encoding="utf-8")
    print("WORKER_SUCCESS",json.dumps(result_obj),flush=True)
except BaseException as exc:
    print("WORKER_EXCEPTION",repr(exc),flush=True); traceback.print_exc(); sys.exit(2)
finally:
    if cap is not None: cap.release()
    gc.collect()
