
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

import json, time, gc, traceback
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
cap = None
try:
    import torch
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)
    import numpy as np
    import pandas as pd
    from ultralytics import YOLO
    print("WORKER_START",flush=True)
    print("python",sys.executable,flush=True)
    print("torch",torch.__version__,"cv2",cv2.__version__,flush=True)
    print("weights",cfg["weights"],flush=True)
    model=YOLO(cfg["weights"])
    print("YOLO_LOADED",flush=True)
    cap=cv2.VideoCapture(cfg["video_path"])
    if not cap.isOpened(): raise RuntimeError("OpenCV could not open source video")
    decoded=processed=det_frames=preview_count=0
    track_counts={}; rows=[]; start=time.time()
    outdir=Path(cfg["preview_dir"]); outdir.mkdir(parents=True,exist_ok=True)
    while decoded < int(cfg["max_decoded_frames"]):
        ok,frame=cap.read()
        if not ok: break
        frame_idx=decoded; decoded+=1
        if frame_idx % int(cfg["frame_stride"]) != 0: continue
        processed+=1
        result=model.track(frame,persist=True,tracker="bytetrack.yaml",classes=[0],conf=float(cfg["conf"]),
                           imgsz=int(cfg["imgsz"]),device="cpu",max_det=int(cfg["max_det"]),verbose=False)[0]
        if result.boxes is not None and len(result.boxes):
            boxes=result.boxes.xyxy.cpu().numpy(); confs=result.boxes.conf.cpu().numpy()
            ids=result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else np.full(len(boxes),-1,dtype=int)
            det_frames+=1
            for box,score,tid in zip(boxes,confs,ids):
                x1,y1,x2,y2=map(float,box); area=max(0.0,x2-x1)*max(0.0,y2-y1)
                rows.append({"frame_index":int(frame_idx),"track_id":int(tid),"confidence":float(score),
                             "x1":x1,"y1":y1,"x2":x2,"y2":y2,"area":area})
                if tid>=0: track_counts[int(tid)]=track_counts.get(int(tid),0)+1
            if preview_count<int(cfg["max_previews"]):
                canvas=frame.copy()
                for box,score,tid in zip(boxes,confs,ids):
                    x1,y1,x2,y2=map(int,box); cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(canvas,f"ID {tid} {score:.2f}",(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                cv2.imwrite(str(outdir/f"frame_{frame_idx:06d}.jpg"),canvas); preview_count+=1
        del result
        if processed%5==0: print(f"PROGRESS decoded={decoded} processed={processed} detections={det_frames} tracks={len(track_counts)}",flush=True)
        if processed%10==0: gc.collect()
    cap.release(); cap=None
    detpath=Path(cfg["detections_path"]); pd.DataFrame(rows).to_csv(detpath,index=False)
    longest=max(track_counts,key=track_counts.get) if track_counts else -1
    obj={"status":"success","video_path":cfg["video_path"],"label":cfg["label"],"decoded_frames":decoded,
         "processed_frames":processed,"detection_frames":det_frames,"detection_frame_rate":det_frames/max(processed,1),
         "tracked_ids":len(track_counts),"longest_track_id":int(longest),
         "longest_track_observations":int(track_counts.get(longest,0)),"preview_frames":preview_count,
         "detection_rows":len(rows),"seconds":round(time.time()-start,2)}
    Path(cfg["result_path"]).write_text(json.dumps(obj,indent=2),encoding="utf-8")
    print("WORKER_SUCCESS",json.dumps(obj),flush=True)
except BaseException as exc:
    print("WORKER_EXCEPTION",repr(exc),flush=True); traceback.print_exc(); sys.exit(2)
finally:
    if cap is not None: cap.release()
    gc.collect()
