"""
Tracker Benchmarking Script: ByteTrack vs BoT-SORT on CCTV Video vid1.mp4
Evaluates FPS, unique track counts, tracklet stability, and person continuity.
"""
import time
import cv2
import torch
import numpy as np
from typing import Dict, List, Tuple
from ultralytics import YOLO
import config

VIDEO_PATH = "video/vid1.mp4"

def evaluate_tracker(tracker_cfg: str, name: str, conf: float = 0.30) -> Dict:
    print(f"\n=======================================================")
    print(f"Benchmarking Tracker: {name} (config={tracker_cfg})")
    print(f"=======================================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(config.YOLO_MODEL_PATH)
    model.to(device)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 24.0
    
    frame_idx = 0
    track_history: Dict[int, List[int]] = {}
    track_bboxes: Dict[int, List[Tuple[int, int, int, int]]] = {}
    track_confs: Dict[int, List[float]] = {}
    
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        results = model.track(
            source=[frame],
            persist=True,
            tracker=tracker_cfg,
            classes=[config.PERSON_CLASS_ID],
            conf=conf,
            iou=0.45,
            verbose=False
        )
        
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                if box.id is None:
                    continue
                tid = int(box.id[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf_val = float(box.conf[0].cpu().numpy())
                
                h_box = xyxy[3] - xyxy[1]
                w_box = xyxy[2] - xyxy[0]
                if h_box < config.MIN_PERSON_HEIGHT or (h_box * w_box) < config.MIN_PERSON_AREA:
                    continue
                    
                if tid not in track_history:
                    track_history[tid] = []
                    track_bboxes[tid] = []
                    track_confs[tid] = []
                track_history[tid].append(frame_idx)
                track_bboxes[tid].append(tuple(xyxy))
                track_confs[tid].append(conf_val)
                
    elapsed = time.time() - start_time
    cap.release()
    
    fps = frame_idx / elapsed if elapsed > 0 else 0
    
    # Analyze tracks by duration
    major_tracks = {tid: frames for tid, frames in track_history.items() if len(frames) >= 30}
    short_tracks = {tid: frames for tid, frames in track_history.items() if len(frames) < 30}
    
    print(f"Finished {frame_idx} frames in {elapsed:.2f}s ({fps:.1f} FPS)")
    print(f"Total Unique Track IDs: {len(track_history)}")
    print(f"Major Tracks (>=30 frames): {len(major_tracks)} IDs")
    print(f"Fragmented/Short Tracks (<30 frames): {len(short_tracks)} IDs")
    
    print("\nTrack Details (Major Tracks):")
    for tid, frames in sorted(major_tracks.items(), key=lambda x: len(x[1]), reverse=True):
        f_start, f_end = frames[0], frames[-1]
        span = f_end - f_start + 1
        coverage = len(frames) / span * 100
        avg_c = np.mean(track_confs[tid])
        print(f"  Track {tid:02d}: {len(frames):4d} frames (f{f_start:03d}..f{f_end:03d}, span={span:3d}, continuity={coverage:5.1f}%, avg_conf={avg_c:.2f})")
        
    return {
        "name": name,
        "config": tracker_cfg,
        "elapsed_sec": elapsed,
        "fps": fps,
        "total_tracks": len(track_history),
        "major_tracks": len(major_tracks),
        "short_tracks": len(short_tracks),
        "track_history": track_history
    }

def main():
    print("=" * 65)
    print("MARTIAN CCTV TRACKER BENCHMARK (ByteTrack vs BoT-SORT)")
    print(f"Test Video: {VIDEO_PATH}")
    print(f"Detector: {config.YOLO_MODEL_PATH} on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 65)
    
    # 1. ByteTrack baseline (appearance-free, pure Kalman + Byte association)
    res_bytetrack = evaluate_tracker("bytetrack.yaml", "ByteTrack (Standard)")
    
    # 2. BoT-SORT (appearance + camera motion compensation)
    res_botsort_default = evaluate_tracker("botsort.yaml", "BoT-SORT (Default)")
    
    # 3. Custom BoT-SORT (our tuned config with ONNX ReID)
    res_botsort_custom = evaluate_tracker(config.TRACKER_CONFIG_PATH, "BoT-SORT (Custom ReID)")
    
    print("\n" + "=" * 65)
    print("COMPARATIVE SUMMARY TABLE")
    print("=" * 65)
    print(f"{'Tracker':<24} | {'FPS':<7} | {'Total IDs':<10} | {'Major IDs':<10} | {'Short IDs':<10}")
    print("-" * 65)
    for r in [res_bytetrack, res_botsort_default, res_botsort_custom]:
        print(f"{r['name']:<24} | {r['fps']:<7.1f} | {r['total_tracks']:<10} | {r['major_tracks']:<10} | {r['short_tracks']:<10}")
    print("=" * 65)

if __name__ == "__main__":
    main()
