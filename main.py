"""
Universal Multi-Camera CCTV People Tracking & Re-Identification System.
Works with ANY input video, webcam, RTSP stream, or multi-camera setup.
Detects both foreground visitors and background/passing pedestrians using YOLO11
and maintains persistent global identities using Spatio-Temporal Deep ReID.
"""
import os
import sys
import time
import argparse
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import config
from core.tracker import MultiCameraTracker
from ui.dashboard import SideDashboardRenderer

class CameraStream:
    def __init__(self, camera_id: str, source: str, frame_offset: int = 0, loop: bool = False):
        self.camera_id = camera_id
        self.source = source
        self.frame_offset = frame_offset
        self.loop = loop
        self.cap = None
        self.fps = 24.0
        self.width = 640
        self.height = 360
        self.total_frames = -1
        self._init_capture()

    def _init_capture(self):
        if self.source.isdigit():
            self.cap = cv2.VideoCapture(int(self.source))
        else:
            self.cap = cv2.VideoCapture(self.source)
            
        if self.cap is not None and self.cap.isOpened():
            val_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if val_fps and val_fps > 0:
                self.fps = val_fps
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self.frame_offset > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_offset)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None or not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        if not ret and not self.source.isdigit():
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_offset)
                ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


def parse_args():
    parser = argparse.ArgumentParser(description="Universal CCTV Multi-Camera People Tracking & ReID System")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="Path(s) to video files or webcam indices (e.g. --sources video/vid1.mp4)")
    parser.add_argument("--model", type=str, default=config.YOLO_MODEL_PATH,
                        help=f"Path to YOLO11 weights (default: {config.YOLO_MODEL_PATH})")
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD,
                        help="Confidence threshold for person detection (default: 0.22)")
    parser.add_argument("--demo", action="store_true", default=False,
                        help="Simulate 2-camera viewpoints from a single video source")
    parser.add_argument("--loop", action="store_true", default=False,
                        help="Loop video continuously")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run without displaying a GUI window (faster for batch/server)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to process before exiting (default: process until end of video)")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Force display of graphical video selector dialog")
    parser.add_argument("--no-gui", action="store_true", default=False,
                        help="Bypass GUI file selector and use default sample video")
    parser.add_argument("--exit-timeout", type=float, default=config.EXIT_TIMEOUT_SECONDS,
                        help=f"Inactivity timeout in seconds before marking a visitor as EXITED (default: {config.EXIT_TIMEOUT_SECONDS})")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename for the tracked video")
    return parser.parse_args()


def build_camera_streams(sources: Optional[List[str]], demo: bool, loop: bool) -> List[CameraStream]:
    streams = []
    default_vid = str(config.VIDEO_DIR / "vid1.mp4")
    
    if sources and len(sources) > 0:
        if len(sources) == 1 and demo:
            print(f"[INFO] Initializing 2 Simulated Camera Viewpoints from: {sources[0]}")
            streams.append(CameraStream("Cam-01", sources[0], frame_offset=0, loop=loop))
            streams.append(CameraStream("Cam-02", sources[0], frame_offset=200, loop=loop))
        else:
            for idx, src in enumerate(sources):
                cam_id = f"Cam-{idx+1:02d}"
                streams.append(CameraStream(cam_id, src, frame_offset=0, loop=loop))
    elif demo:
        print("[INFO] Multi-Camera Demo Mode Active: Initializing 2 viewpoints from default video...")
        streams.append(CameraStream("Cam-01", default_vid, frame_offset=0, loop=loop))
        streams.append(CameraStream("Cam-02", default_vid, frame_offset=200, loop=loop))
    else:
        print(f"[INFO] Tracking video: {default_vid}")
        streams.append(CameraStream("Cam-01", default_vid, frame_offset=0, loop=loop))
        
    return streams


def main():
    args = parse_args()
    
    # Launch GUI video selector dialog if no sources provided and not in headless/no-gui mode
    if (args.sources is None and not args.headless and not args.no_gui) or args.gui:
        try:
            from ui.video_selector_gui import launch_video_selector_gui
            gui_cfg = launch_video_selector_gui()
            if gui_cfg is None:
                print("[INFO] Video selection was cancelled by user. Exiting cleanly.")
                return
            args.sources = gui_cfg.get("sources", args.sources)
            args.demo = gui_cfg.get("demo", args.demo)
            args.loop = gui_cfg.get("loop", args.loop)
            args.model = gui_cfg.get("model", args.model)
            args.conf = gui_cfg.get("conf", args.conf)
            args.exit_timeout = gui_cfg.get("exit_timeout", getattr(args, "exit_timeout", config.EXIT_TIMEOUT_SECONDS))
        except Exception as e:
            print(f"[WARNING] Rich GUI selector encountered an issue: {e}")
            print("[INFO] Launching fallback Windows file selector...")
            try:
                import tkinter as tk
                from tkinter import filedialog
                root = tk.Tk()
                root.withdraw()
                picked = filedialog.askopenfilename(
                    title="Select Video to Track - AI CCTV Tracking",
                    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All Files", "*.*")]
                )
                root.destroy()
                if picked and picked.strip():
                    args.sources = [picked.strip()]
                else:
                    print("[INFO] No video selected. Exiting cleanly.")
                    return
            except Exception as fb_err:
                print(f"[WARNING] File selector error ({fb_err}). Exiting.")
                return

    print("=" * 74)
    print("       UNIVERSAL CCTV MULTI-CAMERA PEOPLE TRACKING & ReID SYSTEM       ")
    print("=" * 74)
    
    streams = build_camera_streams(args.sources, args.demo, args.loop)
    if not streams:
        print("[ERROR] No valid camera streams found. Exiting.")
        return

    if args.conf != config.CONFIDENCE_THRESHOLD:
        config.CONFIDENCE_THRESHOLD = args.conf

    print(f"[INFO] Initializing YOLO11 ({args.model}) & Spatio-Temporal ReID Memory Gallery...")
    tracker = MultiCameraTracker(model_path=args.model)
    dashboard_renderer = SideDashboardRenderer(width=config.DASHBOARD_WIDTH)
    
    num_cams = len(streams)
    first_w = streams[0].width
    first_h = streams[0].height
    
    if num_cams == 1:
        cam_target_h = min(config.MAX_DISPLAY_HEIGHT, first_h)
        cam_target_w = int(first_w * (cam_target_h / first_h))
        grid_w = cam_target_w
        grid_h = cam_target_h
    elif num_cams == 2:
        cam_target_h = config.MAX_DISPLAY_HEIGHT // 2
        cam_target_w = int(first_w * (cam_target_h / first_h))
        grid_w = cam_target_w
        grid_h = cam_target_h * 2
    else:
        cam_target_h = config.MAX_DISPLAY_HEIGHT // 2
        cam_target_w = int(first_w * (cam_target_h / first_h))
        grid_w = cam_target_w * 2
        grid_h = cam_target_h * 2
        
    total_w = grid_w + config.DASHBOARD_WIDTH
    total_h = grid_h
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = args.output if args.output else f"tracked_run_{timestamp_str}.mp4"
    out_path = os.path.join(str(config.OUTPUT_DIR), out_filename)
    
    video_fps = streams[0].fps if streams[0].fps > 0 else config.EXPORT_FPS
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
    video_writer = cv2.VideoWriter(out_path, fourcc, video_fps, (total_w, total_h))
    
    print(f"[INFO] Camera Streams: {num_cams} | Canvas: {total_w}x{total_h} @ {video_fps:.1f} FPS")
    print(f"[INFO] Output recording saved to: {out_path}")
    print("[INFO] Processing video frames. Press 'q' in the window to stop early.\n")
    
    sim_time = datetime.now()
    frame_idx = 0
    fps_start_time = time.time()
    
    window_name = "Martian CCTV Intelligence - Real-Time Multi-Camera Tracking"
    window_initialized = False
        
    try:
        while True:
            sim_time += timedelta(milliseconds=int(1000 / video_fps))
            batch_inputs = []
            all_streams_ended = True
            
            for stream in streams:
                ret, frame = stream.read()
                if ret and frame is not None:
                    all_streams_ended = False
                    frame = cv2.resize(frame, (cam_target_w, cam_target_h))
                else:
                    frame = np.zeros((cam_target_h, cam_target_w, 3), dtype=np.uint8)
                    cv2.putText(frame, f"{stream.camera_id} Signal Lost / Ended", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                batch_inputs.append((stream.camera_id, frame))
                
            if all_streams_ended:
                print("\n[INFO] End of video stream(s) reached.")
                break
                
            # Batched Spatio-Temporal Tracking + Deep ReID
            ann_frames = tracker.process_camera_batch(batch_inputs, sim_time)
            
            # Check departure timeouts
            tracker.journey_manager.check_timeouts(sim_time, timeout_seconds=args.exit_timeout)
            
            # Compose Video Grid
            if num_cams == 1:
                cam_canvas = ann_frames[0]
            elif num_cams == 2:
                cam_canvas = np.vstack([ann_frames[0], ann_frames[1]])
            else:
                top_row = np.hstack([ann_frames[0], ann_frames[1]])
                bot_cam2 = ann_frames[2] if len(ann_frames) > 2 else np.zeros_like(ann_frames[0])
                bot_cam3 = ann_frames[3] if len(ann_frames) > 3 else np.zeros_like(ann_frames[0])
                bot_row = np.hstack([bot_cam2, bot_cam3])
                cam_canvas = np.vstack([top_row, bot_row])
                
            # Render Side Dashboard Panel
            metrics = tracker.journey_manager.get_summary_metrics(
                current_time=sim_time,
                live_camera_counts=tracker.live_camera_counts
            )
            dashboard_panel = dashboard_renderer.render(
                height=cam_canvas.shape[0],
                metrics=metrics,
                current_time=sim_time
            )
            
            composite = np.hstack([cam_canvas, dashboard_panel])
            video_writer.write(composite)
            
            if not args.headless:
                if not window_initialized:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(window_name, composite)
                    display_w = min(1366, total_w)
                    display_h = int(total_h * (display_w / total_w))
                    cv2.resizeWindow(window_name, display_w, display_h)
                    window_initialized = True
                else:
                    cv2.imshow(window_name, composite)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("\n[INFO] User requested stop.")
                    break
                    
            frame_idx += 1
            if frame_idx % 50 == 0:
                elapsed = time.time() - fps_start_time
                fps = frame_idx / elapsed if elapsed > 0 else 0
                print(f"[STATUS] Frame {frame_idx:04d} | Speed: {fps:.1f} FPS | "
                      f"Total Entered: {metrics['total_entered']} | Active: {metrics['currently_present']} | "
                      f"Exited: {metrics['total_exited']}")
                      
            if args.max_frames and frame_idx >= args.max_frames:
                print(f"\n[INFO] Reached requested frame limit ({args.max_frames}). Finishing...")
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        for stream in streams:
            stream.release()
        tracker.journey_manager.flush_all_to_db(sim_time)
        video_writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
            
        print("\n" + "=" * 74)
        print(f"[SUCCESS] Tracking finished successfully.")
        print(f"[SUCCESS] Processed Video: {out_path}")
        print(f"[SUCCESS] Database updated: {config.DATABASE_PATH}")
        final_kpi = tracker.journey_manager.get_summary_metrics()
        print(f"[METRICS] Total Entered: {final_kpi['total_entered']} | Currently Present: {final_kpi['currently_present']} | Total Exited: {final_kpi['total_exited']} | Avg Dwell: {final_kpi['avg_dwell_str']}")
        print("=" * 74)


if __name__ == "__main__":
    main()