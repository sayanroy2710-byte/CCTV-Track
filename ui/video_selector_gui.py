"""
Modern Graphical Launcher & Video Selector Dialog for
AI-Based CCTV People Tracking & Dwell-Time Analytics System.
Allows the user to select any local video file, multiple videos, webcam, or sample video.
"""
import os
import sys
import subprocess
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2

import config

class VideoSelectorGUI:
    def __init__(self):
        self.result = None
        self.root = tk.Tk()
        self.root.title("AI CCTV Tracking — Select Video Input Source")
        self.root.geometry("740x640")
        self.root.minsize(700, 600)
        self.root.configure(bg="#0f172a")

        # Center window on screen
        self._center_window()

        # Build UI
        self._build_header()
        self._build_mode_selector()
        self._build_options()
        self._build_actions()

        # Set default sample video
        self.default_sample_path = str(config.VIDEO_DIR / "vid1.mp4")
        if os.path.exists(self.default_sample_path):
            self.single_file_var.set(self.default_sample_path)
            self._update_file_info(self.default_sample_path)

    def _center_window(self):
        self.root.update_idletasks()
        w = 740
        h = 640
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _build_header(self):
        header_frame = tk.Frame(self.root, bg="#0f172a", pady=10)
        header_frame.pack(fill="x", padx=25)

        title = tk.Label(header_frame, text="AI CCTV People Tracking & Analytics",
                         font=("Segoe UI", 16, "bold"), fg="#38bdf8", bg="#0f172a")
        title.pack(anchor="w")

        subtitle = tk.Label(header_frame, text="Select surveillance video file, multi-camera feeds, or webcam to begin tracking",
                            font=("Segoe UI", 9), fg="#94a3b8", bg="#0f172a")
        subtitle.pack(anchor="w", pady=(2, 0))

        sep = tk.Frame(self.root, height=1, bg="#334155")
        sep.pack(fill="x", padx=25, pady=(6, 12))

    def _build_mode_selector(self):
        card = tk.Frame(self.root, bg="#1e293b", bd=1, relief="solid", highlightbackground="#334155", highlightthickness=1)
        card.pack(fill="x", padx=25, pady=4)

        self.mode_var = tk.IntVar(value=1)

        radio_frame = tk.Frame(card, bg="#1e293b", pady=10, padx=14)
        radio_frame.pack(fill="x")

        r1 = tk.Radiobutton(radio_frame, text="Single Video File", variable=self.mode_var, value=1,
                            command=self._on_mode_change, bg="#1e293b", fg="#f8fafc", selectcolor="#0f172a",
                            activebackground="#1e293b", activeforeground="#38bdf8", font=("Segoe UI", 10, "bold"))
        r1.pack(side="left", padx=10)

        r2 = tk.Radiobutton(radio_frame, text="Multi-Camera Grid (Multiple Videos)", variable=self.mode_var, value=2,
                            command=self._on_mode_change, bg="#1e293b", fg="#f8fafc", selectcolor="#0f172a",
                            activebackground="#1e293b", activeforeground="#38bdf8", font=("Segoe UI", 10, "bold"))
        r2.pack(side="left", padx=10)

        r3 = tk.Radiobutton(radio_frame, text="Live Webcam / RTSP", variable=self.mode_var, value=3,
                            command=self._on_mode_change, bg="#1e293b", fg="#f8fafc", selectcolor="#0f172a",
                            activebackground="#1e293b", activeforeground="#38bdf8", font=("Segoe UI", 10, "bold"))
        r3.pack(side="left", padx=10)

        self.content_container = tk.Frame(card, bg="#1e293b", padx=16, pady=8)
        self.content_container.pack(fill="both", expand=True)

        # --- Panel 1: Single File ---
        self.panel_single = tk.Frame(self.content_container, bg="#1e293b")
        
        lbl_file = tk.Label(self.panel_single, text="Selected Video File Path:", bg="#1e293b", fg="#cbd5e1", font=("Segoe UI", 9, "bold"))
        lbl_file.pack(anchor="w", pady=(0, 4))

        file_row = tk.Frame(self.panel_single, bg="#1e293b")
        file_row.pack(fill="x")

        self.single_file_var = tk.StringVar()
        self.entry_single = tk.Entry(file_row, textvariable=self.single_file_var, font=("Consolas", 10),
                                     bg="#0f172a", fg="#f8fafc", insertbackground="#38bdf8", relief="solid", bd=1)
        self.entry_single.pack(side="left", fill="x", expand=True, ipady=5, padx=(0, 8))

        btn_browse = tk.Button(file_row, text="Browse Video...", font=("Segoe UI", 9, "bold"),
                               bg="#0284c7", fg="#ffffff", activebackground="#0369a1", activeforeground="#ffffff",
                               relief="flat", padx=14, pady=4, cursor="hand2", command=self._browse_single_file)
        btn_browse.pack(side="left")

        sub_row = tk.Frame(self.panel_single, bg="#1e293b", pady=8)
        sub_row.pack(fill="x")

        btn_sample = tk.Button(sub_row, text="Use Sample CCTV Video (vid1.mp4)", font=("Segoe UI", 9),
                               bg="#334155", fg="#cbd5e1", activebackground="#475569", activeforeground="#ffffff",
                               relief="flat", padx=10, pady=3, cursor="hand2",
                               command=self._use_default_sample)
        btn_sample.pack(side="left")

        self.lbl_info = tk.Label(sub_row, text="", bg="#1e293b", fg="#38bdf8", font=("Segoe UI", 8))
        self.lbl_info.pack(side="right")

        # --- Panel 2: Multiple Files ---
        self.panel_multi = tk.Frame(self.content_container, bg="#1e293b")
        lbl_multi = tk.Label(self.panel_multi, text="Selected Camera Video Files:", bg="#1e293b", fg="#cbd5e1", font=("Segoe UI", 9, "bold"))
        lbl_multi.pack(anchor="w", pady=(0, 4))

        self.multi_listbox = tk.Listbox(self.panel_multi, bg="#0f172a", fg="#f8fafc", font=("Consolas", 9),
                                        height=4, selectbackground="#0284c7", relief="solid", bd=1)
        self.multi_listbox.pack(fill="x", pady=(0, 6))

        multi_btn_row = tk.Frame(self.panel_multi, bg="#1e293b")
        multi_btn_row.pack(fill="x")

        btn_add_multi = tk.Button(multi_btn_row, text="Select Multiple Videos...", font=("Segoe UI", 9, "bold"),
                                  bg="#0284c7", fg="#ffffff", relief="flat", padx=10, pady=3, cursor="hand2",
                                  command=self._browse_multi_files)
        btn_add_multi.pack(side="left", padx=(0, 8))

        btn_clear_multi = tk.Button(multi_btn_row, text="Clear List", font=("Segoe UI", 8),
                                    bg="#334155", fg="#cbd5e1", relief="flat", padx=8, pady=3, cursor="hand2",
                                    command=self._clear_multi_files)
        btn_clear_multi.pack(side="left")

        # --- Panel 3: Webcam / RTSP ---
        self.panel_cam = tk.Frame(self.content_container, bg="#1e293b")
        lbl_cam = tk.Label(self.panel_cam, text="Webcam Device Index (e.g. 0, 1) or RTSP Stream URL:", bg="#1e293b", fg="#cbd5e1", font=("Segoe UI", 9, "bold"))
        lbl_cam.pack(anchor="w", pady=(0, 4))

        self.cam_source_var = tk.StringVar(value="0")
        self.entry_cam = tk.Entry(self.panel_cam, textvariable=self.cam_source_var, font=("Consolas", 10),
                                  bg="#0f172a", fg="#f8fafc", insertbackground="#38bdf8", relief="solid", bd=1)
        self.entry_cam.pack(fill="x", ipady=4, pady=(0, 6))

        cam_hints = tk.Label(self.panel_cam, text="Examples: '0' for default USB webcam, or 'rtsp://admin:pass@192.168.1.100:554/stream'",
                             bg="#1e293b", fg="#94a3b8", font=("Segoe UI", 8))
        cam_hints.pack(anchor="w")

        self._on_mode_change()

    def _on_mode_change(self):
        m = self.mode_var.get()
        self.panel_single.pack_forget()
        self.panel_multi.pack_forget()
        self.panel_cam.pack_forget()

        if m == 1:
            self.panel_single.pack(fill="x")
        elif m == 2:
            self.panel_multi.pack(fill="x")
        elif m == 3:
            self.panel_cam.pack(fill="x")

    def _browse_single_file(self):
        initial = str(config.VIDEO_DIR if config.VIDEO_DIR.exists() else Path.home())
        path = filedialog.askopenfilename(
            title="Select CCTV Video to Track",
            initialdir=initial,
            filetypes=[
                ("Video Files (*.mp4, *.avi, *.mkv, *.mov)", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv"),
                ("MP4 Videos (*.mp4)", "*.mp4"),
                ("All Files (*.*)", "*.*")
            ]
        )
        if path:
            self.single_file_var.set(path)
            self._update_file_info(path)

    def _use_default_sample(self):
        if os.path.exists(self.default_sample_path):
            self.single_file_var.set(self.default_sample_path)
            self._update_file_info(self.default_sample_path)
        else:
            messagebox.showwarning("File Not Found", f"Sample video not found at:\n{self.default_sample_path}")

    def _browse_multi_files(self):
        paths = filedialog.askopenfilenames(
            title="Select Multiple Camera Videos for Grid Tracking",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mkv *.mov *.wmv"),
                ("All Files", "*.*")
            ]
        )
        if paths:
            self.multi_listbox.delete(0, tk.END)
            for idx, p in enumerate(paths):
                self.multi_listbox.insert(tk.END, f"Cam-{idx+1:02d}: {p}")

    def _clear_multi_files(self):
        self.multi_listbox.delete(0, tk.END)

    def _update_file_info(self, path):
        if os.path.exists(path):
            try:
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                duration_sec = total / fps if fps and fps > 0 else 0
                dur_str = f"{int(duration_sec // 60):02d}m {int(duration_sec % 60):02d}s"
                size_mb = os.path.getsize(path) / (1024 * 1024)
                self.lbl_info.config(text=f"{w}x{h} | {dur_str} | {size_mb:.1f} MB")
            except Exception:
                self.lbl_info.config(text="Video selected")

    def _build_options(self):
        card = tk.Frame(self.root, bg="#1e293b", bd=1, relief="solid", highlightbackground="#334155", highlightthickness=1)
        card.pack(fill="x", padx=25, pady=8)

        card_title = tk.Label(card, text="Tracking Parameters & Viewpoint Settings",
                              bg="#1e293b", fg="#94a3b8", font=("Segoe UI", 10, "bold"))
        card_title.pack(anchor="w", padx=16, pady=(10, 6))

        grid = tk.Frame(card, bg="#1e293b", padx=16, pady=4)
        grid.pack(fill="x")

        lbl_mod = tk.Label(grid, text="Detection Model:", bg="#1e293b", fg="#cbd5e1", font=("Segoe UI", 9))
        lbl_mod.grid(row=0, column=0, sticky="w", pady=4, padx=(0, 6))

        default_model_name = os.path.basename(config.YOLO_MODEL_PATH)
        self.model_var = tk.StringVar(value=default_model_name)
        models = ["yolo11m.pt", "yolo11n.pt", "yolo11l.pt"]
        if default_model_name not in models:
            models.insert(0, default_model_name)
        self.combo_model = ttk.Combobox(grid, textvariable=self.model_var, values=models, state="readonly", width=14)
        self.combo_model.grid(row=0, column=1, sticky="w", pady=4, padx=(0, 20))

        lbl_conf = tk.Label(grid, text="Confidence Gate:", bg="#1e293b", fg="#cbd5e1", font=("Segoe UI", 9))
        lbl_conf.grid(row=0, column=2, sticky="w", pady=4, padx=(0, 6))

        self.conf_var = tk.DoubleVar(value=config.CONFIDENCE_THRESHOLD)
        self.spin_conf = ttk.Spinbox(grid, from_=0.15, to=0.80, increment=0.02, textvariable=self.conf_var, width=6)
        self.spin_conf.grid(row=0, column=3, sticky="w", pady=4)

        lbl_timeout = tk.Label(grid, text="Exit Waiting Timer (s):", bg="#1e293b", fg="#cbd5e1", font=("Segoe UI", 9))
        lbl_timeout.grid(row=1, column=0, sticky="w", pady=4, padx=(0, 6))

        self.timeout_var = tk.DoubleVar(value=config.EXIT_TIMEOUT_SECONDS)
        self.spin_timeout = ttk.Spinbox(grid, from_=1.0, to=120.0, increment=1.0, textvariable=self.timeout_var, width=18)
        self.spin_timeout.grid(row=1, column=1, sticky="w", pady=4, padx=(0, 20))

        check_row = tk.Frame(card, bg="#1e293b", padx=16, pady=6)
        check_row.pack(fill="x", pady=(0, 6))

        self.demo_var = tk.BooleanVar(value=False)
        c_demo = tk.Checkbutton(check_row, text="Simulate 2-Camera Topology Demo (--demo)", variable=self.demo_var,
                                bg="#1e293b", fg="#f8fafc", selectcolor="#0f172a",
                                activebackground="#1e293b", activeforeground="#38bdf8", font=("Segoe UI", 9))
        c_demo.pack(side="left", padx=(0, 20))

        self.loop_var = tk.BooleanVar(value=False)
        c_loop = tk.Checkbutton(check_row, text="Continuous Video Loop (--loop)", variable=self.loop_var,
                                bg="#1e293b", fg="#f8fafc", selectcolor="#0f172a",
                                activebackground="#1e293b", activeforeground="#38bdf8", font=("Segoe UI", 9))
        c_loop.pack(side="left")

    def _build_actions(self):
        action_frame = tk.Frame(self.root, bg="#0f172a", pady=16)
        action_frame.pack(fill="x", padx=25)

        btn_start = tk.Button(action_frame, text="Start Live CCTV Tracking", font=("Segoe UI", 11, "bold"),
                              bg="#16a34a", fg="#ffffff", activebackground="#15803d", activeforeground="#ffffff",
                              relief="flat", padx=24, pady=8, cursor="hand2", command=self._on_start)
        btn_start.pack(side="left", padx=(0, 12))

        btn_web = tk.Button(action_frame, text="Open Web Portal", font=("Segoe UI", 10),
                            bg="#0284c7", fg="#ffffff", activebackground="#0369a1", activeforeground="#ffffff",
                            relief="flat", padx=16, pady=8, cursor="hand2", command=self._open_web_portal)
        btn_web.pack(side="left", padx=(0, 12))

        btn_cancel = tk.Button(action_frame, text="Cancel", font=("Segoe UI", 10),
                               bg="#334155", fg="#cbd5e1", activebackground="#475569", activeforeground="#ffffff",
                               relief="flat", padx=16, pady=8, cursor="hand2", command=self._on_cancel)
        btn_cancel.pack(side="right")

    def _on_start(self):
        m = self.mode_var.get()
        sources = []

        if m == 1:
            val = self.single_file_var.get().strip()
            if not val:
                messagebox.showerror("Error", "Please select or browse for a video file.")
                return
            if not os.path.exists(val):
                messagebox.showerror("Error", f"Specified video file does not exist:\n{val}")
                return
            sources = [val]
        elif m == 2:
            count = self.multi_listbox.size()
            if count == 0:
                messagebox.showerror("Error", "Please select at least one video file for multi-camera tracking.")
                return
            for i in range(count):
                item = self.multi_listbox.get(i)
                parts = item.split(": ", 1)
                p = parts[1] if len(parts) > 1 else parts[0]
                sources.append(p.strip())
        elif m == 3:
            val = self.cam_source_var.get().strip()
            if not val:
                messagebox.showerror("Error", "Please provide a camera index or RTSP URL.")
                return
            sources = [val]

        selected_model = self.model_var.get().strip()
        if not os.path.isabs(selected_model):
            cand = config.BASE_DIR / selected_model
            model_path = str(cand) if cand.exists() else selected_model
        else:
            model_path = selected_model

        self.result = {
            "sources": sources,
            "demo": self.demo_var.get(),
            "loop": self.loop_var.get(),
            "model": model_path,
            "conf": float(self.conf_var.get()),
            "exit_timeout": float(self.timeout_var.get()),
        }
        self.root.destroy()

    def _open_web_portal(self):
        webbrowser.open("http://localhost:8501")
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", "web_dashboard.py", "--server.headless=true"],
                         cwd=str(config.BASE_DIR), shell=False)

    def _on_cancel(self):
        self.result = None
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.result


def launch_video_selector_gui():
    """
    Launches the video selector GUI.
    Returns: dict with selected sources and parameters, or None if cancelled.
    """
    app = VideoSelectorGUI()
    return app.run()

if __name__ == "__main__":
    res = launch_video_selector_gui()
    print("Selected:", res)
