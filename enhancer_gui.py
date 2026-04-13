import os
import sys
import json
import subprocess
import threading
import queue
import time
from datetime import timedelta

import cv2
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import sys

def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for Nuitka/pyinstaller onefile mode """
    base_path = os.path.dirname(__file__)
    return os.path.join(base_path, relative_path)

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from gfpgan import GFPGANer
except ImportError:
    print("Error: Missing AI libraries. Please install requirements.txt")

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class VideoEnhancerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VisionUp AI - Pro Video Enhancer")
        self.root.geometry("1100x800")
        self.root.configure(bg="#1a1a1a")

        # Configuration
        self.input_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.preview_mode = tk.StringVar(value="both") # none, generated, both
        self.is_processing = False
        self.interrupted = False
        
        # Queues for thread communication
        self.preview_queue = queue.Queue(maxsize=2)
        self.progress_queue = queue.Queue()
        
        self.setup_ui()
        self.check_cuda()

    def check_cuda(self):
        self.is_cuda = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if self.is_cuda else "CPU"
        self.status_label.config(text=f"Ready | Device: {device_name}")

    def setup_ui(self):
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#1a1a1a")
        style.configure("TLabel", background="#1a1a1a", foreground="#ffffff", font=("Inter", 10))
        style.configure("Header.TLabel", font=("Inter", 14, "bold"))
        style.configure("TButton", font=("Inter", 10))
        style.configure("Action.TButton", font=("Inter", 11, "bold"), padding=10)

        # Main Container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(header_frame, text="VisionUp AI", style="Header.TLabel").pack(side=tk.LEFT)
        self.status_label = ttk.Label(header_frame, text="Checking system...", foreground="#888888")
        self.status_label.pack(side=tk.RIGHT)

        # File Selection
        file_frame = ttk.LabelFrame(main_frame, text=" Project Configuration ", padding="15")
        file_frame.pack(fill=tk.X, pady=10)

        ttk.Label(file_frame, text="Input Video:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.input_path, width=80).grid(row=0, column=1, padx=10)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2)

        ttk.Label(file_frame, text="Output Folder:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(file_frame, textvariable=self.output_dir, width=80).grid(row=1, column=1, padx=10)
        ttk.Button(file_frame, text="Browse", command=self.browse_output).grid(row=1, column=2)

        # Settings
        settings_frame = ttk.LabelFrame(main_frame, text=" Enhancement Settings ", padding="15")
        settings_frame.pack(fill=tk.X, pady=10)

        ttk.Label(settings_frame, text="Preview Mode:").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(settings_frame, text="None", variable=self.preview_mode, value="none").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(settings_frame, text="Enhanced Only", variable=self.preview_mode, value="generated").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(settings_frame, text="Side-by-Side", variable=self.preview_mode, value="both").pack(side=tk.LEFT, padx=10)

        # Preview Area
        self.preview_container = ttk.Frame(main_frame, height=400)
        self.preview_container.pack(fill=tk.BOTH, expand=True, pady=20)
        
        self.preview_canvas = tk.Canvas(self.preview_container, bg="#000000", highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        # Progress Area
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=10)

        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)

        self.progress_label = ttk.Label(progress_frame, text="Waiting for input...", foreground="#aaaaaa")
        self.progress_label.pack(side=tk.LEFT)
        
        self.time_label = ttk.Label(progress_frame, text="", foreground="#aaaaaa")
        self.time_label.pack(side=tk.RIGHT)

        # Controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=10)

        self.start_btn = ttk.Button(controls_frame, text="START ENHANCEMENT", style="Action.TButton", command=self.start_processing)
        self.start_btn.pack(side=tk.RIGHT, padx=5)

        self.stop_btn = ttk.Button(controls_frame, text="STOP", state=tk.DISABLED, command=self.stop_processing)
        self.stop_btn.pack(side=tk.RIGHT, padx=5)

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mkv *.mov *.avi")])
        if path:
            self.input_path.set(path)
            if not self.output_dir.get():
                self.output_dir.set(os.path.dirname(path))

    def browse_output(self):
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def start_processing(self):
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input video.")
            return
        
        self.is_processing = True
        self.interrupted = False
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Start processing thread
        self.thread = threading.Thread(target=self.process_video, daemon=True)
        self.thread.start()
        
        # Start UI update loop
        self.root.after(100, self.update_ui_loop)

    def stop_processing(self):
        if messagebox.askyesno("Stop", "Are you sure you want to stop processing? Progress will be saved."):
            self.interrupted = True
            self.status_label.config(text="Stopping... please wait.")

    def update_ui_loop(self):
        if not self.is_processing:
            return

        # Update Progress
        try:
            while True:
                msg = self.progress_queue.get_nowait()
                if msg['type'] == 'progress':
                    self.progress_bar['value'] = msg['value']
                    self.progress_label.config(text=msg['text'])
                    self.time_label.config(text=msg['time'])
                elif msg['type'] == 'done':
                    self.finish_processing(msg['success'], msg.get('message', ''))
                    return
        except queue.Empty:
            pass

        # Update Preview
        try:
            while True:
                frame_data = self.preview_queue.get_nowait()
                self.display_preview(frame_data)
        except queue.Empty:
            pass

        self.root.after(30, self.update_ui_loop)

    def display_preview(self, frames):
        # frames is a list of [original, enhanced] or just [enhanced]
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        if canvas_width < 100 or canvas_height < 100:
            return

        images = []
        for f in frames:
            # Convert BGR to RGB
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(f_rgb)
            images.append(img)

        if len(images) == 2:
            # Side by side
            w, h = images[0].size
            combined = Image.new('RGB', (w*2 + 10, h))
            combined.paste(images[0], (0, 0))
            combined.paste(images[1], (w + 10, 0))
            display_img = combined
        else:
            display_img = images[0]

        # Scale to fit canvas while maintaining aspect ratio
        img_w, img_h = display_img.size
        ratio = min(canvas_width/img_w, canvas_height/img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        
        display_img = display_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(display_img)
        
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(canvas_width//2, canvas_height//2, image=self.tk_img, anchor=tk.CENTER)

    def finish_processing(self, success, message):
        self.is_processing = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress_bar['value'] = 0
        
        if success:
            messagebox.showinfo("Success", message or "Video enhancement completed successfully!")
        elif not self.interrupted:
            messagebox.showerror("Error", message or "An error occurred during processing.")
        
        self.status_label.config(text="Ready")

    def process_video(self):
        try:
            target_path = self.input_path.get()
            output_dir = self.output_dir.get()
            base_name = os.path.basename(target_path)
            output_video = os.path.join(output_dir, f"Enhanced_{os.path.splitext(base_name)[0]}.mp4")
            
            temp_dir = os.path.join(output_dir, "temp_frames")
            state_file = os.path.join(output_dir, "processing_state.json")
            os.makedirs(temp_dir, exist_ok=True)

            # Init Models
            self.progress_queue.put({'type': 'progress', 'value': 0, 'text': "Initializing AI Models...", 'time': ""})
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=4,
                model_path=get_resource_path('RealESRGAN_x4plus.pth'), 
                model=model,
                tile=400,        
                tile_pad=10,
                pre_pad=0,
                half=self.is_cuda
            )

            face_enhancer = GFPGANer(
                model_path=get_resource_path('GFPGANv1.3.pth'),
                upscale=4,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler 
            )

            # Video Info
            cap = cv2.VideoCapture(target_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            start_frame = 0
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    start_frame = data.get("last_processed_frame", 0)

            # Processing
            cap = cv2.VideoCapture(target_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            start_time = time.time()
            
            for i in range(start_frame, total_frames):
                if self.interrupted:
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Enhance
                with SuppressStdout():
                    _, _, sr_bgr = face_enhancer.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
                
                # Save
                frame_name = os.path.join(temp_dir, f"frame_{i:06d}.jpg")
                cv2.imwrite(frame_name, sr_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
                # Update State
                if i % 10 == 0:
                    with open(state_file, 'w') as f:
                        json.dump({"last_processed_frame": i}, f)

                # UI Updates
                elapsed = time.time() - start_time
                frames_done = i - start_frame + 1
                fps_avg = frames_done / elapsed if elapsed > 0 else 0
                remaining_frames = total_frames - i
                eta = remaining_frames / fps_avg if fps_avg > 0 else 0
                
                eta_str = str(timedelta(seconds=int(eta)))
                progress_pct = (i / total_frames) * 100
                
                self.progress_queue.put({
                    'type': 'progress', 
                    'value': progress_pct, 
                    'text': f"Processing frame {i}/{total_frames} ({fps_avg:.2f} fps)",
                    'time': f"ETA: {eta_str}"
                })

                # Preview
                mode = self.preview_mode.get()
                if mode == "generated":
                    self.preview_queue.put([sr_bgr])
                elif mode == "both":
                    self.preview_queue.put([frame, sr_bgr])

            cap.release()

            if not self.interrupted:
                # Finalize
                self.progress_queue.put({'type': 'progress', 'value': 99, 'text': "Merging audio and stitching video...", 'time': ""})
                self.merge_audio(os.path.join(temp_dir, 'frame_%06d.jpg'), target_path, output_video, int(fps))
                
                if os.path.exists(state_file):
                    os.remove(state_file)
                
                self.progress_queue.put({'type': 'done', 'success': True, 'message': f"Video saved to:\n{output_video}"})
            else:
                self.progress_queue.put({'type': 'done', 'success': False, 'message': "Processing stopped by user."})

        except Exception as e:
            self.progress_queue.put({'type': 'done', 'success': False, 'message': f"Error: {str(e)}"})

    def merge_audio(self, frames_pattern, input_video_source, final_output, fps=30):
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', frames_pattern,
            '-i', input_video_source,
            '-c:v', 'libx264',
            '-crf', '18',
            '-pix_fmt', 'yuv420p',
            '-map', '0:v:0',
            '-map', '1:a:0?',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            final_output
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoEnhancerGUI(root)
    root.mainloop()
