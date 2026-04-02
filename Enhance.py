import subprocess
import glob
import os
import sys
import shutil
input_dir = "Video"
valid_extensions = ("*.mp4", "*.mkv", "*.mov", "*.avi")
video_files = []
for ext in valid_extensions:
    video_files.extend(glob.glob(os.path.join(input_dir, ext)))

if not video_files:
    print("❌ Error: No video found in the 'Video' folder. Please drop a file there and restart.")
    exit()

TargetVideoFileName = video_files[0]
print(f"✅ Found video: {TargetVideoFileName}")

import cv2
import torch
import numpy as np
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from tqdm import tqdm
import json
class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

isCuda = torch.cuda.is_available()
device = torch.device('cuda' if isCuda else 'cpu')
if isCuda:
    print("Working with CUDA")

# 1. Initialize Models
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
upsampler = RealESRGANer(
    scale=4,
    model_path='RealESRGAN_x4plus.pth', 
    model=model,
    tile=400,        
    tile_pad=10,
    pre_pad=0,
    half=isCuda
)

face_enhancer = GFPGANer(
    model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
    upscale=4,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=upsampler 
)

def merge_audio(frames_pattern, input_video_source, final_output, fps=30):
    """
    Stitches PNG frames into a video and attaches audio from the original source.
    """
    print(f"🎬 Stitching frames at {fps} FPS and merging audio...")
    
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),          # Set input framerate before the image input
        '-i', frames_pattern,            # Input: frame_%06d.png
        '-i', input_video_source,        # Input: Original video for audio
        '-c:v', 'libx264',               # Use H.264 codec
        '-crf', '18',                    # High quality (17-23 is standard; lower is better)
        '-pix_fmt', 'yuv420p',           # Ensures compatibility with most players
        '-map', '0:v:0',                 # Take video from the PNG sequence
        '-map', '1:a:0?',                # Take audio from source (optional ?)
        '-c:a', 'aac',                   # Encode audio to AAC
        '-b:a', '192k',                  # Decent audio bitrate
        '-shortest',                     # Match the shorter of the two streams
        final_output
    ]

    try:
        # Run the command
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Success! Final video saved to: {final_output}")
    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg Error:\n{e.stderr}")

def enhance(TargetVideoFileName):
    print("Preparing Video...")
    base_name = os.path.basename(TargetVideoFileName)
    OutputVideoFileName = os.path.join(input_dir, f"Enhanced_{base_name}")
    OutputVideoFileName = os.path.splitext(OutputVideoFileName)[0] + ".mp4"
    frames_dir = "Video/temp_frames"
    state_file = "Video/processing_state.json"
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(TargetVideoFileName)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = 0
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            data = json.load(f)
            start_frame = data.get("last_processed_frame", 0)
            print(f"Resuming from frame: {start_frame}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 2. Process First Frame to get dimensions
    print("Started Enhancing Video (Press 'q' to quit early)...")
    frame_count = 1
    with tqdm(total=total_frames, initial=start_frame, desc="Enhancing Video", unit="frame") as pbar:
        current_frame = start_frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Enhance frame
            with SuppressStdout():
                _, _, sr_bgr = face_enhancer.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
            
            # Write and display
            frame_name = os.path.join(frames_dir, f"frame_{current_frame:06d}.jpg")
            cv2.imwrite(frame_name, sr_bgr,[int(cv2.IMWRITE_JPEG_QUALITY), 95])
            current_frame += 1
            pbar.update(1)
            
            if current_frame % 10 == 0:
                with open(state_file, 'w') as f:
                    json.dump({"last_processed_frame": current_frame}, f)

    # 5. Cleanup
    cap.release()
    #cv2.destroyAllWindows()
    print("Video Generated!\nCreating the final video file with Audio...")
    if os.path.exists(TargetVideoFileName):
        merge_audio('Video/temp_frames/frame_%06d.jpg',TargetVideoFileName, OutputVideoFileName,fps=fps)
        #shutil.rmtree(frames_dir)
    if os.path.exists(state_file):
            os.remove(state_file)
    print("Done! Video saved as:", OutputVideoFileName)
for f in video_files:
    enhance(f)
print("Exitting...")