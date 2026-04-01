import subprocess
import glob
import os
import sys
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

def merge_audio(input_video, enhanced_video, final_output):
    print("Merging original audio with enhanced video...")
    cmd = [
        'ffmpeg', '-y',
        '-i', enhanced_video,  # Video source (no audio)
        '-i', input_video,     # Audio source
        '-map', '0:v:0',       # Take video from first input
        '-map', '1:a:0?',      # Take audio from second input (if exists)
        '-c:v', 'copy',        # Don't re-encode video
        '-c:a', 'aac',         # Encode audio to AAC
        '-shortest',           # Match length
        final_output
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def enhance(TargetVideoFileName):
    print("Preparing Video...")
    OutputVideoFileName = "Enhanced - "+TargetVideoFileName # Changed extension to .mp4
    TempVideoFile = "temp_no_audio.mp4"
    cap = cv2.VideoCapture(TargetVideoFileName)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 2. Process First Frame to get dimensions
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Can't read first frame")

    # GFPGAN and RealESRGAN expect OpenCV BGR natively, no conversion needed!
    print("Enhancing first frame to determine resolution...")
    _, _, sr_bgr = face_enhancer.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)

    # 3. Setup Video Writer
    h, w = sr_bgr.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(TempVideoFile, fourcc, fps, (w, h))

    # Write first frame
    out.write(sr_bgr)

    # 4. Process Remaining Frames
    print("Started Enhancing Video (Press 'q' to quit early)...")
    frame_count = 1
    with tqdm(total=total_frames, initial=1, desc="Enhancing Video", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Enhance frame
            with SuppressStdout():
                _, _, sr_bgr = face_enhancer.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
            
            # Write and display
            out.write(sr_bgr)
            pbar.update(1)
            # Optional: Resize display windows so they fit on your screen (since it's 4x larger now)
            #display_frame = cv2.resize(sr_bgr, (w // 4, h // 4)) 
            #cv2.imshow('Video Output (Scaled Down for Preview)', display_frame)
            #cv2.imshow('Video Input', frame)
            
            #frame_count += 1
            #if frame_count % 10 == 0:
            #    print(f"Processed {frame_count} frames...")

            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

    # 5. Cleanup
    cap.release()
    out.release()
    #cv2.destroyAllWindows()
    print("Video Generated!\nCreating the final video file with Audio...")
    if os.path.exists(TargetVideoFileName):
        merge_audio(TargetVideoFileName, TempVideoFile, OutputVideoFileName)
        os.remove(TempVideoFile)
    print("Done! Video saved as:", OutputVideoFileName)
for f in video_files:
    enhance(f)
print("Exitting...")