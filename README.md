# 🎂 Birthday Video Enhancer (AI Powered)

Restore and upscale your old birthday memories to 4K using state-of-the-art AI. This tool uses **Real-ESRGAN** for background upscaling and **GFPGAN** for crystal-clear face restoration.

Optimized for performance by **TheCodingFarm** using a multi-threaded Python pipeline and Docker.

---

## 🛠 Prerequisites

### **For Windows Users:**
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Ensure you have an **NVIDIA GPU** and the latest [NVIDIA Drivers](https://www.nvidia.com/download/index.aspx) installed.
3. Open Docker Desktop Settings -> **Resources** -> **WSL Integration** -> Enable your default distro (usually Ubuntu).

### **For Linux Users:**
1. Install [Docker Engine](https://docs.docker.com/engine/install/).
2. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

---

## 🚀 How to Use

1. **Download/Clone** this repository to your machine.
2. **Prepare your Video**: 
   * Open the `Video` folder.
   * Drop the video file you want to enhance (mp4, mkv, mov, or avi) inside. 
   * *Note: The script processes all videos in the folder one by one.*
3. **Start Enhancing**:
   * **Windows:** Simply double-click **`run.bat`**.
   * **Linux/Manual:** Open a terminal in the root folder and run:
     ```bash
     docker compose up --build
     ```
4. **Relax**: AI upscaling is a heavy process. A progress bar will show you the estimated time. Once finished, your video will appear in the `Video` folder with the prefix `Enhanced_`.

---

## 💾 Features & Safety
* **Resume Support:** If your PC restarts or crashes, just run the tool again. It will automatically detect where it left off and resume from the last saved frame.
* **Disk Efficiency:** Uses high-quality JPEG frames for temporary storage to save disk space.
* **Parallel Processing:** Uses a 3-stage threaded pipeline (Reader -> GPU -> Writer) to ensure your GPU is always running at 100% efficiency.

---

## 📜 Acknowledgements & Licenses

This project acts as a wrapper and optimization layer for the following incredible research works. Please respect their respective licenses:

* **Real-ESRGAN**: Developed by [Xintao Wang et al.](https://github.com/xinntao/Real-ESRGAN). Used for background upscaling. (License: Apache 2.0)
* **GFPGAN**: Developed by [Tencent ARC Lab](https://github.com/TencentARC/GFPGAN). Used for blind face restoration. (License: Apache 2.0)
* **BasicSR**: Open-source image and video restoration toolbox. (License: Apache 2.0)

**Notice:** This repository includes pre-trained model weights (`RealESRGAN_x4plus.pth`) which are redistributed under the terms of the Apache 2.0 license. All credit for the model architecture and training goes to the original authors linked above.