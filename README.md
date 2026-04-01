# 🎂 Birthday Video Enhancer (AI Powered)

Boost your old memories to 4K using Real-ESRGAN and GFPGAN face restoration. Developed by **TheCodingFarm**.

## 🛠 Prerequisites

Download the [RealESRGAN_x4plus.pth](https://release-assets.githubusercontent.com/github-production-release-asset/387326890/08f0e941-ebb7-48f0-9d6a-73e87b710e7e?sp=r&sv=2018-11-09&sr=b&spr=https&se=2026-04-01T16%3A36%3A17Z&rscd=attachment%3B+filename%3DRealESRGAN_x4plus.pth&rsct=application%2Foctet-stream&skoid=96c2d410-5711-43a1-aedd-ab1947aa7ab0&sktid=398a6654-997b-47e9-b12b-9515b896b4de&skt=2026-04-01T15%3A36%3A12Z&ske=2026-04-01T16%3A36%3A17Z&sks=b&skv=2018-11-09&sig=deavgMAIF51Ck6%2BtWwl4gi9gp1whPGzA%2FaJYzdLOx1E%3D&jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmVsZWFzZS1hc3NldHMuZ2l0aHVidXNlcmNvbnRlbnQuY29tIiwia2V5Ijoia2V5MSIsImV4cCI6MTc3NTA1OTU3MiwibmJmIjoxNzc1MDU3NzcyLCJwYXRoIjoicmVsZWFzZWFzc2V0cHJvZHVjdGlvbi5ibG9iLmNvcmUud2luZG93cy5uZXQifQ.n9-Ks9IsbwaZ8CxsJIo687QqDyWHpqRdl9CsD9AEaV4&response-content-disposition=attachment%3B%20filename%3DRealESRGAN_x4plus.pth&response-content-type=application%2Foctet-stream) and place it in the same folder alongside Enhance.py file

### For Windows Users:
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Ensure you have an **NVIDIA GPU** and the latest drivers installed.
3. Open Docker Desktop settings -> Resources -> WSL Integration -> Enable your distro.

### For Linux Users:
1. Install [Docker](https://docs.docker.com/engine/install/).
2. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

---

## 🚀 How to Use

1. **Download this folder** (or clone the repo).
2. **Place your video** (mp4 or mkv) inside the folder named `Video`. 
   * *Note: Ensure there is only ONE video in that folder at a time.*
3. **Run the Enhancer**:
   Open your terminal/command prompt in this folder and type:
   ```bash
   docker compose up --build