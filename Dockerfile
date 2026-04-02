FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies (these stay in the image)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and the model
COPY requirements.txt .
COPY ./RealESRGAN_x4plus.pth .
COPY Enhance.py .

# Copy a startup script (we will create this next)
COPY entrypoint.sh /entrypoint.sh
RUN dos2unix entrypoint.sh /entrypoint.sh && chmod +x /entrypoint.sh

RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]