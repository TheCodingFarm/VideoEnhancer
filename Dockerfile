# Use an official NVIDIA PyTorch image as the base
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
     \
    && rm -rf /var/lib/apt/lists/*
# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script and the model weights (if you have them locally)
COPY . .

# Create the Video directory
RUN mkdir -p Video

# Run the script
CMD ["python", "your_script_name.py"]