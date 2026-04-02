#!/bin/bash

# Path to the venv inside the container
VENV_PATH="/pyvenv"

# 1. Create venv if it doesn't exist in the volume
if [ ! -d "$VENV_PATH/bin" ]; then
    echo "Creating virtual environment in volume..."
    python -m venv $VENV_PATH
fi

# 2. Activate the venv
source $VENV_PATH/bin/activate

# 3. Install/Update requirements 
# (Pip will skip if already satisfied, making this fast)
echo "Syncing requirements..."
pip install --no-cache-dir -r /app/requirements.txt

# 4. Apply your BasicSR fix inside the venv
# This ensures the fix persists in your volume
sed -i 's/torchvision\.transforms\.functional_tensor/torchvision.transforms.functional/g' \
    $(python -c "import site; print(site.getsitepackages()[0])")/basicsr/data/degradations.py

# 5. Execute the main script
echo "Starting Enhancement..."
exec python Enhance.py