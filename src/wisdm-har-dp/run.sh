#!/bin/bash
# Federated Learning with Differential Privacy

# Setup Paths
SCRIPT_DIR=$(dirname "$(realpath "$0")")
WORKDIR="$SCRIPT_DIR/workdir"
DATASET_SOURCE="../../WISDM_ar_v1.1"  # Adjust path as needed

echo "Setting up Experiment Environment:"

# Create workdir
if [ -d "$WORKDIR" ]; then
    echo "Cleaning existing workdir..."
    rm -rf "$WORKDIR"
fi
mkdir -p "$WORKDIR"
echo "Created $WORKDIR"

# Copy Dataset
if [ -d "$DATASET_SOURCE" ]; then
    echo "Copying dataset from $DATASET_SOURCE..."
    cp -r "$DATASET_SOURCE" "$WORKDIR/"
else
    echo "Error: Dataset not found at $DATASET_SOURCE"
    echo "Please ensure the dataset path is correct."
    exit 1
fi

# Copy project files (excluding workdir)
echo "Copying project files..."
rsync -av --exclude='workdir' --exclude='__pycache__' --exclude='*.pyc' "$SCRIPT_DIR/" "$WORKDIR/" > /dev/null

# Rename directory if it has hyphens (Python can't import modules with hyphens)
if [ -d "$WORKDIR/wisdm-har-dp" ]; then
    echo "Renaming wisdm-har-dp to wisdm_har_dp..."
    mv "$WORKDIR/wisdm-har-dp" "$WORKDIR/wisdm_har_dp"
fi

echo "Setup Complete. Running Flower App:"
echo ""

# Run the Flower app inside workdir
cd "$WORKDIR" || exit

# Show what we have
echo "Workdir contents:"
ls -la
echo ""
echo "Package contents:"
ls -la wisdm_har_dp/
echo ""

# Install the package in editable mode so Flower can find it
echo "Installing package..."
pip install -e .

echo ""
echo "Running Flower..."
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
flwr run .

echo ""
echo "Checking outputs:"
echo "Looking for output files..."
find "$WORKDIR" -name "*.npy" -o -name "*.png" -o -name "*.txt" 2>/dev/null | head -20
echo ""
echo "Contents of outputs directory (if exists):"
ls -la "$WORKDIR/outputs/" 2>/dev/null || echo "  outputs/ directory not found"
echo ""

# Run visualization script if history exists
if [ -f "$WORKDIR/outputs/federated_training_history.npy" ]; then
    echo "Found training history, generating plots..."
    cd "$WORKDIR"
    python create_plots.py
else
    echo "No training history found. Training may not have saved results."
    echo "Check if server_module_loaded.txt exists:"
    cat "$WORKDIR/outputs/server_module_loaded.txt" 2>/dev/null || echo "  server module was NOT loaded"
fi
