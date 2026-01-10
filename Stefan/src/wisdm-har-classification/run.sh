#!/bin/bash

# 1. Setup Paths
SCRIPT_DIR=$(dirname .)
WORKDIR="$SCRIPT_DIR/workdir"                 # The new workdir path
DATASET_SOURCE="../../../WISDM_ar_v1.1"       # Path to dataset (relative to current location)

echo "--- Setting up Experiment Environment ---"

# 2. Create workdir
if [ -d "$WORKDIR" ]; then
    echo "Cleaning existing workdir..."
    rm -rf "$WORKDIR"
fi
mkdir -p "$WORKDIR"
echo "Created $WORKDIR"

# 3. Copy Dataset
# We check if the dataset exists at the specified relative path
if [ -d "$DATASET_SOURCE" ]; then
    echo "Copying dataset from $DATASET_SOURCE..."
    cp -r "$DATASET_SOURCE" "$WORKDIR/"
else
    echo "Error: Dataset not found at $DATASET_SOURCE"
    echo "Please ensure you are running this script from the correct directory."
    exit 1
fi

# 4. Copy Python script and related files
# We copy everything from the script's directory into workdir, EXCLUDING workdir itself
# to prevent infinite recursion.
echo "Copying project files..."
# Using rsync for clean exclusion, falling back to cp if you prefer simple tools
# This command copies all files from SCRIPT_DIR to WORKDIR, excluding the WORKDIR folder itself
rsync -av --exclude='workdir' "$SCRIPT_DIR/" "$WORKDIR/" > /dev/null

echo "--- Setup Complete. Running Script ---"
echo ""

# 5. Run the Python Script inside workdir
cd "$WORKDIR" || exit

flwr run .
