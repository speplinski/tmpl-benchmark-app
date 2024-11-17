#!/bin/bash

# Copyright (c) Szymon P. Peplinski.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Use either wget or curl to download the checkpoints
if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    echo "Please install wget or curl to download the checkpoints."
    exit 1
fi

# Define the URLs for TMPL checkpoints
TMPL_BASE_URL="https://storage.googleapis.com/polish_landscape/checkpoints/npk_experiment_v42"
TMPL_net_D_url="${TMPL_BASE_URL}/80_net_D.pth"
TMPL_net_G_url="${TMPL_BASE_URL}/80_net_G.pth"

# Create the 'tmpl' directory if it doesn't exist
mkdir -p tmpl
cd tmpl || { echo "Failed to change to tmpl directory"; exit 1; }

# TMPL checkpoints
echo "Downloading net_D checkpoint..."
$CMD $TMPL_net_D_url || { echo "Failed to download checkpoint from $TMPL_net_D_url"; exit 1; }

echo "Downloading net_G checkpoint..."
$CMD $TMPL_net_G_url || { echo "Failed to download checkpoint from $TMPL_net_G_url"; exit 1; }

echo "All checkpoints are downloaded successfully."

cd ..
