#!/bin/bash

# ==============================================================================
# DATA DOWNLOAD & RESTRUCTURING PIPELINE
# ==============================================================================

# Locate the root directory of the local repository (where download_data.sh is located)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define immutable raw data paths according to the workspace design
RAW_TOP="$REPO_ROOT/data/raw/top"

# Guarantee the structural integrity of the raw data directory trees
mkdir -p "$RAW_TOP"

echo "System dependencies validation..."
echo "--------------------------------------------------"

# Runtime Dependency enforcement
MISSING_DEPS=()
if ! command -v aria2c &> /dev/null; then MISSING_DEPS+=("aria2"); fi
if ! command -v unzip &> /dev/null; then MISSING_DEPS+=("unzip"); fi

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    echo "Missing required tools: ${MISSING_DEPS[*]}. Installing..."
    sudo apt update && sudo apt install -y ${MISSING_DEPS[@]}
    if [ $? -ne 0 ]; then
        echo "Automated setup failed. Please install dependencies manually."
        exit 1
    fi
fi

echo "--------------------------------------------------"
echo "Populating raw data matrices using accelerated download threads..."
echo "--------------------------------------------------"

# Create a secure temporary workspace for orchestration mapping
F_TEMPORAL=$(mktemp)

# ==============================================================================
# Top Tagging Datasets
# ==============================================================================
# Mapping Top Tagging source files directly into their respective destination path
echo "$RAW_TOP" > "$RAW_TOP/dir.path" # Tracking context hook
echo "https://zenodo.org/records/2603256/files/test.h5?download=1" >> "$F_TEMPORAL"
echo "  dir=$RAW_TOP" >> "$F_TEMPORAL"
echo "  out=test.h5" >> "$F_TEMPORAL"
echo "https://zenodo.org/records/2603256/files/train.h5?download=1" >> "$F_TEMPORAL"
echo "  dir=$RAW_TOP" >> "$F_TEMPORAL"
echo "  out=train.h5" >> "$F_TEMPORAL"
echo "https://zenodo.org/records/2603256/files/val.h5?download=1" >> "$F_TEMPORAL"
echo "  dir=$RAW_TOP" >> "$F_TEMPORAL"
echo "  out=val.h5" >> "$F_TEMPORAL"

# Run aria2c reading from the unified mapped file configuration
# -c  : Resume any partially completed downloads, skipping fully completed ones gracefully
# --auto-file-renaming=false : Prevent downloading duplicate files with suffixes like .1, .2
# -j2 : Restrict parallel file execution to prevent connection dropping
# -x4 : Safe threshold limit per host for CERN/Zenodo protection
aria2c -c --auto-file-renaming=false -j2 -x4 -s4 --no-netrc -i "$F_TEMPORAL"
ESTADO=$?
rm "$F_TEMPORAL"

echo "--------------------------------------------------"
echo "Sanitizing filenames & cleaning URI query suffixes..."
echo "--------------------------------------------------"

# Post-processing Phase: Clean '?download=1' strings natively within each folder
for folder in "$RAW_QG" "$RAW_TOP" "$RAW_HIGGS"; do
    if [ -d "$folder" ]; then
        (
            cd "$folder" || exit
            for file in *\?download=1; do
                if [ -f "$file" ]; then
                    mv "$file" "${file%\?download=1}"
                fi
            done
        )
    fi
done

# ==============================================================================
# PHASE D: Post-Download Archive Extraction (HIGGS)
# ==============================================================================
# Nos aseguramos de ir a RAW_HIGGS de forma segura sin romper rutas previas o relativas
(
    cd "$RAW_HIGGS" || exit 1
    ZIP_FILE="higgs.zip"

    if [ -f "$ZIP_FILE" ]; then
        echo "Extracting $ZIP_FILE in raw repository branch..."
        unzip -oq "$ZIP_FILE"
        if [ $? -eq 0 ]; then
            echo "Extraction complete."
            rm "$ZIP_FILE"
            echo "Purged $ZIP_FILE to protect host storage space."
        else
            echo "Error unzipping $ZIP_FILE archive."
        fi
    fi
)

echo "--------------------------------------------------"
if [ $ESTADO -eq 0 ]; then
    echo "Portability setup completed. Data environment is synchronized!"
else
    echo "Download complete but some transfer streams might have warned."
fi
echo "--------------------------------------------------"
