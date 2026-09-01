#!/bin/bash

# ==============================================================================
# DATA DOWNLOAD & RESTRUCTURING PIPELINE FOR QKANs-ML4SCI (GSoC 2026)
# ==============================================================================

# Locate the root directory of the local repository (where download_data.sh is located)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define immutable raw data paths according to the workspace design
RAW_QG="$REPO_ROOT/data/raw/quark_gluon"
RAW_TOP="$REPO_ROOT/data/raw/top"
RAW_HIGGS="$REPO_ROOT/data/raw/higgs_boson"

# Step 1: Guarantee the structural integrity of the raw data directory trees
mkdir -p "$RAW_QG"
mkdir -p "$RAW_TOP"
mkdir -p "$RAW_HIGGS"

echo "🚀 System dependencies validation..."
echo "--------------------------------------------------"

# Step 2: Runtime Dependency enforcement (aria2 and unzip checking)
MISSING_DEPS=()
if ! command -v aria2c &> /dev/null; then MISSING_DEPS+=("aria2"); fi
if ! command -v unzip &> /dev/null; then MISSING_DEPS+=("unzip"); fi

if [ ${#MISSING_DEPS[@]} -ne 0 ]; then
    echo "🔍 Missing required tools: ${MISSING_DEPS[*]}. Installing..."
    sudo apt update && sudo apt install -y ${MISSING_DEPS[@]}
    if [ $? -ne 0 ]; then
        echo "❌ Automated setup failed. Please install dependencies manually."
        exit 1
    fi
fi

echo "--------------------------------------------------"
echo "🛰️  Populating raw data matrices using accelerated download threads..."
echo "--------------------------------------------------"

# Create a secure temporary workspace for orchestration mapping
F_TEMPORAL=$(mktemp)

# ==============================================================================
# PHASE A: Top Tagging Datasets (Reference Tracks)
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

# ==============================================================================
# PHASE B: Higgs Boson Dataset (UCI Compression Hub)
# ==============================================================================
# Directing the HIGGS source file to its target structural bucket
echo "https://archive.ics.uci.edu/static/public/280/higgs.zip" >> "$F_TEMPORAL"
echo "  dir=$RAW_HIGGS" >> "$F_TEMPORAL"
echo "  out=higgs.zip" >> "$F_TEMPORAL"

# ==============================================================================
# PHASE C: Quark-Gluon Jet Parts (_0 to _19 Substructure Sheets)
# ==============================================================================
# Populating the 20 kinematic arrays iteratively into the dedicated directory
for i in {0..19}; do
    echo "https://zenodo.org/records/19362155/files/QG_jets_fp32_${i}.npz?download=1" >> "$F_TEMPORAL"
    echo "  dir=$RAW_QG" >> "$F_TEMPORAL"
    echo "  out=QG_jets_fp32_${i}.npz" >> "$F_TEMPORAL"
done

# Run aria2c reading from the unified mapped file configuration
# -c  : Resume any partially completed downloads, skipping fully completed ones gracefully
# --auto-file-renaming=false : Prevent downloading duplicate files with suffixes like .1, .2
# -j2 : Restrict parallel file execution to prevent connection dropping
# -x4 : Safe threshold limit per host for CERN/Zenodo protection
aria2c -c --auto-file-renaming=false -j2 -x4 -s4 --no-netrc -i "$F_TEMPORAL"
ESTADO=$?
rm "$F_TEMPORAL"

echo "--------------------------------------------------"
echo "🧹 Sanitizing filenames & cleaning URI query suffixes..."
echo "--------------------------------------------------"

# Post-processing Phase: Clean '?download=1' strings natively within each folder
for folder in "$RAW_QG" "$RAW_TOP" "$RAW_HIGGS"; do
    if [ -d "$folder" ]; then
        # El comando cd debe hacerse en una subshell o retornar para que no deje
        # al script en un directorio diferente que rompa las rutas relativas siguientes.
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
        echo "📦 Extracting $ZIP_FILE in raw repository branch..."
        unzip -oq "$ZIP_FILE"
        if [ $? -eq 0 ]; then
            echo "✅ Extraction complete."
            rm "$ZIP_FILE"
            echo "🗑️  Purged $ZIP_FILE to protect host storage space."
        else
            echo "❌ Error unzipping $ZIP_FILE archive."
        fi
    fi
)

echo "--------------------------------------------------"
if [ $ESTADO -eq 0 ]; then
    echo "🎉 Portability setup completed. Data environment is synchronized!"
else
    echo "⚠️  Download complete but some transfer streams might have warned."
fi
echo "--------------------------------------------------"
