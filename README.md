# Quantum Sinusoidal-Kolmogorov-Arnold-Networks-for-High-Energy-Physics
The purpose of the repository is to write the code developed for the project QKAN for GSoC at ML4SCI in 2026

---

## Environment
Follow the next command to install.

```bash
pip install uv
```

Create the virtual environment
```bash
uv sync
```

---

## Structure

The current file includes
* notebooks: dir with initial exploration
* src: dir with the code for the project, including data processing, model training and evaluation.
* scripts: dir with utility scripts for data downloading, preprocessing, and other tasks.
* README.md: this file, which provides an overview of the project, instructions for setup and usage, and other relevant information.

**Main files:**
* `src/architectures/hep_kan.py`: Contains the implementation of the HEP-KAN model architecture, based on the Kolmogorov-Arnold representation, with methods with some modifications for the specific use case of high-energy physics data.
* `scripts/train_kan.py`: Script for training the HEP-KAN model, including data loading, model initialization, training loop, evaluation, and saving of results.
* `src/architectures/classic_kan.py`: Contains the base KAN model implementation, which is extended by the HEP-KAN model.

The file `src/architectures/hep_kan.py` is the core of the model implementation, while `scripts/train_kan.py` is the main entry point for training the model on the provided datasets.
---

## Download datasets
To improve the time of download we use aria2. We also need unzip to extract the Higgs file.

Before running the download script, ensure you have `aria2` and `unzip` installed on your WSL/Linux environment. 

### Installation on Ubuntu/Debian (WSL):
Run the following command in your terminal:

```bash
sudo apt update && sudo apt install -y aria2 unzip
```

Once the prerequisites are installed, make the script executable and run it:
```bash
chmod +x download_data.sh
./download_data.sh
```

### Resuming interrupted downloads
If the download script fails or is interrupted due to network issues, you can safely run it again:
```bash
./download_data.sh
```

* Resuming: aria2 automatically handles partial downloads and will resume from where it left off.

* Zenodo Edge Case: If the script successfully downloaded and renamed files like train.h5 or test.h5 before interrupting, running the script again might re-download them because the clean filenames no longer match the source URL query. If you want to avoid this, you can comment out the completed URLs inside the script before running it again, or simply let aria2 overwrite them. 
