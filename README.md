# Quantum Sinusoidal-Kolmogorov-Arnold-Networks for High Energy Physics

This repository contains the code developed for **QKAN**, a project for Google Summer of Code 2026 at [ML4SCI](https://ml4sci.org/).

The project builds a classical **Kolmogorov-Arnold Network (KAN)** for High Energy Physics (HEP) jet classification, prunes it down to a small, interpretable topology, and extracts each surviving edge into a compact basis function representation, either **Chebyshev polynomials** (default) or a fixed-frequency **sine basis** ("SineKAN", [Reinhardt et al. 2024](https://arxiv.org/abs/2407.04149)). That extracted graph warm-starts a **Variational Quantum Circuit (QKAN)** built with [PennyLane](https://pennylane.ai/), which is then fine-tuned and evaluated on ideal, shot-noise, and noisy quantum backends. A classical Random Forest is trained alongside as a fast, strong reference point.

---

## Background / Method

**Task.** Binary jet classification on HEP dataset:
- **Top tagging** (main pipeline) — [Zenodo record 2603256](https://zenodo.org/records/2603256), with an invariant-mass cut (145–205 GeV) isolating top-quark jets from background.

Each jet is represented by 22 features: total jet mass `m`, particle multiplicity `n`, and per-particle `DR_i`/`pT_i` for the 10 leading particles.

**Data regime.** The balanced dataset is split once into **5 disjoint, class balanced statistical replicate subsets**. `--seed` selects the `seed % n_subsets`-th replicate and drives the *entire* pipeline (base training → pruning → retraining → extraction → quantum training/evaluation) on that one subset, so running several seeds gives independent end-to-end replicates instead of one full-dataset point estimate. `scripts/run_preprocessing.py` builds this canonical cache; every other script only selects from it (and raises if the cache doesn't exist yet).

**Pipeline.**
1. **Base KAN training**. `src/architectures/classic_kan.py`, the base [pykan](https://github.com/KindXiaoming/pykan)-derived KAN trainer.
2. **HEP-specific model + pruning**. `src/architectures/hep_kan.py`. Pruning combines pykan's native attribution-threshold pruning (`node_th`/`edge_th`) with an additive hard fan-in cap (`prune_fanin`) that keeps only the top-2 highest-attribution input edges per hidden neuron.
3. **Retraining** the pruned topology.
4. **Warm-start extraction**. `src/architectures/extractor.py` (`SymbolicWarmStartExtractor`, Chebyshev basis, fit against the retrained numeric spline branch at a fixed degree) or `src/architectures/sine_basis.py` + `extractor_sine.py` (fixed-frequency SineKAN basis), each surviving edge's isolated response is fit into a small set of coefficients, producing a structured "quantum graph" of sum/multiplication nodes.
5. **Quantum circuit**. `src/architectures/qkan_model.py` (`QKANModel`): classical inputs are angle encoded via repeated RY/RZ "data re-uploading" per edge, multiplication nodes use `IsingZZ` + `CNOT`, and the prediction is a single-qubit `PauliZ` expectation. Two backends: `ideal` (`lightning.qubit`), and `noisy` (Qiskit `FakeManilaV2` + noise model).
6. **Quantum fine-tuning + evaluation**. `src/architectures/quantum_kan.py` (`QuantumKANTrainer`): Adam + `ReduceLROnPlateau`, early stopping, `BCEWithLogitsLoss`. Evaluates the model both as a warm-started baseline (pre-training) and after fine tuning, on both `ideal` and `noisy` backends.
7. **Classical baseline**. `src/architectures/random_forest.py` (`RandomForestTrainer`), trained on all 22 features via `scripts/train_rf.py`, for a fast, strong classical comparison point.

---

## Current status / findings

- **Warm-start comparison** (seeds 10–14, untrained circuit, ideal backend): Chebyshev mean AUC **0.698 ± 0.004** vs. SineKAN **0.644 ± 0.019** vs. random init **0.482 ± 0.029**. Both real warm starts clearly beat chance; the gap is attributed to Chebyshev's much better per-edge fit quality (R² ≈ 0.997–0.999 vs. sine's 0.49–0.61). See `reports/benchmarks/sine_vs_chebyshev_vs_random_baseline.md`.
- **Baseline-AUC regression, found and fixed**: an adaptive minimum-degree Chebyshev search (accepting the first fit degree whose R² cleared a threshold) silently collapsed baseline AUC from ~0.80 to ~0.26–0.36 once training data shrank to 1/15 of the initial full pool. Root-caused and fixed by reverting to an unconditional fixed-degree-4 fit, restoring AUC to ≈0.80–0.81 uniformly across seeds. See `reports/AUC_test/` and `reports/implementations/reported_changes.md`.
- **Quantum-baseline collapse investigation**: of 5 hypothesized causes for a separate confusion-matrix-collapse pattern (fixed 0.5 decision threshold, warm-start bias, aggressive pruning, weak gradients, a script/regime mismatch), 4 were tested and rejected. The confirmed driver is that classical pruning collapses most runs down to only ~2 surviving input features — a structural bottleneck that can't be loosened without pushing the qubit count past what the current simulator can evaluate in reasonable time. See `reports/benchmarks/baseline_quantum_collapse_investigation.md` and `baseline_quantum_collapse_solutions_tested.md`.
- **Simulation performance (exploratory)**: a JAX + `default.qubit` reimplementation of the VQC forward/train step is ~129x faster than the current Torch + `lightning.qubit` path at today's circuit sizes (4–7 qubits); GPU shows no benefit yet at this scale. Not adopted in production. See `reports/benchmarks/vqc_jax_vs_torch_training_step.md`.

---

## Structure

The repository includes:
* `notebooks/`: exploratory data analysis (`EDA_top.ipynb`), a step-by-step walkthrough of the training process (`Training_process.ipynb`), and results analysis (`Results.ipynb`).
* `src/`: code for the project.
  * `architectures/`. model implementations: `classic_kan.py`, `hep_kan.py`, `extractor.py`, `sine_basis.py`/`extractor_sine.py`, `qkan_model.py`, `quantum_kan.py`, `random_forest.py`.
  * `preprocessing/`: `balance.py` (class balancing + `n_subsets`-way subset split, 5 by default), `processor_top.py`, `processor_qg.py`.
  * `utils/`: `hyperparams.py`, `workspace.py` (path/config resolution), `metrics.py`, `reporting.py` (metrics aggregation), `evaluate_qkan.py`.
* `scripts/`: CLI entry points for data downloading, preprocessing, training, and evaluation (see below).
* `tests/`: pytest suite.
* `outputs/`, `data/`: generated/cached artifacts (see "Output files" below).
* `README.md`: this file.

**Main files:**
* `src/architectures/hep_kan.py`: HEP-KAN model architecture (Kolmogorov-Arnold repr., modified for HEP data).
* `src/architectures/classic_kan.py`: Base KAN model implementation extended by HEP-KAN.
* `src/architectures/extractor.py`: Chebyshev warm-start extraction from a pruned classical KAN into a quantum-ready graph.
* `src/architectures/qkan_model.py`: The PennyLane variational quantum circuit (`QKANModel`).
* `src/architectures/quantum_kan.py`: Quantum training/evaluation loop (`QuantumKANTrainer`).
* `src/architectures/random_forest.py`: Classical Random Forest baseline.
* `scripts/train_kan.py`: Script for training the HEP-KAN model.
* `scripts/train_qkan.py`: Script for extracting the warm start and training/evaluating the quantum circuit.

---

## Environment

Python **3.11 or 3.12** is required (`requires-python = ">=3.11, <3.13"`). Two equivalent ways to set up the environment are provided; use one of them.

### Option A: uv (recommended, reproducible from `uv.lock`)

```bash
pip install uv
uv sync                       # create/sync the virtual environment (.venv)
uv sync --extra cuda          # ...with the CUDA (cu121) torch build
uv sync --extra cpu           # ...or with the CPU-only torch build (mutually exclusive with cuda)
```

### Option B: classic pip + venv (from `requirements.txt`)

`requirements.txt` is a frozen snapshot (`pip freeze`) of the project virtual environment, with every version pinned. It targets the **CUDA 12.1** build of torch (`torch==2.4.1+cu121`) and already carries the PyTorch `--extra-index-url` line at the top.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a **CPU-only** machine, change the `--extra-index-url` at the top of `requirements.txt` to `https://download.pytorch.org/whl/cpu` and the torch pin to `torch==2.4.1+cpu`. `pykan` is installed from GitHub (pinned to a commit in `requirements.txt`), so `git` must be available.

To regenerate the file after changing the environment, run this with the venv activated:

```bash
pip freeze > requirements.txt      # then re-add the header lines (--extra-index-url ...)
```

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

---

## Running the pipeline

### Data regimes and directory layout

Preprocessing/training can run in two regimes; each one has its own directories, so runs never overwrite each other:

* **Default (partitioned):** invariant-mass cut (145–205 GeV, top tagging) and `n_subsets` disjoint, class-balanced partitions per split (`hyperparams.py`). `--seed` selects subset `seed % n_subsets`.
* **`--full-dataset` (optional, off by default):** no mass cut and `n_subsets=1`, i.e. the entire dataset. Train/val/test stay separate splits.

```
data/processed/<task>/<cut>/<full|n{N}>/                      # canonical cache: preprocessed_subsets.pt, global_scaler.pkl
outputs/<task>/<cut>/<full|n{N}_subset{k}>/seed_<seed>/       # one run: models/ plots/ results/ logs/ hyperparameters.json
outputs/<task>/aggregate/metrics_table.parquet                # all runs, tagged by `variant`
```

`<cut>` is `mass_cut` or `no_mass_cut` (top tagging only; quark-gluon has no mass cut). A run with `n_subsets=1` is labeled `full`; otherwise the directory name carries the partition count and the selected subset (e.g. `n5_subset3`).

### Top-tagging pipeline

```bash
python scripts/run_preprocessing.py [--full-dataset] --force   # build the canonical cache for a regime
python scripts/train_kan.py --seed 42 [--full-dataset]
python scripts/train_qkan.py --seed 42 [--full-dataset]        # must match the regime of the classical run
python scripts/train_rf.py --task top --seed 42 [--full-dataset]
python scripts/collect_metrics.py --task top
```

### Basis comparison and sweeps

```bash
python scripts/eval_sine_baseline.py --seeds 10 11 12 13 14     # Chebyshev vs. SineKAN vs. random warm-start comparison
scripts/run_all_classic.sh                                      # single seed, classical -> quantum -> metrics collection
scripts/run_seeds.sh                                             # multi-seed sweep (self-backgrounding via nohup)
scripts/run_seeds_random_init.sh                                 # random-VQC-init ablation
```

All paths are produced by `src/utils/workspace.py` (`get_config(task, seed, full_dataset=...)`); don't hardcode them.

---

## Output files

The output files generated by the training and evaluation scripts are typically stored in a designated directory (`outputs/`). These files may include:
* Model checkpoints: Saved states of the trained model, usually in `.pt` or `.ckpt` format.
* Training logs: Logs containing information about the training process, such as loss and accuracy metrics.
* Evaluation results: Files containing the performance metrics of the trained model on the test dataset.
* Plots and visualizations: Graphical representations of the training progress and evaluation results.

The resulting output files are generated in the `outputs/` directory, organized according to their type (checkpoints, logs, evaluation results, and visualizations). With the current structure of the repository, you can easily locate and manage the outputs corresponding to different runs and experiments and you don't need to create manually any subdirectories. Also, if you want to see the current results from the author, you can access [this drive file](https://drive.google.com/drive/folders/1elF53g99h0OQvAfiQd8k2qCoYObbV1o6?usp=sharing).

---

## Testing

```bash
uv run pytest tests/ -v            # with uv
python -m pytest tests/ -v         # with an activated pip/venv environment
```

The suite covers pruning fan-in behavior, Chebyshev/sine warm-start extraction, quantum-baseline evaluation behavior, efficiency-metric computation, metrics reporting/workspace path utilities, and dataset balancing/subset splitting.

---

## License

MIT License, © 2026 Jorge Toral. See [LICENSE](LICENSE).

---

## Acknowledgements / References

* Kolmogorov-Arnold Networks — [pykan](https://github.com/KindXiaoming/pykan) (installed from GitHub; the project's modifications live in `src/architectures/hep_kan.py` as a subclass, not in a fork).
* SineKAN — Reinhardt et al., 2024, [arXiv:2407.04149](https://arxiv.org/abs/2407.04149).
* [ML4SCI](https://ml4sci.org/) / Google Summer of Code 2026.
