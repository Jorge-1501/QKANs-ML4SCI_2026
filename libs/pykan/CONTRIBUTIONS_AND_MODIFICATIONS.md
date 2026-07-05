# Technical Report: Modifications to KAN Core Logic

## 1. Original Attribution
- **Base Engine:** pykan (MultKAN implementation)
- **Original Author:** Ziming Liu et al.
- **Repository:** [https://github.com/KindXiaoming/pykan](https://github.com/KindXiaoming/pykan)

## 2. Research Context
These modifications were implemented by **Jorge Luis Toral Gamez** as part of an undergraduate Physics thesis at **Benemérita Universidad Autónoma de Puebla (BUAP)**. The core objective is to optimize Kolmogorov-Arnold Networks for discriminating particle jets in High-Energy Physics (HEP) datasets.

## 3. Detailed Summary of Changes

### A. Execution Efficiency & Logging (`MultKAN.py`)
- **Logging Bypass:** Modified `log_history` to return immediately. This prevents the automatic generation of intensive log files and checkpoints during training, optimizing disk I/O and execution speed for large-scale HEP simulations.
- **Auto-save Commenting:** Commented out internal model versioning prints to ensure a cleaner terminal output during high-iteration training cycles.

### B. Visualization Enhancements (`MultKAN.py`)
- **Automated Plot Saving:** Extended the `plot()` method to include a `save_path` parameter. This allows for direct export of network architectures to high-resolution files (DPI=400) with proper bounding boxes, facilitating integration into LaTeX documentation.
- **Visual Scaling:** Adjusted the `w_large` parameter (neuron width scaling) from a static 2.0 to a dynamic `4.0 * scale` to improve clarity in complex network visualizations.

### C. Symbolic Logic & Data Casting (`MultKAN.py`)
- **Tensor Handling:** Implemented explicit `.detach().cpu().numpy()` and `.item()` casting for affine parameters and node scales. This ensures compatibility between PyTorch tensors and Sympy symbolic representations, preventing type errors during model simplification.

### D. Fitting Hyperparameters (`utils.py`)
- **Grid Density Tuning:** Adjusted the default `grid_number` in `fit_params` from 101 to 40. This change optimizes the curve-fitting process for the specific density of the particle physics data used in this research.

---
*The full technical diff is preserved in `logic_modifications.diff` for peer-review and reproducibility purposes.*
