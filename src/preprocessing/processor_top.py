# processor_top.py
import os
import h5py
import hdf5plugin
import pickle
import numpy as np
import torch
import gc
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler
from pathlib import Path
from src.utils.workspace import get_config, set_seed

# ============================================================================
# ============================================================================
# Vectorized Jet Kinematics Engine (Calculations directly on raw NumPy arrays)
# ============================================================================
# ============================================================================

def _compute_physics_features(raw_matrix, config, scaler=None):
    """
    Core math transformation. Combines macro-physics and Pareto sub-structure 
    into a structured [N, m] matrix ready for KAN / Random Forest baselines. 
    N is the number of jets and m is the number of features (2 + 2 * n_particles).
    The last 4 features are the global jet four-momentum components (E, px, py, pz) and the 
    rest are the four-momentum particle-level features.
    The first two columns are global jet properties (scaled invariant mass and multiplicity), 
    followed by pairs of columns for each particle: [dR_i, z_i].

    - dr: Geometric distance of the i-th particle from the jet axis in the (eta, phi) plane.
    - z: Fraction of the jet's transverse momentum carried by the i-th particle.

    Parameters:
    - raw_matrix: NumPy array of shape [N, 800] containing raw jet data.
    - config: Configuration dictionary containing parameters for processing.
    - scaler: Optional StandardScaler object for normalizing multiplicity.

    return
    - processed_matrix: NumPy array of shape [N, 2 + 2 * n_particles] containing processed features.
    - scaler: StandardScaler object used for multiplicity normalization.
    - mass_mask: Boolean array indicating which jets passed the invariant mass cut.
    """
    # -------------------------------------------------------------------------
    # 1. Global Masking and Initial Geometric Reduction
    # -------------------------------------------------------------------------
    E_jet   = np.sum(raw_matrix[:, 0:800:4], axis=1)
    px_jet  = np.sum(raw_matrix[:, 1:800:4], axis=1)
    py_jet  = np.sum(raw_matrix[:, 2:800:4], axis=1)
    pz_jet  = np.sum(raw_matrix[:, 3:800:4], axis=1)

    pt_jet  = np.sqrt(px_jet**2 + py_jet**2)
    eta_jet = -np.log(np.tan(np.arctan2(pt_jet, pz_jet + 1e-12) / 2.0) + 1e-12)
    phi_jet = np.arctan2(py_jet, px_jet + 1e-12)

    mass_sq = np.clip(E_jet**2 - (px_jet**2 + py_jet**2 + pz_jet**2), 0.0, None)
    invariant_mass = np.sqrt(mass_sq)

    del px_jet, py_jet, pz_jet, E_jet, pt_jet, mass_sq
    gc.collect()

    # Global kinematic filter for the Jet (Invariant Mass > 10 GeV)
    mass_mask = (invariant_mass > 95.0) & (invariant_mass < 176.0)
    #mass_mask = invariant_mass > 10.0
    invariant_mass = invariant_mass[mass_mask]
    raw_matrix = raw_matrix[mass_mask]
    eta_jet = eta_jet[mass_mask]
    phi_jet = phi_jet[mass_mask]
    
    # Extract the raw global multiplicity before truncating the matrix
    multiplicity = np.sum(raw_matrix[:, 0:800:4] > 1e-3, axis=1)
    n_events = raw_matrix.shape[0]

    # Controlled reduction to the optimal window of 100 particles to save RAM
    n_initial_particles = 100
    
    # Extract optimized local blocks in memory
    px_block = raw_matrix[:, 1:n_initial_particles*4:4].copy()
    py_block = raw_matrix[:, 2:n_initial_particles*4:4].copy()
    pz_block = raw_matrix[:, 3:n_initial_particles*4:4].copy()

    # -------------------------------------------------------------------------
    # STEP 1 AND 2: GHOST IDENTIFICATION AND GEOMETRIC CLONING
    # -------------------------------------------------------------------------
    pt_block = np.sqrt(px_block**2 + py_block**2)
    is_real = pt_block >= 1e-3

    # Controlled initialization along the jet axis to mitigate artificial teleportation
    eta_i = np.repeat(eta_jet[:, None], n_initial_particles, axis=1)
    phi_i = np.repeat(phi_jet[:, None], n_initial_particles, axis=1)

    theta_real = np.arctan2(pt_block[is_real], pz_block[is_real] + 1e-12)
    eta_i[is_real] = -np.log(np.tan(theta_real / 2.0) + 1e-12)
    phi_i[is_real] = np.arctan2(py_block[is_real], px_block[is_real] + 1e-12)
    
    del pz_block, px_block, py_block, theta_real
    gc.collect()

    # -------------------------------------------------------------------------
    # STEP 3: RELATIVE MINKOWSKI DISTANCE WITHOUT ASYMMETRIC ARTIFACTS
    # -------------------------------------------------------------------------
    d_eta_i = eta_i - eta_jet[:, None]
    d_phi_i = np.arctan2(np.sin(phi_i - phi_jet[:, None]), np.cos(phi_i - phi_jet[:, None]))
    d_R = np.sqrt(d_eta_i**2 + d_phi_i**2)

    del phi_i, d_eta_i, d_phi_i
    gc.collect()

    # -------------------------------------------------------------------------
    # STEP 4: IN-PLACE PHYSICAL VIABILITY FILTER (EXCLUSIVE p_T SPRAY)
    # -------------------------------------------------------------------------
    # Filter: Outside the cone (> 0.80) OR outside the calorimeter acceptance (|eta| >= 3.0)
    unphysical_mask = (~is_real) | (d_R > 0.80) | (np.abs(eta_i) >= 3.0)
    pt_block[unphysical_mask] = 0.0

    del eta_i, is_real, unphysical_mask
    gc.collect()

    # -------------------------------------------------------------------------
    # STEP 5: HIGH-SPEED SYNCHRONIZED SORTING
    # -------------------------------------------------------------------------
    sorted_indices = np.argsort(-pt_block, axis=1)
    row_indices = np.arange(n_events)[:, None]
    
    pt_block = pt_block[row_indices, sorted_indices]
    d_R = d_R[row_indices, sorted_indices]

    del sorted_indices, row_indices
    gc.collect()

    # -------------------------------------------------------------------------
    # STEP 6: DYNAMIC AMPUTATION BY PARETO (80% CONE ENERGY THRESHOLD)
    # -------------------------------------------------------------------------
    pt_cumsum = np.cumsum(pt_block, axis=1)
    pt_total_jet = pt_cumsum[:, -1]
    
    pt_total_cone_safe = np.where(pt_total_jet <= 0, 1.0, pt_total_jet)
    pt_frac_cumsum = pt_cumsum / pt_total_cone_safe[:, None]
    
    idx_80 = np.argmax(pt_frac_cumsum >= 0.80, axis=1) + 1
    n_particles = int(np.percentile(idx_80, 90))
    n_particles = max(n_particles, 5) 

    pt_block = pt_block[:, :n_particles]
    d_R = d_R[:, :n_particles]

    # CCompute the effective energy fraction z_effective relative to the purified cone
    sum_pt_final = np.sum(pt_block, axis=1)
    sum_pt_final_safe = np.where(sum_pt_final <= 0, 1.0, sum_pt_final)
    z_effective = pt_block / sum_pt_final_safe[:, None]

    # Forzar d_R a 0.0 en los canales vacíos remanentes de padding legítimo
    d_R[pt_block <= 0.0] = 0.0

    del pt_block, pt_cumsum, pt_frac_cumsum, sum_pt_final, sum_pt_final_safe, idx_80
    gc.collect()

    # -------------------------------------------------------------------------
    # PHYSICAL VALIDATION LOGGING
    # -------------------------------------------------------------------------
    '''
    if (~is_ghost).any():
        print(f"eta_i range (processed): {eta_i[:, :n_particles][~is_ghost].min():.2f} \
        to {eta_i[:, :n_particles][~is_ghost].max():.2f}")
    
    valid_dR = d_R[d_R > 0]
    if valid_dR.size > 0:
        print(f"dR range (real particles): {valid_dR.min():.2f} to {valid_dR.max():.2f}")

    print("\n*--- DYNAMIC CONSTITUENT AUDIT (PARETO 80%) ---*")
    print(f"--> NUMBER OF CONSTITUENTS FIXED FOR THIS BATCH: {n_particles}")
    print(f"Average individual index to capture 80% of p_T: {idx_80.mean():.2f}")
    print(f"Minimum number of real particles in final window: {real_particles_per_jet.min()}")
    print(f"Maximum number of real particles in final window: {real_particles_per_jet.max()}")
    print(f"Average number of real particles in final window: {real_particles_per_jet.mean():.2f}")
    print("*-----------------------------------------------------------*\n")
    
    sum_z_test = np.sum(z_effective, axis=1)
    active_jets = sum_z_test > 0
    if active_jets.any():
        is_normalized = np.allclose(sum_z_test[active_jets], 1.0, atol=1e-3)
        print(f"Relative Normalization Test (sum(z_i) == 1.0): {is_normalized}")
    '''

    # -------------------------------------------------------------------------
    # ASYMPTOTIC COMPRESSION AND GLOBAL VARIABLE NORMALIZATION
    # -------------------------------------------------------------------------
    # Invariant Mass: Log + RobustScaler + Asymptotic Tanh (Protects Outliers)
    m_log = np.log(invariant_mass + 1.0)
    del invariant_mass
    
    robust_scaler = RobustScaler()
    m_robust = robust_scaler.fit_transform(m_log.reshape(-1, 1)).flatten()
    m_scaled = np.tanh(m_robust)
    
    del m_log, m_robust
    gc.collect()

    # Multiplicity: StandardScaler + Strict 3 Sigma Clip
    # Preserves the real morphology of physical peaks without deforming tails
    if scaler is None:
        scaler = StandardScaler()
        M_robust = scaler.fit_transform(multiplicity.reshape(-1, 1)).flatten()
    else:
        M_robust = scaler.transform(multiplicity.reshape(-1, 1)).flatten()
        
    del multiplicity
    
    sigma_max = 3.0
    M_clipped = np.clip(M_robust, -sigma_max, sigma_max)
    M_scaled = M_clipped / sigma_max

    del M_robust, M_clipped
    gc.collect()

    # -------------------------------------------------------------------------
    # INTERLEAVED PACKING COLUMNS [N, 2 + 2 * n_particles]
    # -------------------------------------------------------------------------
    processed_matrix = np.zeros((n_events, 2 + 2 * n_particles), dtype=np.float32)
    processed_matrix[:, 0] = m_scaled
    processed_matrix[:, 1] = M_scaled
    
    processed_matrix[:, 2::2] = d_R
    processed_matrix[:, 3::2] = z_effective
    
    del d_R, z_effective, m_scaled, M_scaled
    gc.collect()
        
    return processed_matrix, scaler, mass_mask

# ============================================================================
# ============================================================================
# Load and preprocess data
# ============================================================================
# ============================================================================

def load_and_preprocess_data(data_dir, processed_dir, task, seed=42, force_process=False):
    """ 
    Processes separate train.h5, val.h5, and test.h5 files sequentially.
    """
    set_seed(seed)
    config = get_config(task, seed)
    
    DATA_DIR = Path(data_dir)
    PROCESSED_DIR = Path(processed_dir)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    cache_file = PROCESSED_DIR / "preprocessed_data.pt"
    scaler_file = Path(config["scaler_path"])

    # --- STEP 1: CACHE SYSTEM DETECTOR ---
    if cache_file.exists() and not force_process:
        print(f"\n[CACHE DETECTED] Loading preprocessed matrices from: '{cache_file}'")
        try:
            cached_data = torch.load(cache_file)
            if scaler_file.exists():
                with open(scaler_file, "rb") as f:
                    scaler = pickle.load(f)
            print(">> Multi-scale arrays successfully loaded from cache environment.")
            return (
                cached_data['X_train_tensor'], cached_data['y_train_tensor'],
                cached_data['X_val_tensor'], cached_data['y_val_tensor'],
                cached_data['X_test_tensor'], cached_data['y_test_tensor'],
                cached_data['X_train_sample'], scaler
            )
        except Exception as e:
            print(f"CRITICAL cache error: {e}. Falling back to execution loops.")

    # --- STEP 2: SEQUENTIAL PROCESS (Train, Val, Test) ---
    raw_files = {
        "train": DATA_DIR / "train.h5",
        "val": DATA_DIR / "val.h5",
        "test": DATA_DIR / "test.h5"
    }
    
    processed_tensors = {}
    scaler = None

    for split, file_path in raw_files.items():
        print("\n*---------------------------------------------------*")
        print(f"Loading and processing raw split: [{split.upper()}] from HDF5")
        
        if not file_path.exists():
            raise FileNotFoundError(f"Missing mandatory TopTagging partition file: {file_path.name}")
            
        with h5py.File(file_path, "r") as f:
            # Reconstruct structured event tables
            f = f["table"]["table"] # Navigate to the nested group containing the data
            raw_matrix = f["values_block_0"][:50000]
            # Index 1 holds the categorical value (1: Top Signal, 0: QCD Background)
            raw_labels = f["values_block_1"][:50000, 1]

        print(f"Data chunk successfully mounted in RAM. Extracted shape: {raw_matrix.shape}")
        
        # DIAGNÓSTICO ANTES DEL PROCESAMIENTO
        print(f"--> Distribución CRUDA en disco de etiquetas para [{split.upper()}]:")
        clases, conteos = np.unique(raw_labels, return_counts=True)
        for c, n in zip(clases, conteos):
            print(f"    Clase {c}: {n} eventos")

        print(f"DEBUG ANTES: raw_matrix shape = {raw_matrix.shape}, raw_labels shape = {raw_labels.shape}")

        X_norm, split_scaler, mask = _compute_physics_features(raw_matrix, config, scaler=scaler)

        print(f"DEBUG DESPUÉS: X_norm shape = {X_norm.shape}, mask True count = {np.sum(mask)}")
        print(f"DEBUG ETIQUETAS FILTRADAS: Ceros: {np.sum(raw_labels[mask] == 0)}, Unos: {np.sum(raw_labels[mask] == 1)}")

        # Transform vector components
        #X_norm, split_scaler, mask = _compute_physics_features(raw_matrix, config, scaler=scaler)
        
        if split == "train":
            scaler = split_scaler
            with open(scaler_file, "wb") as f:
                pickle.dump(split_scaler, f)
            print(f"Global scaler object saved to: '{scaler_file}'")

        # Convert straight to standalone float Torch tensors
        processed_tensors[f"X_{split}"] = torch.from_numpy(X_norm).float()
        #raw_labels = raw_labels[mask]
        y_tensor = torch.from_numpy(raw_labels[mask]).float()
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1) # Match required [N, 1] output dimension
        processed_tensors[f"y_{split}"] = y_tensor

        del raw_matrix, raw_labels, X_norm, y_tensor
        gc.collect()

    print("\n--- Final balanced datasets built (Vectorized Slices Framework) ---")
    print(f"X_train shape: {processed_tensors['X_train'].shape} | y_train shape: {processed_tensors['y_train'].shape}")
    print(f"X_val shape:   {processed_tensors['X_val'].shape} | y_val shape:   {processed_tensors['y_val'].shape}")
    print(f"X_test shape:  {processed_tensors['X_test'].shape} | y_test shape:  {processed_tensors['y_test'].shape}")

    # --- STEP 3: SYMBOLIC INTERPOLATION SAMPLE (10k Sub-sample) ---
    print(f"\nIsolating clean sub-sample for high-speed symbolic KAN regressions...")
    sample_size = int(0.05*len(processed_tensors["X_train"]))
    X_train_all = processed_tensors["X_train"]
    
    if len(X_train_all) > sample_size:
        # Uniform sampling permutation over the GPU/CPU data graph boundary
        random_indices = torch.randperm(len(X_train_all))[:sample_size]
        X_train_sample = X_train_all[random_indices]
    else:
        X_train_sample = X_train_all
    print(f"Warm-up tensor isolated. Size: {len(X_train_sample)} physics target nodes.")

    # logging con clases únicas y conteo de ellas
    print(f"Clases únicas en Train: {torch.unique(processed_tensors['y_train'])} con conteos: {torch.bincount(processed_tensors['y_train'].long().squeeze())}")
    print(f"Clases únicas en Val: {torch.unique(processed_tensors['y_val'])} con conteos: {torch.bincount(processed_tensors['y_val'].long().squeeze())}")

    # --- STEP 4: MEMORY MAP CHECKPOINT SERIALIZATION ---
    processed_data = {
        'X_train_tensor': processed_tensors["X_train"],
        'y_train_tensor': processed_tensors["y_train"],
        'X_val_tensor':   processed_tensors["X_val"],
        'y_val_tensor':   processed_tensors["y_val"],
        'X_test_tensor':  processed_tensors["X_test"],
        'y_test_tensor':  processed_tensors["y_test"],
        'X_train_sample': X_train_sample
    }
    
    torch.save(processed_data, cache_file)
    print(f"\n[CACHE WRITTEN] Saving preprocessed database block into: '{cache_file}'")
    print(f"Inference metrics scaler object written into workspace folder structures.")

    return (
        processed_data['X_train_tensor'], processed_data['y_train_tensor'],
        processed_data['X_val_tensor'], processed_data['y_val_tensor'],
        processed_data['X_test_tensor'], processed_data['y_test_tensor'],
        processed_data['X_train_sample'], scaler
    )
