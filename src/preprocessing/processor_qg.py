# processor_qg.py
import os
import gc
import pickle
import numpy as np
import torch
from pathlib import Path
from src.utils.workspace import get_config, set_seed

# Dicionario global de cargas de partículas (PDG ID -> Charge)
CHARGES_DICT = {
    211: 1.0, -211: -1.0, 111: 0.0, 22: 0.0, 
    321: 1.0, -321: -1.0, 2212: 1.0, -2212: -1.0, 
    11: -1.0, -11: 1.0, 13: -1.0, -13: 1.0, 
    2112: 0.0, -2112: 0.0, 130: 0.0, 310: 0.0
}

def _compute_qg_physics_features(X, y, config, fit_scalers=True, scaler_dict=None):
    """
    Pipeline de procesamiento optimizado en RAM para clasificación de Quarks y Gluones.
    Modifica y procesa los arreglos por medio de rebanadas (slices) vectorizadas.
    """
    n_events, max_particles, n_features = X.shape # (100000, 139, 4)
    
    # -------------------------------------------------------------------------
    # PASO 1: Filtrado Cinemático y Geométrico Coherente (Jet Clean-up)
    # -------------------------------------------------------------------------
    pt_raw = X[:, :, 0]
    eta_raw = X[:, :, 1]
    phi_raw = X[:, :, 2]
    pdg_raw = X[:, :, 3]

    sum_pt = np.sum(pt_raw, axis=1)
    sum_pt_safe = np.where(sum_pt == 0, 1.0, sum_pt)

    # Cálculo del eje del Jet (Manejo elemental de periodicidad en phi por estabilidad)
    # Se usa la media ponderada por pT directa
    eta_jet = np.sum(pt_raw * eta_raw, axis=1) / sum_pt_safe
    
    # Manejo de discontinuidad en phi (-pi, pi) utilizando componentes vectoriales
    sin_phi_avg = np.sum(pt_raw * np.sin(phi_raw), axis=1) / sum_pt_safe
    cos_phi_avg = np.sum(pt_raw * np.cos(phi_raw), axis=1) / sum_pt_safe
    phi_jet = np.arctan2(sin_phi_avg, cos_phi_avg)

    del sum_pt, sum_pt_safe, sin_phi_avg, cos_phi_avg
    gc.collect()

    # Cálculo vectorizado de Delta R_i usando slices sobre la memoria existente
    d_eta = eta_raw - eta_jet[:, None]
    d_phi = np.arctan2(np.sin(phi_raw - phi_jet[:, None]), np.cos(phi_raw - phi_jet[:, None]))
    d_R = np.sqrt(d_eta**2 + d_phi**2)

    del d_eta, d_phi
    gc.collect()

    # Máscara de aceptación física
    unphysical_mask = (pt_raw <= 1e-3) | (d_R > 0.4)
    
    # Enmascaramiento absoluto IN-PLACE sobre la matriz original X para salvar memoria
    X[unphysical_mask] = 0.0
    # Sincronizar d_R con la máscara física
    d_R[unphysical_mask] = 0.0

    del unphysical_mask
    gc.collect()

    # -------------------------------------------------------------------------
    # PASO 2: Extracción de Características Globales (Antes del Truncamiento)
    # -------------------------------------------------------------------------
    # Multiplicidad Real
    multiplicity = np.sum(X[:, :, 0] > 0.0, axis=1).astype(np.float32)

    # Carga Ponderada del Jet (Q_jet)
    # Vectorización del mapeo de PDG ID usando un vector plano de NumPy
    max_pdg = int(np.max(np.abs(pdg_raw)))
    charge_lookup = np.zeros(max_pdg + 1, dtype=np.float32)
    # (Corrección de signo)
    for pdg, chg in CHARGES_DICT.items():
        if abs(pdg) <= max_pdg:
            # 1. Almacenamos el valor absoluto de la carga en la tabla de búsqueda
            charge_lookup[abs(pdg)] = abs(chg) 

    # 2. Reconstruimos recuperando el signo original del PDG ID de cada constituyente
    charge_matrix = charge_lookup[np.abs(pdg_raw).astype(np.int32)] * np.sign(pdg_raw)
    
    kappa = 0.5
    pt_weighted = np.power(X[:, :, 0], kappa)
    numerator = np.sum(charge_matrix * pt_weighted, axis=1)
    
    pt_jet = np.sum(X[:, :, 0], axis=1)
    pt_jet_safe = np.where(pt_jet == 0, 1.0, pt_jet)
    denominator = np.power(pt_jet_safe, kappa)
    
    q_jet = numerator / denominator

    del charge_lookup, charge_matrix, pt_weighted, numerator, denominator
    gc.collect()

    # -------------------------------------------------------------------------
    # PASO 3: Ordenamiento Cinemático y Recálculo del Corte de Pareto
    # -------------------------------------------------------------------------
    # Obtener índices para ordenamiento descendente basado en pT
    sorted_indices = np.argsort(-X[:, :, 0], axis=1)
    row_indices = np.arange(n_events)[:, None]

    # Ordenamiento en bloque usando indexación avanzada
    X = X[row_indices, sorted_indices, :]
    d_R = d_R[row_indices, sorted_indices]

    del sorted_indices, row_indices
    gc.collect()

    # Identificación dinámica de Pareto al 80% de la energía del Jet
    pt_cumsum = np.cumsum(X[:, :, 0], axis=1)
    pt_total_safe = np.where(pt_jet[:, None] <= 0, 1.0, pt_jet[:, None])
    pt_frac_cumsum = pt_cumsum / pt_total_safe

    # Encontrar primer índice por evento que cruza el 80%
    idx_80 = np.argmax(pt_frac_cumsum >= 0.80, axis=1) + 1
    # Percentil 90 global para fijar el truncamiento uniforme N_cut
    n_cut = int(np.percentile(idx_80, 90))
    n_cut = max(n_cut, 5) # Garantizar un piso mínimo de partículas

    # Truncamiento fijo del eje de constituyentes
    X = X[:, :n_cut, :]
    d_R = d_R[:, :n_cut]

    del pt_cumsum, pt_total_safe, pt_frac_cumsum, idx_80
    gc.collect()

    # -------------------------------------------------------------------------
    # PASO 4: Construcción y Escalamiento al Rango Dinámico de la KAN
    # -------------------------------------------------------------------------
    # Cálculo de Momento Relativo Local z_i
    z_effective = X[:, :, 0] / pt_jet_safe[:, None]
    z_effective[X[:, :, 0] <= 0.0] = 0.0

    # Inicialización de scalers para el set de Entrenamiento
    if fit_scalers:
        scaler_dict = {
            'z_max': float(np.max(z_effective)),
            'q_max': float(np.max(np.abs(q_jet))),
            'n_min': float(np.min(multiplicity)),
            'n_max': float(np.max(multiplicity))
        }
        # Evitar divisiones por cero en transformaciones
        if scaler_dict['q_max'] == 0: scaler_dict['q_max'] = 1.0
        if scaler_dict['z_max'] == 0: scaler_dict['z_max'] = 1.0
        if scaler_dict['n_max'] == scaler_dict['n_min']: scaler_dict['n_max'] += 1e-5

    # Aplicación de normalizaciones estrictas
    z_scaled = z_effective / scaler_dict['z_max']
    dr_scaled = d_R / 0.4
    q_scaled = q_jet / scaler_dict['q_max']
    
    # MinMax a rango [-1, 1] para la multiplicidad
    n_scaled = 2.0 * ((multiplicity - scaler_dict['n_min']) / (scaler_dict['n_max'] - scaler_dict['n_min'])) - 1.0

    del z_effective, d_R, q_jet, multiplicity, pt_jet, pt_jet_safe
    gc.collect()

    # Empacamiento entrelazado: [Mult, Q_jet, z_1, DR_1, ..., z_N, DR_N]
    processed_matrix = np.zeros((n_events, 2 + 2 * n_cut), dtype=np.float32)
    processed_matrix[:, 0] = n_scaled
    processed_matrix[:, 1] = q_scaled
    processed_matrix[:, 2::2] = z_scaled
    processed_matrix[:, 3::2] = dr_scaled

    del n_scaled, q_scaled, z_scaled, dr_scaled
    gc.collect()

    # Control de sanidad numérico estricto
    assert not np.isnan(processed_matrix).any(), "NaN detectado en la matriz final."
    assert not np.isinf(processed_matrix).any(), "Inf detectado en la matriz final."

    return processed_matrix, scaler_dict

def load_and_preprocess_data(data_dir, processed_dir, task, seed=42, force_process=False):
    """
    Orquestador del pipeline secuencial para la carga de datos estructurados de Quark-Gluon.
    Procesa iterativamente archivos .npz y genera submuestras para regresión simbólica.
    """
    set_seed(seed)
    config = get_config(task=task, seed=seed)
    
    DATA_DIR = Path(data_dir)
    PROCESSED_DIR = Path(processed_dir)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    cache_file = PROCESSED_DIR / "qg_preprocessed_data.pt"
    scaler_file = PROCESSED_DIR / "qg_scalers.pkl"

    # --- PASO 1: CACHE SYSTEM DETECTOR ---
    if cache_file.exists() and not force_process:
        print(f"\n[CACHE DETECTED] Loading preprocessed matrices from: '{cache_file}'")
        try:
            cached_data = torch.load(cache_file)
            if scaler_file.exists():
                with open(scaler_file, "rb") as f:
                    scalers = pickle.load(f)
            print(">> Multi-scale arrays successfully loaded from cache environment.")
            return (
                cached_data['X_train_tensor'], cached_data['y_train_tensor'],
                cached_data['X_val_tensor'], cached_data['y_val_tensor'],
                cached_data['X_test_tensor'], cached_data['y_test_tensor'],
                cached_data['X_train_sample'], scalers
            )
        except Exception as e:
            print(f"CRITICAL cache error: {e}. Falling back to execution loops.")

    # --- PASO 2: CARGA Y PARTICIÓN SÍNCRONA (CON CORRECCIÓN DE MULTIPLICIDAD) ---
    # Lista analítica de archivos .npz del dataset simulado (Pythia 8)
    npz_files = sorted(list(DATA_DIR.glob("QG_jets_fp32_*.npz")))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No se encontraron archivos .npz en la ruta {data_dir}")

    print(f"Detectados {len(npz_files)} archivos de datos. Iniciando procesamiento en cascada...")

    X_list, y_list = [], []
    max_m_global = 0
    
    # Primera pasada rápida para extraer datos e identificar el M máximo global en RAM
    for file_path in npz_files[:3]:  # Control estricto de RAM local
        print(f"Montando en memoria RAM: {file_path.name}")
        with np.load(file_path, 'r') as data:
            X_block = data['X'][:]
            y_block = data['y'][:]
            
            max_m_global = max(max_m_global, X_block.shape[1])
            
            X_list.append(X_block)
            y_list.append(y_block)

    # Homogeneizar el eje 1 (multiplicidad) in-place antes de concatenar
    for i in range(len(X_list)):
        current_m = X_list[i].shape[1]
        if current_m < max_m_global:
            pad_width = max_m_global - current_m
            # Aplicamos padding de ceros estrictamente al final del eje 1, dejando intactos los ejes 0 y 2
            X_list[i] = np.pad(X_list[i], ((0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0.0)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    
    del X_list, y_list
    gc.collect()

    # Partición lógica tradicional indexada 70/15/15
    n_total = X_all.shape[0]
    indices = np.random.permutation(n_total)
    
    train_idx = indices[:int(0.70 * n_total)]
    val_idx = indices[int(0.70 * n_total):int(0.85 * n_total)]
    test_idx = indices[int(0.85 * n_total):]

    processed_tensors = {}

    # --- PASO 3: FEATURE ENGINEERING EN CASCADA ---
    print(f"Ejecutando feature engineering vectorizado para el set de ENTRENAMIENTO...")
    X_train, scalers = _compute_qg_physics_features(X_all[train_idx], y_all[train_idx], config=config, fit_scalers=True)
    processed_tensors["X_train"] = torch.from_numpy(X_train).float()
    processed_tensors["y_train"] = torch.from_numpy(y_all[train_idx]).float().unsqueeze(1)
    
    with open(scaler_file, "wb") as f:
        pickle.dump(scalers, f)
    print(f"Global scaler object saved to: '{scaler_file}'")

    print(f"Ejecutando feature engineering para VALIDACIÓN...")
    X_val, _ = _compute_qg_physics_features(X_all[val_idx], y_all[val_idx], config=config, fit_scalers=False, scaler_dict=scalers)
    processed_tensors["X_val"] = torch.from_numpy(X_val).float()
    processed_tensors["y_val"] = torch.from_numpy(y_all[val_idx]).float().unsqueeze(1)

    print(f"Ejecutando feature engineering para TEST...")
    X_test, _ = _compute_qg_physics_features(X_all[test_idx], y_all[test_idx], config=config, fit_scalers=False, scaler_dict=scalers)
    processed_tensors["X_test"] = torch.from_numpy(X_test).float()
    processed_tensors["y_test"] = torch.from_numpy(y_all[test_idx]).float().unsqueeze(1)

    del X_all, y_all, indices, X_train, X_val, X_test
    gc.collect()

    print("\n--- Final balanced datasets built (Vectorized Slices Framework) ---")
    print(f"X_train shape: {processed_tensors['X_train'].shape} | y_train shape: {processed_tensors['y_train'].shape}")
    print(f"X_val shape:   {processed_tensors['X_val'].shape} | y_val shape:   {processed_tensors['y_val'].shape}")
    print(f"X_test shape:  {processed_tensors['X_test'].shape} | y_test shape:  {processed_tensors['y_test'].shape}")

    # --- PASO 4: INTERPOLATION SAMPLE (5% Sub-sample para Regresión Simbólica) ---
    print(f"\nIsolating clean sub-sample for high-speed symbolic KAN regressions...")
    sample_size = int(0.05 * len(processed_tensors["X_train"]))
    X_train_all = processed_tensors["X_train"]
    
    if len(X_train_all) > sample_size:
        random_indices = torch.randperm(len(X_train_all))[:sample_size]
        X_train_sample = X_train_all[random_indices]
    else:
        X_train_sample = X_train_all
    print(f"Warm-up tensor isolated. Size: {len(X_train_sample)} physics target nodes.")

    # Diagnóstico sintáctico final de balance de tensores nativos
    print(f"Clases únicas en Train: {torch.unique(processed_tensors['y_train'])} con conteos: {torch.bincount(processed_tensors['y_train'].long().squeeze())}")
    print(f"Clases únicas en Val: {torch.unique(processed_tensors['y_val'])} con conteos: {torch.bincount(processed_tensors['y_val'].long().squeeze())}")

    # --- PASO 5: SERIALIZACIÓN COMPLETA DE DATA BLOCKS ---
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

    return (
        processed_data['X_train_tensor'], processed_data['y_train_tensor'],
        processed_data['X_val_tensor'], processed_data['y_val_tensor'],
        processed_data['X_test_tensor'], processed_data['y_test_tensor'],
        processed_data['X_train_sample'], scalers
    )
