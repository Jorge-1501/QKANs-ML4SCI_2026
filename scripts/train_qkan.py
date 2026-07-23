# train_qkan.py
import argparse
import traceback
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.utils import workspace
import src.preprocessing.processor_top as processor
from src.architectures.extractor import SymbolicWarmStartExtractor
from src.architectures.quantum_kan import QuantumKANTrainer

def main(args):
    # 1. Configuración
    CONFIG = workspace.get_config(task=args.task, seed=args.seed)
    workspace.make_dirs(CONFIG)
    print(f"Modo de backend seleccionado (Entrenamiento): {args.train_backend} \n")

    # 2. Carga de Datos Clásicos
    # AHORA CAPTURAMOS X_test y y_test
    X_train, y_train, X_val, y_val, X_test, y_test, X_sample, scaler = processor.load_and_preprocess_data(
        data_dir=os.path.join("data", "raw", args.task),
        processed_dir=CONFIG["processed_data_dir"],
        task=args.task,
        force_process=False,
        seed=args.seed
    )

    # 3. Extracción Automática (Warm-Start Agnóstico)
    extractor = SymbolicWarmStartExtractor(CONFIG)
    
    classic_model_path = os.path.join(CONFIG['final_model_path'], "05_final")
    output_weights_path = os.path.join(CONFIG["polynomial_weights_dir"], "quantum_weights.pt")
    report_path = CONFIG.get("Chebyshev_coefficients_path", os.path.join(CONFIG["results_dir"], "chebyshev_report.txt"))
    
    # Si se fuerza o no existe, extrae los pesos del modelo clásico
    if args.force or not os.path.exists(output_weights_path):
        extractor.extract_and_save(classic_model_path, output_weights_path, report_path)
    else:
        print(f"[Orquestador] Pesos cuánticos dinámicos encontrados en {output_weights_path}. Saltando extracción.")

    # 4. Inicialización y Entrenamiento Cuántico
    q_trainer = QuantumKANTrainer(CONFIG, train_backend=args.train_backend)
    
    # Graficar el circuito dinámico generado antes de entrenar
    q_trainer.model.plot_circuit(CONFIG.get("circuit_plot", os.path.join(CONFIG["plots_dir"], "quantum-circuit.png")))
    
    # Bucle de optimización cuántica
    history = q_trainer.fit(X_train, y_train, X_val, y_val, resume=True, force=args.force)

    # 5. Evaluación Final
    # AHORA LLAMAMOS AL MÉTODO INTERNO DE LA CLASE
    q_trainer.evaluate(X_test, y_test, eval_backend=args.eval_backend)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena QKAN con arquitectura inferida del modelo clásico.")
    parser.add_argument('--seed', type=int, default=42, help='Semilla global')
    parser.add_argument('--train_backend', type=str, choices=['noisy', 'ideal', 'shots'], default='ideal')
    parser.add_argument('--eval_backend', type=str, choices=['noisy', 'ideal', 'shots'], default='noisy')
    parser.add_argument('--force', action='store_true', help='Fuerza extracción y reentrenamiento')
    parser.add_argument('--task', type=str, choices=['top', 'quark-gluon'], default='top')
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"Ocurrió un error fatal: {e}")
        traceback.print_exc()