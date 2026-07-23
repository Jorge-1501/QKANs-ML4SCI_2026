# src/architectures/qkan_model.py
import pennylane as qml
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

class QKANModel(nn.Module):
    def __init__(self, metadata_path, backend_mode="ideal"):
        super(QKANModel, self).__init__()
        
        # Cargar metadatos del Warm-Start clásico
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No se encontraron pesos cuánticos en {metadata_path}. Ejecuta el extractor primero.")
            
        metadata = torch.load(metadata_path, weights_only=False)
        self.active_inputs = metadata["active_inputs"]
        self.n_qubits = metadata["n_qubits"]
        self.degree = metadata["degree"]
        
        # Inicializar parámetros entrenables con la forma extraída
        self.edge_weights = nn.Parameter(torch.tensor(metadata["edge_weights"], dtype=torch.float32))
        self.out_weights = nn.Parameter(torch.tensor(metadata["out_weights"], dtype=torch.float32))
        
        # Configurar dispositivo cuántico
        self.backend_mode = backend_mode
        self.dev = self._initialize_device()
        
        # Crear el QNode acoplado al dispositivo instanciado
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _initialize_device(self):
        if self.backend_mode == "noisy":
            print("[QKAN] Configurando simulador ruidoso (FakeManilaV2)...")
            from qiskit_ibm_runtime.fake_provider import FakeManilaV2
            return qml.device("qiskit.aer", wires=self.n_qubits, noise=FakeManilaV2(), backend='aer_simulator_density_matrix', shots=1024)
        elif self.backend_mode == "shots":
            print("[QKAN] Configurando simulador por muestreo (shots)...")
            return qml.device("default.qubit", wires=self.n_qubits, shots=1024)
        else:
            print("[QKAN] Configurando simulador ideal (lightning.qubit)...")
            return qml.device("lightning.qubit", wires=self.n_qubits)

    def _qkan_edge(self, x_val, weights, wire):
        """Mapeo a la base de Chebyshev usando Data Re-uploading."""
        theta = torch.acos(torch.clamp(x_val, -0.9999, 0.9999))
        for i in range(self.degree):
            qml.RY(weights[i], wires=wire)
            qml.RZ(theta, wires=wire)
        qml.RY(weights[self.degree], wires=wire)

    def _circuit(self, inputs):
        """Construcción dinámica del circuito cuántico."""
        # 1. Feature Map / Capa Oculta
        for q_idx in range(self.n_qubits):
            self._qkan_edge(inputs[q_idx], self.edge_weights[q_idx], wire=q_idx)
            
        # 2. Entrelazamiento Aditivo/Multiplicativo dinámico
        # Para >1 qubit, usamos el qubit 0 como acumulador
        if self.n_qubits > 1:
            for q_idx in range(1, self.n_qubits):
                # Usamos RZZ para permitir interferencia continua (Multiplicación/Suma cruzada)
                qml.IsingZZ(self.out_weights[0], wires=[q_idx, 0])
                # Malla CNOT clásica (opcional, dependiendo de la física deseada)
                qml.CNOT(wires=[q_idx, 0])
                
        # 3. Capa de Salida
        qml.RY(self.out_weights[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    def forward(self, x):
        """Filtra las entradas para usar solo las variables activas."""
        # Filtrar x para mantener solo las columnas correspondientes a active_inputs
        x_filtered = x[:, self.active_inputs]
        batch_size = x_filtered.shape[0]
        outputs = torch.zeros(batch_size, device=x.device)
        
        for i in range(batch_size):
            outputs[i] = self.qnode(x_filtered[i])
            
        # El escalar final ahora es dinámico (out_weights[0] actúa como escala general en el circuito)
        return outputs 

    def plot_circuit(self, save_path):
        """Graficador dinámico del circuito."""
        print(f"[QKAN] Generando diagrama del circuito en {save_path}...")
        dummy_inputs = torch.rand(self.n_qubits)
        
        fig, ax = qml.draw_mpl(
            self.qnode, decimals=2, style="pennylane"
        )(dummy_inputs)
        
        plt.title(f"QKAN Agnóstica ({self.n_qubits} Qubits)", fontsize=24)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()