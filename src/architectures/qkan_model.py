# src/architectures/qkan_model.py
import pennylane as qml
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

class QKANModel(nn.Module):
    def __init__(self, metadata_path, backend_mode="ideal"):
        super(QKANModel, self).__init__()
        
        # Load classical Warm-Start metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No quantum weights found in {metadata_path}. Run the extractor first.")
            
        metadata = torch.load(metadata_path, weights_only=False)
        self.active_inputs = metadata["active_inputs"]
        self.n_qubits = metadata["n_qubits"]
        self.degree = metadata["degree"]
        
        # Initialize trainable parameters correctly as nn.Parameters
        edge_data = metadata["edge_weights"].detach().clone().float() if isinstance(metadata["edge_weights"], torch.Tensor) else torch.tensor(metadata["edge_weights"], dtype=torch.float32)
        out_data = metadata["out_weights"].detach().clone().float() if isinstance(metadata["out_weights"], torch.Tensor) else torch.tensor(metadata["out_weights"], dtype=torch.float32)

        self.edge_weights = nn.Parameter(edge_data)
        self.out_weights = nn.Parameter(out_data)
        
        # Configure quantum device
        self.backend_mode = backend_mode
        self.dev = self._initialize_device()
        
        # Create the QNode coupled to the instantiated device
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _initialize_device(self):
        if self.backend_mode == "noisy":
            print("[QKAN] Noisy simulator: FakeManilaV2...")
            from qiskit_ibm_runtime.fake_provider import FakeManilaV2
            from qiskit_aer.noise import NoiseModel

            fake_backend = FakeManilaV2()
            noise_model = NoiseModel.from_backend(fake_backend)
            
            return qml.device("qiskit.aer",
                                wires=self.n_qubits,
                                noise_model=noise_model,
                                backend='aer_simulator_density_matrix',
                                set_shots=1024)
        elif self.backend_mode == "shots":
            print("[QKAN] Shots-based simulator...")
            return qml.device("default.qubit",
                                wires=self.n_qubits,
                                set_shots=1024)
        else:
            print("[QKAN] Ideal simulator (lightning.qubit)...")
            return qml.device("lightning.qubit",
                                wires=self.n_qubits)

    def _qkan_edge(self, x_val, weights, wire):
        """
        Input layer for a single qubit, representing the edge function in QKAN.
        """
        theta = torch.acos(torch.clamp(x_val, -0.9999, 0.9999))
        for i in range(self.degree):
            qml.RY(weights[i], wires=wire)
            qml.RZ(theta, wires=wire)
        qml.RY(weights[self.degree], wires=wire)

    def _circuit(self, inputs):
        """
        Dynamic construction of the quantum circuit.
        """
        # Feature Map
        for q_idx in range(self.n_qubits):
            self._qkan_edge(inputs[q_idx], self.edge_weights[q_idx], wire=q_idx)
            
        # 2. Dynamic Additive/Multiplicative Entanglement
        # For >1 qubit, we use qubit 0 as the accumulator
        if self.n_qubits > 1:
            for q_idx in range(1, self.n_qubits):
                # Use RZZ to allow continuous interference (Cross Multiplication/Additive)
                qml.IsingZZ(self.out_weights[0], wires=[q_idx, 0])
                # Classical CNOT mesh (optional, depending on the desired physics)
                qml.CNOT(wires=[q_idx, 0])
                
        # Output Layer
        qml.RY(self.out_weights[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    def forward(self, x):
        """
        Filter the inputs to use only the active variables.
        """
        # Filter x to keep only the columns corresponding to active_inputs
        x_filtered = x[:, self.active_inputs]
        batch_size = x_filtered.shape[0]
        outputs = torch.zeros(batch_size, device=x.device)
        
        for i in range(batch_size):
            outputs[i] = self.qnode(x_filtered[i])
            
        # The final scalar is now dynamic (out_weights[0] acts as a general scale in the circuit)
        return outputs 

    def plot_circuit(self, save_path):
        """
        Dynamic circuit plotter.
        """
        print(f"[QKAN] Generating circuit diagram at {save_path}...")
        dummy_inputs = torch.rand(self.n_qubits)
        
        fig, ax = qml.draw_mpl(
            self.qnode, decimals=2, style="pennylane"
        )(dummy_inputs)
        
        plt.title(f"QKAN Agnostic ({self.n_qubits} Qubits)", fontsize=24)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()