# qkan_model.py
import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import os

# ============================================================================
# 1. Device configuration
# ============================================================================

n_qubits = 4

backend_mode = os.environ.get("QKAN_BACKEND", "ideal").lower()

if backend_mode == "noisy":
    print("Using qiskit-aer FakeManilaV2 backend for noise simulation...")
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2
    backend_ibm = FakeManilaV2()
    dev = qml.device("qiskit.aer", wires=n_qubits, noise=backend_ibm, backend = 'aer_simulator_density_matrix', shots=1024)
elif backend_mode == "ideal":
    print("Using lightning.qubit for ideal statevector simulation...")
    dev = qml.device("lightning.qubit", wires=n_qubits)
elif backend_mode == "shots":
    print("Using default.qubit with shots for sampling...")
    dev = qml.device("default.qubit", wires=n_qubits, shots=1024)



# ============================================================================
# 2. QNode definition
# ============================================================================

def qkan_edge(x, weights, wire, degree=4):
    """
    Implements the KAN function using Data Re-uploading.
    This emulates the Chebyshev expansion on quantum hardware.
    args:
        x (float): input feature
        weights (array): array of learnable parameters (length = degree + 1)
        wire (int): qubit to use for this edge
        degree (int): degree of the Chebyshev expansion (default 4)

    The function applies a sequence of rotations and data re-uploading to encode the input feature.
    The final rotation is applied after the re-uploading steps.
    """
    theta = torch.acos(torch.clamp(x, -0.9999, 0.9999))
    
    for i in range(degree):
        qml.RY(weights[i], wires=wire)
        qml.RZ(theta, wires=wire)
    
    qml.RY(weights[degree], wires=wire)

@qml.qnode(dev, interface="torch")
def qkan_circuit(inputs, w_pt, w_eta, w_mass, w_n, w_out):
    """
    Complete QKAN circuit [4, 1, 1].
    """
    # --- STEP 1: CHEB and MUL ---
    qkan_edge(inputs[0], w_pt, wire=0, degree=4)
    qkan_edge(inputs[1], w_eta, wire=1, degree=4)
    qkan_edge(inputs[2], w_mass, wire=2, degree=4)
    qkan_edge(inputs[3], w_n, wire=3, degree=4)
    
    # --- STEP 2: LCU and SUM ---
    qml.CNOT(wires=[1, 0])
    qml.CNOT(wires=[2, 0])
    qml.CNOT(wires=[3, 0])
    
    # --- STEP 3: Outer Layer ---
    qml.RY(w_out[0], wires=0)
    qml.RZ(w_out[1], wires=0)
    
    return qml.expval(qml.PauliZ(0))

# ============================================================================
# 3. PyTorch Module Wrapper
# ============================================================================

class QKANModel(nn.Module):
    def __init__(self, init_weights_path=None):
        super(QKANModel, self).__init__()
        
        if init_weights_path:
            self.w_pt = nn.Parameter(torch.tensor(np.load(f"{init_weights_path}/w_pt.npy"), dtype=torch.float32))
            self.w_eta = nn.Parameter(torch.tensor(np.load(f"{init_weights_path}/w_eta.npy"), dtype=torch.float32))
            self.w_mass = nn.Parameter(torch.tensor(np.load(f"{init_weights_path}/w_mass.npy"), dtype=torch.float32))
            self.w_n = nn.Parameter(torch.tensor(np.load(f"{init_weights_path}/w_n.npy"), dtype=torch.float32))
            self.w_out = nn.Parameter(torch.tensor(np.load(f"{init_weights_path}/w_out.npy"), dtype=torch.float32))
        else:
            self.w_pt = nn.Parameter(torch.randn(5))
            self.w_eta = nn.Parameter(torch.randn(5))
            self.w_mass = nn.Parameter(torch.randn(5))
            self.w_n = nn.Parameter(torch.randn(5))
            self.w_out = nn.Parameter(torch.randn(2))

    def forward(self, x):
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size, device=x.device)
        
        for i in range(batch_size):
            outputs[i] = qkan_circuit(
                x[i], 
                self.w_pt, self.w_eta, self.w_mass, self.w_n, 
                self.w_out
            )
        
        return outputs * -5.0

def plot_qkan_circuit(save_path):
    import matplotlib.pyplot as plt
    print("Generating the quantum circuit plot...")
    
    # To draw a circuit in PennyLane, we need to pass random data
    # so that the compiler knows the size of the tensors and how many gates to draw.
    dummy_inputs = torch.tensor([0.5, -0.2, 0.8, -0.1])
    dummy_w = torch.rand(5)  # Dummy weights for the edges (degree 4 -> 5 weights)
    dummy_w_out = torch.rand(2) # Dummy weights for the output (degree 1 -> 2 weights)
    
    # We use qml.draw_mpl to generate the figure with Matplotlib
    fig, ax = qml.draw_mpl(
        qkan_circuit, 
        decimals=2, # Show up to 2 decimals if the parameters are fixed
        style="pennylane" # Clean and modern visual style
    )(dummy_inputs, dummy_w, dummy_w, dummy_w, dummy_w, dummy_w_out)
    
    # Title for the plot
    plt.title("QKAN Architecture [4, 1, 1] with Data Re-uploading", fontsize=38)
    
    # Save the image in high quality
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved as '{save_path}'!") 