import pennylane as qml
import torch

dev = qml.device('lightning.qubit', wires=2)

@qml.qnode(dev, interface='torch')
def circuit_batched(inputs, weights, zz_weight):
    t1 = inputs[:, 0]
    qml.RY(weights[0], wires=0)
    qml.RZ(t1, wires=0)
    
    t2 = inputs[:, 1]
    qml.RY(weights[1], wires=1)
    qml.RZ(t2, wires=1)
    
    qml.IsingZZ(zz_weight, wires=[0, 1])
    qml.CNOT(wires=[0, 1])
    
    return qml.expval(qml.PauliZ(1))

@qml.qnode(dev, interface='torch')
def circuit_single(inputs, weights, zz_weight):
    t1 = inputs[0]
    qml.RY(weights[0], wires=0)
    qml.RZ(t1, wires=0)
    
    t2 = inputs[1]
    qml.RY(weights[1], wires=1)
    qml.RZ(t2, wires=1)
    
    qml.IsingZZ(zz_weight, wires=[0, 1])
    qml.CNOT(wires=[0, 1])
    
    return qml.expval(qml.PauliZ(1))

inputs = torch.tensor([[0.1, 0.2], [0.5, 0.6], [0.9, 0.1]])
weights = torch.tensor([0.2, 0.3])
zz_weight = torch.tensor(0.5)

y_batched = circuit_batched(inputs, weights, zz_weight)
print('Batched:', y_batched)

y_looped = torch.stack([circuit_single(inp, weights, zz_weight) for inp in inputs])
print('Looped:', y_looped)
print('Equal?', torch.allclose(y_batched, y_looped))
