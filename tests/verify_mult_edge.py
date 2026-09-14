import pennylane as qml
import torch

dev = qml.device('lightning.qubit', wires=1)

@qml.qnode(dev, interface='torch')
def circuit_batched(inputs, weights_1, weights_2):
    # inputs has shape (batch, 2)
    t1 = inputs[:, 0]
    qml.RY(weights_1[0], wires=0)
    qml.RZ(t1, wires=0)
    qml.RY(weights_1[1], wires=0)
    
    t2 = inputs[:, 1]
    qml.RY(weights_2[0], wires=0)
    qml.RZ(t2, wires=0)
    qml.RY(weights_2[1], wires=0)
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface='torch')
def circuit_single(inputs, weights_1, weights_2):
    t1 = inputs[0]
    qml.RY(weights_1[0], wires=0)
    qml.RZ(t1, wires=0)
    qml.RY(weights_1[1], wires=0)
    
    t2 = inputs[1]
    qml.RY(weights_2[0], wires=0)
    qml.RZ(t2, wires=0)
    qml.RY(weights_2[1], wires=0)
    return qml.expval(qml.PauliZ(0))

inputs = torch.tensor([[0.1, 0.2], [0.5, 0.6], [0.9, 0.1]])
weights_1 = torch.tensor([0.2, 0.3])
weights_2 = torch.tensor([0.4, 0.5])

y_batched = circuit_batched(inputs, weights_1, weights_2)
print('Batched:', y_batched)

y_looped = torch.stack([circuit_single(inp, weights_1, weights_2) for inp in inputs])
print('Looped:', y_looped)
print('Equal?', torch.allclose(y_batched, y_looped))
