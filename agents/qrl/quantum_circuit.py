import pennylane as qml
import torch

N_QUBITS = 8
N_LAYERS = 3

#define the quantum device
dev = qml.device("default.qubit", wires = N_QUBITS)

#ring entanglement function
def ring_entanglement(wires):
    n = len(wires)
    for i in range(n):
        qml.CNOT(wires=[i, (i+1)%n])

@qml.qnode(dev, interface = "torch", diff_method = "backprop")
def quantum_circuit(inputs, weights):
#feature encoding - angle embedding
    wires = list(range(N_QUBITS))
    qml.AngleEmbedding(inputs, wires=wires, rotation="Y")

#variational circuit - rotational layers and ring entanglement
    for layer in range(N_LAYERS):
        for q in wires:
            qml.Rot(*weights[layer, q], wires=q)
        ring_entanglement(wires)

#measurement - expectation value of Pauli-Z operator
    return [qml.expval(qml.PauliZ(w)) for w in wires]

#return the shape of the weights tensor
def weight_shapes():
    return {"weights": (N_LAYERS, N_QUBITS, 3)}