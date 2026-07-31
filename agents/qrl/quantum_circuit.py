import pennylane as qml
import torch

N_QUBITS = 8
N_LAYERS = 3

dev = qml.device("default.qubit", wires=N_QUBITS)


def ring_entanglement(wires):
    n = len(wires)
    for i in range(n):
        qml.CNOT(wires=[i, (i + 1) % n])


@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    wires = list(range(N_QUBITS))
    qml.AngleEmbedding(inputs, wires=wires, rotation="Y")

    for layer in range(N_LAYERS):
        for q in wires:
            qml.Rot(*weights[layer, q], wires=q)
        ring_entanglement(wires)

    return [qml.expval(qml.PauliZ(w)) for w in wires]


def weight_shapes():
    return {"weights": (N_LAYERS, N_QUBITS, 3)}