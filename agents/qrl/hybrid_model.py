import torch
import torch.nn as nn
import pennylane as qml

from agents.qrl.quantum_circuit import quantum_circuit, weight_shapes, N_QUBITS

"""
Basically a dueling DQN network, but with a quantum circuit in the body instead of classical layers.
Classical : obs -> [Linear+LayerNorm+ReLU] x2 -> dueling heads -> Q
Quantum   : obs -> classical encoder -> VQC (8 qubits) -> dueling heads -> Q

Everything after the 'features' vector (value stream, advantage
stream, dueling aggregation) is same as classical DQN network on purpose, so
that any performance difference we observe in benchmarking can be
attributed to the body (classical layers vs. quantum circuit), not to
incidental architecture differences elsewhere.
"""
class hybridQuantumNetwork(nn.Module):
    def __init__(self, obs_dim:int = 15, action_dim:int = 3, n_qubits:int = N_QUBITS):
        super().__init__()
        
        self.n_qubits = n_qubits

#classical encoder - maps 15-d observations to n_qubits features for the quantum circuit 
#tanh layer converts values to [-1, 1] range so that whne scaled by pi, angles stay in [-p1, p1] range instead of wrapping around
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, n_qubits),
            nn.Tanh()
        )

        # small-Gaussian init to avoid barren plateaus: keeps the circuit
        # close to identity at the start, where gradients are larger
        def small_init(tensor):
            return nn.init.normal_(tensor, mean=0.0, std=0.01)

        weight_init = {"weights": small_init}

#quantum layer - maps n_qubits features to n_qubits features using a variational quantum circuit
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes(), init_method=weight_init)

#dueling heads
        self.value_stream = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.ReLU(),
            nn.Linear(n_qubits, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(n_qubits, n_qubits),
            nn.ReLU(),
            nn.Linear(n_qubits, action_dim)
        )

#forward pass - maps observations to Q-values
    def forward(self, obs:torch.Tensor):

        encoded = self.encoder(obs)*torch.pi
        q_features = self.q_layer(encoded)

        values = self.value_stream(q_features)
        advantages = self.advantage_stream(q_features)

        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)

        return q_values
    

