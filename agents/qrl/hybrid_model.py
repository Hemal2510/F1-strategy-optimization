import torch
import torch.nn as nn
import pennylane as qml

from agents.qrl.quantum_circuit import quantum_circuit, weight_shapes, N_QUBITS


class hybridQuantumNetwork(nn.Module):
    # Hybrid quantum-classical dueling DQN network
    def __init__(self, obs_dim: int = 15, action_dim: int = 3, n_qubits: int = N_QUBITS):
        super().__init__()
        
        self.n_qubits = n_qubits

        # Classical encoder to map inputs to qubit angles
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, n_qubits),
            nn.Tanh()
        )

        def small_init(tensor):
            return nn.init.normal_(tensor, mean=0.0, std=0.01)

        weight_init = {"weights": small_init}

        # Quantum layer (VQC)
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes(), init_method=weight_init)

        # Dueling heads
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

    def forward(self, obs: torch.Tensor):
        encoded = self.encoder(obs) * torch.pi
        q_features = self.q_layer(encoded)

        values = self.value_stream(q_features)
        advantages = self.advantage_stream(q_features)

        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)

        return q_values
