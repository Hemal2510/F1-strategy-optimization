import torch 
import torch.nn as nn

class DQNNetwork(nn.Module):
    def __init__(self, obs_dim : int = 15, action_dim: int = 6, hid_dim: int = 256):
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(obs_dim,hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim,hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hid_dim,hid_dim//2),
            nn.ReLU(),
            nn.Linear(hid_dim//2,1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hid_dim,hid_dim//2),
            nn.ReLU(),
            nn.Linear(hid_dim//2, action_dim),

        )
    
    def forward(self, obs: torch.tensor) -> torch.tensor:
        features = self.feature_layer(obs)
        values = self.value_stream(features)
        advantage = self.advantage_stream(features)

        q_values = values + advantage - advantage.mean(dim = 1 , keepdim = True)

        return q_values