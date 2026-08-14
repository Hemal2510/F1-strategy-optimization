from __future__ import annotations

import sys
import torch
import numpy as np
from pathlib import Path

# Allow importing modules from F1-strategy-showcase root
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.append("C:\\Users\\hemal\\AppData\\Roaming\\Python\\Python312\\site-packages")

from agents.dqn.dqn_agent import DQNAgent, DQNConfig
from agents.qrl.qrl_agent import QRLAgent, QRLConfig
from agents.dqn.action_mask import get_action_mask

DQN_CHECKPOINT = ROOT_DIR / "checkpoints/dqn/checkpoints_v2/best.pt"
QRL_CHECKPOINT = ROOT_DIR / "checkpoints/qrl/checkpoints_qrl_v6/latest.pt"

class LiveModelRunner:
    """
    Loads both trained agents (DQN & QRL) and runs live inference on F1StrategyEnv states.
    """
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.dqn_agent = self._load_dqn()
        self.qrl_agent = self._load_qrl()

    def _load_dqn(self) -> DQNAgent:
        if not DQN_CHECKPOINT.exists():
            raise FileNotFoundError(f"DQN checkpoint not found at: {DQN_CHECKPOINT}")
        
        checkpoint = torch.load(DQN_CHECKPOINT, map_location=self.device, weights_only=False)
        saved_config = dict(checkpoint.get("config", {}))
        saved_config["device"] = str(self.device)
        
        # Build config
        cfg = DQNConfig(obs_dim=15, action_dim=6)
        for k, v in saved_config.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        
        agent = DQNAgent(cfg)
        agent.online_net.load_state_dict(checkpoint["online_net"], strict=True)
        agent.online_net.eval()
        return agent

    def _load_qrl(self) -> QRLAgent:
        if not QRL_CHECKPOINT.exists():
            raise FileNotFoundError(f"QRL checkpoint not found at: {QRL_CHECKPOINT}")
        
        checkpoint = torch.load(QRL_CHECKPOINT, map_location=self.device, weights_only=False)
        saved_config = dict(checkpoint.get("config", {}))
        saved_config["device"] = str(self.device)
        
        # Build config
        cfg = QRLConfig(obs_dim=15, action_dim=6, n_qubits=8)
        for k, v in saved_config.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
                
        agent = QRLAgent(cfg)
        agent.online_net.load_state_dict(checkpoint["online_net"], strict=True)
        agent.online_net.eval()
        return agent

    @torch.no_grad()
    def select_actions(self, observation: np.ndarray, action_mask: np.ndarray) -> dict:
        """
        Runs inference on the same state for both DQN and QRL.
        """
        # DQN Inference
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        dqn_q = self.dqn_agent.online_net(obs_tensor).squeeze(0)
        dqn_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        dqn_legal_q = dqn_q.masked_fill(~dqn_mask, -1e9)
        dqn_action = int(torch.argmax(dqn_legal_q).item())
        dqn_q_vals = dqn_q.detach().cpu().numpy()

        # QRL Inference
        qrl_q = self.qrl_agent.online_net(obs_tensor).squeeze(0)
        qrl_mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        qrl_legal_q = qrl_q.masked_fill(~qrl_mask, -1e9)
        qrl_action = int(torch.argmax(qrl_legal_q).item())
        qrl_q_vals = qrl_q.detach().cpu().numpy()

        # Handle NaNs in Q-values (baselines have NaN Q-values)
        dqn_q_list = [float(x) if not np.isnan(x) else None for x in dqn_q_vals]
        qrl_q_list = [float(x) if not np.isnan(x) else None for x in qrl_q_vals]

        return {
            "dqn": {
                "action": dqn_action,
                "q_values": dqn_q_list
            },
            "qrl": {
                "action": qrl_action,
                "q_values": qrl_q_list
            }
        }
