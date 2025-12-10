# network.py
from typing import Tuple
import torch
import torch.nn as nn

class BetaPolicyHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ab = self.fc(z)
        alpha_raw, beta_raw = ab[..., 0], ab[..., 1]
        alpha = self.softplus(alpha_raw) + 1.0
        beta  = self.softplus(beta_raw)  + 1.0
        return alpha, beta

class RecurrentActorCritic(nn.Module):
    """
    공유 인코더 → LSTM → (정책 Beta-Head, 가치 V-Head)
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, batch_first=True)
        self.policy = BetaPolicyHead(hidden_dim)
        self.value  = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def init_state(self, batch_size: int = 1):
        h = torch.zeros(1, batch_size, self.lstm.hidden_size)
        c = torch.zeros(1, batch_size, self.lstm.hidden_size)
        return (h, c)

    def forward(self, obs: torch.Tensor, state):
        # obs: (B, 1, obs_dim)
        feat = self.encoder(obs.squeeze(1))       # (B,128)
        lstm_in = feat.unsqueeze(1)               # (B,1,128)
        z, new_state = self.lstm(lstm_in, state)  # z: (B,1,H)
        z = z.squeeze(1)                          # (B,H)
        alpha, beta = self.policy(z)
        value = self.value(z).squeeze(-1)         # (B,)
        return alpha, beta, value, new_state
