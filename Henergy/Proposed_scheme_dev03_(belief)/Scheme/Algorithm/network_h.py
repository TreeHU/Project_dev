# network.py
from typing import Tuple
import torch
import torch.nn as nn

# ---------------------------
# 초기화 유틸
# ---------------------------
def orthogonal_init(m: nn.Module, gain: float = 1.0):
    """Linear 계층만 직교 초기화 (안정한 시작)."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def init_lstm(lstm: nn.LSTM):
    """LSTM 안정 초기화: 입력 xavier, 순환 orthogonal, bias 0."""
    for name, p in lstm.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(p)
        elif "weight_hh" in name:
            nn.init.orthogonal_(p)
        elif "bias" in name:
            nn.init.zeros_(p)

# ---------------------------
# Actor
# ---------------------------
class ActorNet(nn.Module):
    """
    LSTM1 -> BatchNorm1d(두 LSTM 사이) -> LSTM2 -> LeakyReLU -> Linear -> Sigmoid
    - B*T < 2 (예: B=1, T=1)에서는 BN을 '스킵'하여 오류 방지
    """
    def __init__(self, obs_dim: int, act_dim: int = 1, hidden: int = 128, bn_momentum: float = 0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden  = hidden

        self.lstm1 = nn.LSTM(input_size=obs_dim, hidden_size=hidden, batch_first=True)
        self.bn_between = nn.BatchNorm1d(hidden, momentum=bn_momentum, eps=1e-5)
        self.lstm2 = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)

        self.act  = nn.LeakyReLU(0.1, inplace=True)
        self.head = nn.Linear(hidden, act_dim)
        self.sig  = nn.Sigmoid()

        init_lstm(self.lstm1); init_lstm(self.lstm2)
        self.apply(lambda m: orthogonal_init(m, gain=1.0))
        nn.init.orthogonal_(self.head.weight, gain=0.5); nn.init.zeros_(self.head.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        # s: (B, D) or (B, T, D)
        if s.dim() == 2:
            s = s.unsqueeze(1)          # (B,1,D)
        elif s.dim() != 3:
            s = s.view(s.size(0), -1).unsqueeze(1)

        z1, _ = self.lstm1(s)           # (B,T,H)

        # --- BN between LSTM1 and LSTM2 ---
        B, T, H = z1.size()
        x = z1.contiguous().view(B*T, H)      # (B*T, H)
        if self.training and x.size(0) < 2:   # B*T==1일 때 BN 오류 회피
            z1_bn = z1                        # BN 스킵 (identity)
        else:
            z1_bn = self.bn_between(x).view(B, T, H)

        z2, (h2, _) = self.lstm2(z1_bn)       # h2: (1,B,H)
        z  = h2[-1]                            # (B,H)

        z   = self.act(z)
        a01 = self.sig(self.head(z))          # (B, act_dim)
        return a01

# ---------------------------
# Critic
# ---------------------------
class CriticNet(nn.Module):
    """
    concat(s,a) -> LSTM1 -> BatchNorm1d(두 LSTM 사이) -> LSTM2 -> LeakyReLU -> Linear -> Q
    - B*T < 2에서는 BN 스킵
    """
    def __init__(self, obs_dim: int, act_dim: int = 1, hidden: int = 128, bn_momentum: float = 0.1):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden  = hidden

        in_dim = obs_dim + act_dim
        self.lstm1 = nn.LSTM(input_size=in_dim, hidden_size=hidden, batch_first=True)
        self.bn_between = nn.BatchNorm1d(hidden, momentum=bn_momentum, eps=1e-5)
        self.lstm2 = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)

        self.act   = nn.LeakyReLU(0.1, inplace=True)
        self.q_out = nn.Linear(hidden, 1)

        init_lstm(self.lstm1); init_lstm(self.lstm2)
        self.apply(lambda m: orthogonal_init(m, gain=1.0))
        nn.init.orthogonal_(self.q_out.weight, gain=0.5); nn.init.zeros_(self.q_out.bias)

    def forward(self, s: torch.Tensor, a01: torch.Tensor) -> torch.Tensor:
        # s: (B,D) or (B,T,D), a01: (B,A) or (B,T,A)
        if s.dim() == 2 and a01.dim() == 2:
            x = torch.cat([s, a01], dim=-1).unsqueeze(1)   # (B,1,D+A)
        elif s.dim() == 3 and a01.dim() == 3:
            x = torch.cat([s, a01], dim=-1)                # (B,T,D+A)
        else:
            s = s.view(s.size(0), -1); a01 = a01.view(a01.size(0), -1)
            x = torch.cat([s, a01], dim=-1).unsqueeze(1)

        z1, _ = self.lstm1(x)             # (B,T,H)

        # --- BN between LSTM1 and LSTM2 ---
        B, T, H = z1.size()
        x_bn = z1.contiguous().view(B*T, H)
        if self.training and x_bn.size(0) < 2:
            z1_bn = z1                      # BN 스킵
        else:
            z1_bn = self.bn_between(x_bn).view(B, T, H)

        z2, (h2, _) = self.lstm2(z1_bn)
        z  = h2[-1]                          # (B,H)

        z  = self.act(z)
        q  = self.q_out(z)                   # (B,1)
        return q