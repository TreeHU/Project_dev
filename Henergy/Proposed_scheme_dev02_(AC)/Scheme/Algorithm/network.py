# network.py
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def orthogonal_init(m: nn.Module, gain: float = 1.0):
    """간단한 직교 초기화(선택)."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class SafeBatchNorm1d(nn.Module):
    """
    배치 크기가 1일 때는 BatchNorm을 건너뛰어 오류를 방지.
    (train/eval 모두 입력을 그대로 통과시킴)
    """
    def __init__(self, num_features: int, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C) or (B, C, L)
        if x.size(0) <= 1:
            return x
        return self.bn(x)


class ActorNet(nn.Module):
    """
    Deterministic Actor: s -> a01 \in (0,1)
    - Linear + ReLU + SafeBatchNorm -> LSTM(seq_len=1) -> SafeBatchNorm -> Linear -> Sigmoid
    - 입력 s: (B, obs_dim)
    - 출력: (B, act_dim)  ; a01 공간(0~1)
    """
    def __init__(self, obs_dim: int, act_dim: int = 1, hidden: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden  = hidden

        self.enc_fc  = nn.Linear(obs_dim, hidden)
        self.enc_act = nn.ReLU(inplace=True)
        self.enc_bn  = SafeBatchNorm1d(hidden)          # ★ 안전 배치정규화

        self.lstm    = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.lstm_bn = SafeBatchNorm1d(hidden)          # ★ 안전 배치정규화

        self.head = nn.Linear(hidden, act_dim)
        self.sig  = nn.Sigmoid()

        # (선택) 초기화: Linear에만 적용
        self.apply(lambda m: orthogonal_init(m, gain=1.0))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: (B, obs_dim)
        return: a01 in (0,1)  shape (B, act_dim)
        """
        if s.dim() != 2:
            s = s.view(s.size(0), -1)                  # (B, obs_dim)

        feat = self.enc_act(self.enc_fc(s))            # (B, H)
        feat = self.enc_bn(feat)                       # (B, H)
        z, _ = self.lstm(feat.unsqueeze(1))            # (B, 1, H)
        z = z.squeeze(1)                               # (B, H)
        z = self.lstm_bn(z)                            # (B, H)
        a01 = self.sig(self.head(z))                   # (B, act_dim) in (0,1)
        return a01


class CriticNet(nn.Module):
    """
    Critic: (s, a01) -> Q(s,a)
    - concat(s, a01) -> Linear + ReLU + SafeBatchNorm -> LSTM(seq_len=1)
      -> SafeBatchNorm + LeakyReLU -> Linear -> Q
    - 입력 s: (B, obs_dim)
      입력 a01: (B, act_dim)  ; 반드시 0~1 공간의 액션
    - 출력 Q: (B, 1)
    """
    def __init__(self, obs_dim: int, act_dim: int = 1, hidden: int = 128):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden  = hidden

        in_dim = obs_dim + act_dim
        self.enc_fc  = nn.Linear(in_dim, hidden)
        self.enc_act = nn.ReLU(inplace=True)
        self.enc_bn  = SafeBatchNorm1d(hidden)         # ★ 안전 배치정규화

        self.lstm    = nn.LSTM(input_size=hidden, hidden_size=hidden, batch_first=True)
        self.lstm_bn = SafeBatchNorm1d(hidden)         # ★ 안전 배치정규화

        self.post_act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.q_out    = nn.Linear(hidden, 1)

        # (선택) 초기화
        self.apply(lambda m: orthogonal_init(m, gain=1.0))

    def forward(self, s: torch.Tensor, a01: torch.Tensor) -> torch.Tensor:
        """
        s:   (B, obs_dim)
        a01: (B, act_dim)  in (0,1)
        return: Q(s,a)  shape (B, 1)
        """
        if s.dim() != 2:
            s = s.view(s.size(0), -1)
        if a01.dim() != 2:
            a01 = a01.view(a01.size(0), -1)

        x = torch.cat([s, a01], dim=-1)                # (B, obs_dim + act_dim)
        feat = self.enc_act(self.enc_fc(x))            # (B, H)
        feat = self.enc_bn(feat)                       # (B, H)
        z, _ = self.lstm(feat.unsqueeze(1))            # (B, 1, H)
        z = z.squeeze(1)                               # (B, H)
        z = self.lstm_bn(z)                            # (B, H)
        z = self.post_act(z)                           # (B, H)
        q = self.q_out(z)                              # (B, 1)
        return q