# network_mamba.py
from typing import Optional
import torch
import torch.nn as nn

# mamba-ssm
# (버전에 따라 import 경로가 다를 수 있어 try로 처리)
try:
    from mamba_ssm.modules.mamba_simple import Mamba
except Exception:
    # 일부 버전은 아래처럼 노출되기도 합니다.
    from mamba_ssm import Mamba


# ---------------------------
# 초기화 유틸
# ---------------------------
def orthogonal_init(m: nn.Module, gain: float = 1.0):
    """Linear 계층만 직교 초기화."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------
# 공통: (B,D) / (B,T,D) 입력 정규화
# ---------------------------
def _ensure_btd(x: torch.Tensor) -> torch.Tensor:
    # x: (B,D) or (B,T,D)
    if x.dim() == 2:
        return x.unsqueeze(1)  # (B,1,D)
    if x.dim() == 3:
        return x
    # 그 외: (B,*) 형태로 펴서 (B,1,D)로
    return x.view(x.size(0), -1).unsqueeze(1)


# ---------------------------
# Mamba 2-layer backbone (BN between optional)
# ---------------------------
class Mamba2LayerBackbone(nn.Module):
    """
    proj_in -> Mamba -> (BN between) -> Mamba -> last-token pooling -> (B,H)
    - BN은 (B*T,H)로 펼쳐서 적용하며, training 중 B*T<2이면 스킵
    """
    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bn_momentum: float = 0.1,
        use_bn_between: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.use_bn_between = use_bn_between

        # 입력 차원을 Mamba d_model로 맞추기 위한 projection
        self.proj_in = nn.Linear(in_dim, d_model)

        self.mamba1 = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.bn_between = nn.BatchNorm1d(d_model, momentum=bn_momentum, eps=1e-5)

        self.mamba2 = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # init
        self.apply(lambda m: orthogonal_init(m, gain=1.0))
        # proj_in은 너무 크게 시작하지 않게 gain을 낮춰도 됨(선택)
        nn.init.orthogonal_(self.proj_in.weight, gain=1.0)
        nn.init.zeros_(self.proj_in.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,D) or (B,T,D)
        return: (B,H)
        """
        x = _ensure_btd(x)                # (B,T,D)
        x = self.proj_in(x)               # (B,T,H)

        z1 = self.mamba1(x)               # (B,T,H)

        if self.use_bn_between:
            B, T, H = z1.shape
            flat = z1.contiguous().view(B * T, H)  # (B*T,H)

            # BN은 train 모드에서 배치가 1이면 통계 계산 문제 → 스킵
            if self.training and flat.size(0) < 2:
                z1_bn = z1
            else:
                z1_bn = self.bn_between(flat).view(B, T, H)
        else:
            z1_bn = z1

        z2 = self.mamba2(z1_bn)           # (B,T,H)

        # LSTM의 h_last 대체: 마지막 토큰 pooling
        out = z2[:, -1, :]                # (B,H)
        return out


# ---------------------------
# Actor (Mamba backbone)
# ---------------------------
class ActorNet(nn.Module):
    """
    Mamba1 -> BN -> Mamba2 -> LeakyReLU -> Linear -> Sigmoid
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 1,
        hidden: int = 128,
        bn_momentum: float = 0.1,
        # Mamba 하이퍼파라미터(필요 시 조정)
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden

        self.backbone = Mamba2LayerBackbone(
            in_dim=obs_dim,
            d_model=hidden,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bn_momentum=bn_momentum,
            use_bn_between=True,
        )

        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.head = nn.Linear(hidden, act_dim)
        self.sig = nn.Sigmoid()

        # init head
        self.apply(lambda m: orthogonal_init(m, gain=1.0))
        nn.init.orthogonal_(self.head.weight, gain=0.5)
        nn.init.zeros_(self.head.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        z = self.backbone(s)              # (B,H)
        z = self.act(z)
        a01 = self.sig(self.head(z))      # (B,act_dim) in [0,1]
        return a01


# ---------------------------
# Critic (Mamba backbone)
# ---------------------------
class CriticNet(nn.Module):
    """
    concat(s,a) -> Mamba1 -> BN -> Mamba2 -> LeakyReLU -> Linear -> Q
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 1,
        hidden: int = 128,
        bn_momentum: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden

        in_dim = obs_dim + act_dim
        self.backbone = Mamba2LayerBackbone(
            in_dim=in_dim,
            d_model=hidden,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bn_momentum=bn_momentum,
            use_bn_between=True,
        )

        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.q_out = nn.Linear(hidden, 1)

        # init head
        self.apply(lambda m: orthogonal_init(m, gain=1.0))
        nn.init.orthogonal_(self.q_out.weight, gain=0.5)
        nn.init.zeros_(self.q_out.bias)

    def forward(self, s: torch.Tensor, a01: torch.Tensor) -> torch.Tensor:
        # s: (B,D) or (B,T,D), a01: (B,A) or (B,T,A)
        if s.dim() == 2 and a01.dim() == 2:
            x = torch.cat([s, a01], dim=-1).unsqueeze(1)   # (B,1,D+A)
        elif s.dim() == 3 and a01.dim() == 3:
            x = torch.cat([s, a01], dim=-1)                # (B,T,D+A)
        else:
            s2 = s.view(s.size(0), -1)
            a2 = a01.view(a01.size(0), -1)
            x = torch.cat([s2, a2], dim=-1).unsqueeze(1)   # (B,1,D+A)

        z = self.backbone(x)          # (B,H)
        z = self.act(z)
        q = self.q_out(z)             # (B,1)
        return q
