# network.py
from typing import Optional
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

# ---------------------------
# Positional Encoding (Learnable)
# ---------------------------
class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional embedding.
    x: (B, T, H) -> x + pos[:, :T, :]
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pos[:, :T, :]

# ---------------------------
# Transformer Backbone (Pre-LN)
# ---------------------------
class TransformerBackbone(nn.Module):
    """
    Input projection -> (optional) positional encoding -> TransformerEncoder -> last token pooling
    - Supports x: (B, D) or (B, T, D)
    """
    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        use_pos_emb: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model

        self.in_proj = nn.Linear(in_dim, d_model)

        self.pos_emb = LearnablePositionalEncoding(d_model, max_len=max_len) if use_pos_emb else None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.apply(lambda m: orthogonal_init(m, gain=1.0))
        # 입력 프로젝션은 너무 크지 않게
        nn.init.orthogonal_(self.in_proj.weight, gain=1.0)
        nn.init.zeros_(self.in_proj.bias)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, D) or (B, T, D)
        key_padding_mask: (B, T) where True indicates PAD tokens (to ignore)
        return: pooled (B, H)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,D)
        elif x.dim() != 3:
            x = x.view(x.size(0), -1).unsqueeze(1)

        h = self.in_proj(x)  # (B,T,H)
        if self.pos_emb is not None:
            h = self.pos_emb(h)

        # TransformerEncoder: uses key_padding_mask (True=ignore)
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)  # (B,T,H)

        # 마지막 토큰 pooling (기존 LSTM의 last hidden과 대응)
        pooled = h[:, -1, :]  # (B,H)
        return pooled

# ---------------------------
# Actor
# ---------------------------
class ActorNet(nn.Module):
    """
    Transformer backbone -> LeakyReLU -> Linear -> Sigmoid
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 1,
        hidden: int = 128,      # d_model
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        use_pos_emb: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden

        self.backbone = TransformerBackbone(
            in_dim=obs_dim,
            d_model=hidden,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            max_len=max_len,
            use_pos_emb=use_pos_emb,
        )

        self.act  = nn.LeakyReLU(0.1, inplace=True)
        self.head = nn.Linear(hidden, act_dim)
        self.sig  = nn.Sigmoid()

        self.apply(lambda m: orthogonal_init(m, gain=1.0))
        nn.init.orthogonal_(self.head.weight, gain=0.5)
        nn.init.zeros_(self.head.bias)

    def forward(self, s: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        s: (B, D) or (B, T, D)
        key_padding_mask: (B, T) (optional)
        """
        z = self.backbone(s, key_padding_mask=key_padding_mask)  # (B,H)
        z = self.act(z)
        a01 = self.sig(self.head(z))  # (B, act_dim)
        return a01

# ---------------------------
# Critic
# ---------------------------
class CriticNet(nn.Module):
    """
    concat(s,a) -> Transformer backbone -> LeakyReLU -> Linear -> Q
    """
    def __init__(
        self,
        obs_dim: int,
        act_dim: int = 1,
        hidden: int = 128,      # d_model
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        use_pos_emb: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden

        in_dim = obs_dim + act_dim
        self.backbone = TransformerBackbone(
            in_dim=in_dim,
            d_model=hidden,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            max_len=max_len,
            use_pos_emb=use_pos_emb,
        )

        self.act   = nn.LeakyReLU(0.1, inplace=True)
        self.q_out = nn.Linear(hidden, 1)

        self.apply(lambda m: orthogonal_init(m, gain=1.0))
        nn.init.orthogonal_(self.q_out.weight, gain=0.5)
        nn.init.zeros_(self.q_out.bias)

    def forward(self, s: torch.Tensor, a01: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        s: (B,D) or (B,T,D)
        a01: (B,A) or (B,T,A)
        key_padding_mask: (B,T) (optional)
        """
        if s.dim() == 2 and a01.dim() == 2:
            x = torch.cat([s, a01], dim=-1).unsqueeze(1)  # (B,1,D+A)
        elif s.dim() == 3 and a01.dim() == 3:
            x = torch.cat([s, a01], dim=-1)               # (B,T,D+A)
        else:
            s2 = s.view(s.size(0), -1)
            a2 = a01.view(a01.size(0), -1)
            x = torch.cat([s2, a2], dim=-1).unsqueeze(1)

        z = self.backbone(x, key_padding_mask=key_padding_mask)  # (B,H)
        z = self.act(z)
        q = self.q_out(z)  # (B,1)
        return q
