"""
프로젝트 : 강화학습 기반 전력 거래 입찰 전략
일자 : 2025-08-28

구현 : VPP Day-Ahead/Intraday 15-min Bidding with Actor-Critic (Model-Free)
"""

import os
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Union

def resample_impute(df: pd.DataFrame, ts_col: str, freq_minutes: int) -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col])
    out = out.sort_values(ts_col).set_index(ts_col)
    rule = f"{freq_minutes}T"
    out = out.resample(rule).mean()
    out = out.interpolate(limit_direction="both")
    out = out.reset_index()
    return out

def _find_col(candidates, columns):
    cols_lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None

class VPPBidEnv:
    """
    Model-Free 데이터 환경.

    """
    def __init__(self, df: pd.DataFrame, episode_len: int = 96, lambda_ramp: float = 0.01, seed: int = 0):
        assert "pv_total" in df.columns
        self.df = df.reset_index(drop=True).copy()
        self.episode_len = episode_len
        self.lambda_ramp = lambda_ramp
        self.rng = np.random.RandomState(seed)

        self.A_MAX = float(np.percentile(self.df["pv_total"], 99)) #이상치 완화 상한, 오름차순 하여 아래에서 데이터가 99% 인 값을 반환
        self.pv_std = float(self.df["pv_total"].std() + 1e-6) 
        self.temp_std = float(self.df["temp_sim"].std() + 1e-6)
        self.wind_std = float(self.df["wind_sim"].std() + 1e-6)

        self.t0 = 0
        self.ptr = 0
        self.prev_action = 0.0

    def _get_obs(self, idx: int) -> np.ndarray:
        row = self.df.iloc[idx]
        pv_prev = self.df.iloc[max(idx -1, 0)]["pv_total"]
        obs = np.array([
            pv_prev / self.A_MAX, #pv_prev_norm
            row["ghi"], # [0,1]
            (row["temp_sim"] / (self.temp_std * 4)),
            (row["wind_sim"] / (self.wind_std * 4)),
            row["cloud_sim"],
            row["tod_sin"], row["tod_cos"],
            row["doy_sin"], row["doy_cos"],
        ], dtype=np.float32)
        return obs
    
    @property
    def obs_dim(self):
        return 9
    
    def reset(self) -> np.ndarray:
        max_start = len(self.df) - self.episode_len - 2
        self.t0 = int(self.rng.randint(1, max(2, max_start)))
        self.ptr = 0
        self.prev_action = 0.0
        return self._get_obs(self.t0)
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        """
        action: [0, A_MAX] 범위의 실수 
        """
        t_idx = self.t0 + self.ptr
        real = float(self.df.iloc[t_idx]["pv_totla"])
        act = float(np.clip(action, 0.0, self.A_MAX))

        scale = (self.A_MAX ** 2)
        err = (act - real)
        ramp = (act - self.prev_action)
        reward = -(err * err) / scale - self.lambda_ramp * (ramp * ramp) /scale

        self.prev_action = act
        self.ptr += 1
        done = self.ptr >= self.episode_len
        next_obs = self._get_obs(self.t0 + self.ptr)
        info = {"real": real, "action": act, "err": err}
        return next_obs, float(reward), bool(done), info
    
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
        beta = self.softplus(beta_raw) + 1.0
        return alpha, beta

class RecurrentActorCritic(nn.Module):
    """
    공유 인코더 → LSTM → (정책 Beta-Head, 가치 V-Head)
    - 입력: obs_t (배치 x obs_dim)
    - LSTM이 Latent Vector를 생성(요구사항의 'State: Latent Vector')
    """
    



