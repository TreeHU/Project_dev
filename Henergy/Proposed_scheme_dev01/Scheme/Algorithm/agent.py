# Algorithm/agent.py
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Algorithm.network import RecurrentActorCritic
from Algorithm.rolloutbuffer import RolloutBuffer   # 추가: 분리된 버퍼 임포트

# 정책이 Beta(알파, 베타) 분포를 쓰므로, 액션 a01 = [0,1] 의 log_prob 와 분포 객체를 반환
def beta_log_prob(alpha: torch.Tensor, beta: torch.Tensor, a01: torch.Tensor):
    from torch.distributions import Beta
    dist = Beta(alpha, beta)
    lp = dist.log_prob(torch.clamp(a01, 1e-6, 1 - 1e-6))
    return lp, dist


@dataclass
class AgentConfig:
    obs_dim: int
    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    device: torch.device = torch.device("cpu")

class AgentA2C:
    """Actor-Critic(LSTM, Beta policy) + A2C 업데이트"""
    def __init__(self, cfg: AgentConfig, A_MAX: float):
        self.cfg = cfg
        self.device = cfg.device
        self.model = RecurrentActorCritic(cfg.obs_dim, cfg.hidden_dim).to(self.device)
        self.opt   = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.buffer = RolloutBuffer(self.device)  # 분리된 버퍼 사용
        self.h: Optional[torch.Tensor] = None
        self.c: Optional[torch.Tensor] = None
        self.A_MAX = A_MAX

    def begin_episode(self):
        self.h, self.c = self.model.init_state(batch_size=1)
        self.h, self.c = self.h.to(self.device), self.c.to(self.device)
        self.buffer.clear()

    def act(self, obs_np: np.ndarray):
        """정책에서 액션 샘플 (on-policy rollout)"""
        self.model.train()
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device).view(1, 1, -1)
        alpha, beta, value, (self.h, self.c) = self.model(obs_t, (self.h, self.c))
        dist = torch.distributions.Beta(alpha, beta)
        a01 = dist.rsample()
        logp, dist2 = beta_log_prob(alpha, beta, a01)
        entropy = dist2.entropy()
        act_real = (a01.squeeze().item()) * self.A_MAX
        return act_real, logp.squeeze(), value.squeeze(), entropy.squeeze()

    def store(self, logp, value, entropy, reward, action_real, real):
        self.buffer.store(logp, value, entropy, reward, action_real, real)

    def update(self):
        # Returns / Advantages
        logps_t, values_t, entropies_t, rewards_t = self.buffer.get_tensors()
        with torch.no_grad():
            G = torch.zeros(1, device=self.device)
        returns = []
        for r in reversed(rewards_t):
            G = r + self.cfg.gamma * G
            returns.append(G)
        returns_t = torch.stack(list(reversed(returns)))

        advantages = returns_t - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        policy_loss = -(advantages.detach() * logps_t).mean() - self.cfg.entropy_coef * entropies_t.mean()
        value_loss  = self.cfg.value_coef * (returns_t.detach() - values_t).pow(2).mean()
        loss = policy_loss + value_loss

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        self.opt.step()

        ep_reward = float(rewards_t.sum().item())
        ep_mae = self.buffer.mae()
        self.buffer.clear()
        return loss.item(), policy_loss.item(), value_loss.item(), ep_reward, ep_mae

    def save(self, path: str, obs_dim: int):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "A_MAX": self.A_MAX,
            "obs_dim": obs_dim
        }, path)
