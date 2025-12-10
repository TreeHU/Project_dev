# Algorithm/agent.py
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ==== [FIX] MKLDNN LSTM 비활성화 (역전파 버전 충돌 방지) ====
torch.backends.mkldnn.enabled = False

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Algorithm.network import ActorNet, CriticNet
from Algorithm.rolloutmemory import RolloutMemory


# =========================
# Config
# =========================
@dataclass
class AgentConfig:
    obs_dim: int
    act_dim: int = 1
    hidden_dim: int = 128
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 5e-3
    batch_size: int = 128
    memory_capacity: int = 100_000
    epsilon_start: float = 0.3
    epsilon_decay: float = 0.999
    epsilon_min: float = 0.05
    noise_std: float = 0.1
    max_grad_norm: float = 1.0
    device: torch.device = torch.device("cpu")


class AgentA2C:
    """
    결정적 Actor–Critic (DDPG 스타일)
    - ActorNet: s -> a01 \in (0,1)
    - CriticNet: (s, a01) -> Q(s,a)
    """
    def __init__(self, cfg: AgentConfig, A_MAX: float):
        self.cfg = cfg
        self.device = cfg.device
        self.A_MAX = float(A_MAX)

        # 네트워크/타깃
        self.actor      = ActorNet(cfg.obs_dim, cfg.act_dim, cfg.hidden_dim).to(self.device)
        self.critic     = CriticNet(cfg.obs_dim, cfg.act_dim, cfg.hidden_dim).to(self.device)
        self.actor_tgt  = ActorNet(cfg.obs_dim, cfg.act_dim, cfg.hidden_dim).to(self.device)
        self.critic_tgt = CriticNet(cfg.obs_dim, cfg.act_dim, cfg.hidden_dim).to(self.device)
        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=cfg.lr_actor)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

        self.memory = RolloutMemory(cfg.memory_capacity, cfg.obs_dim, cfg.act_dim, self.device)

        # 탐색
        self.epsilon   = cfg.epsilon_start
        self.noise_std = cfg.noise_std

        # 로그용 누적
        self._ep_reward = 0.0
        self._ep_err_abs_sum = 0.0
        self._ep_steps = 0

        self._last_obs: Optional[np.ndarray] = None
        self._last_a01: Optional[float] = None

        self.pv_prev_norm_idx: int = 0

    # ---------- 에피소드 제어 ----------
    def begin_episode(self):
        self._ep_reward = 0.0
        self._ep_err_abs_sum = 0.0
        self._ep_steps = 0
        self._last_obs = None
        self._last_a01 = None

    # ---------- 액션 ----------
    def act(self, obs_np: np.ndarray) -> float:
        self.actor.train()
        s = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, obs_dim)
        with torch.no_grad():
            a01 = self.actor(s).squeeze(0)  # (act_dim,)
        a01 = a01.clamp(0.0, 1.0).cpu().numpy()

        # ε-mix + 가우시안 노이즈
        if np.random.rand() < self.epsilon:
            a01 = np.random.uniform(0.0, 1.0, size=a01.shape)
        a01 = np.clip(a01 + np.random.normal(0.0, self.noise_std, size=a01.shape), 0.0, 1.0)

        a01_scalar = float(a01[0])
        act_real = a01_scalar * self.A_MAX

        self._last_obs = obs_np.copy()
        self._last_a01 = a01_scalar
        return act_real

    def step_end(self, next_obs: np.ndarray, done: bool):
        # 마지막 샘플의 done만 갱신
        if done and getattr(self.memory, "size", 0) > 0:
            last_idx = (self.memory.ptr - 1) % self.memory.capacity
            self.memory.dones[last_idx, 0] = 1.0
        self._last_obs = np.asarray(next_obs, dtype=np.float32)

    # ---------- 학습 스텝 ----------
    def _soft_update(self, net: nn.Module, tgt: nn.Module, tau: float):
        # ==== [FIX] in-place 제거, copy_ 사용 ====
        with torch.no_grad():
            for p, tp in zip(net.parameters(), tgt.parameters()):
                tp.copy_(tp * (1.0 - tau) + p * tau)

    def _critic_step(self, s, a01, r, sn, dn):
        # target: y = r + gamma*(1-done)*Q_tgt(sn, actor_tgt(sn))
        with torch.no_grad():
            a01_next = self.actor_tgt(sn)
            q_next   = self.critic_tgt(sn, a01_next)
            y        = r + self.cfg.gamma * (1.0 - dn) * q_next

        q = self.critic(s, a01)
        loss = F.mse_loss(q, y)  # ==== [FIX] F.mse_loss 사용 ====

        self.opt_critic.zero_grad(set_to_none=True)  # ==== [FIX]
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.opt_critic.step()
        return loss.item()

    def _actor_step(self, s):
        a01 = self.actor(s)
        q = self.critic(s, a01)
        loss = -q.mean()

        self.opt_actor.zero_grad(set_to_none=True)  # ==== [FIX]
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
        self.opt_actor.step()
        return loss.item()

    # ---------- 에피소드 종료 시 업데이트 ----------
    def update(self):
        if len(self.memory) < self.cfg.batch_size:
            out = (0.0, 0.0, 0.0)
            self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)
            self.begin_episode()
            return out

        s, a, r, sn, dn = self.memory.sample(self.cfg.batch_size)

        # ==== [FIX] 안전한 변환(as_tensor + to) ====
        s  = torch.as_tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(a,  dtype=torch.float32, device=self.device)  # (B,1) in a01-space
        r  = torch.as_tensor(r,  dtype=torch.float32, device=self.device)
        sn = torch.as_tensor(sn, dtype=torch.float32, device=self.device)
        dn = torch.as_tensor(dn, dtype=torch.float32, device=self.device)

        v_loss = self._critic_step(s, a, r, sn, dn)
        p_loss = self._actor_step(s)

        self._soft_update(self.actor,  self.actor_tgt,  self.cfg.tau)
        self._soft_update(self.critic, self.critic_tgt, self.cfg.tau)

        total_loss = p_loss + v_loss

        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

        self.begin_episode()
        return float(total_loss), float(p_loss), float(v_loss)

    # ---------- 저장 ----------
    def save(self, path: str, obs_dim: int):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_tgt": self.actor_tgt.state_dict(),
            "critic_tgt": self.critic_tgt.state_dict(),
            "A_MAX": self.A_MAX,
            "obs_dim": obs_dim
        }, path)
