# Algorithm/agent.py
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import contextlib

# ==== [FIX] MKLDNN 비활성화 (역전파 버전 충돌 방지) ====
torch.backends.mkldnn.enabled = False

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Algorithm.network_transformer import ActorNet, CriticNet
from Algorithm.rolloutmemory import RolloutMemory


# =========================
# Config
# =========================
@dataclass
class AgentConfig:
    obs_dim: int
    act_dim: int = 1
    hidden_dim: int = 256

    lr_actor: float = 3e-4
    lr_critic: float = 3e-4

    # === [NEW] optimizer stability for Transformer ===
    weight_decay: float = 1e-4          # AdamW 권장
    use_adamw: bool = True

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

    # === [NEW] critic 안정화 ===
    use_huber: bool = True              # MSE 대신 SmoothL1
    huber_beta: float = 1.0
    q_target_clip: float = 0.0          # 0이면 disable, 예: 10.0~50.0 권장(보상 스케일에 맞게)

    # === [NEW] actor regularization ===
    action_l2_coef: float = 1e-4        # 너무 크면 policy가 0.5로 몰림


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

        # === [NEW] AdamW 권장 (Transformer 안정) ===
        if cfg.use_adamw:
            self.opt_actor  = optim.AdamW(self.actor.parameters(),  lr=cfg.lr_actor,  weight_decay=cfg.weight_decay)
            self.opt_critic = optim.AdamW(self.critic.parameters(), lr=cfg.lr_critic, weight_decay=cfg.weight_decay)
        else:
            self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=cfg.lr_actor)
            self.opt_critic = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

        self.memory = RolloutMemory(cfg.memory_capacity, cfg.obs_dim, cfg.act_dim, self.device)

        # 탐색
        self.epsilon   = cfg.epsilon_start
        self.noise_std = cfg.noise_std

        self._last_obs: Optional[np.ndarray] = None
        self._last_a01: Optional[float] = None

    # ---------- 에피소드 제어 ----------
    def begin_episode(self):
        self._last_obs = None
        self._last_a01 = None

    # ---------- 액션 ----------
    def act(self, obs_np: np.ndarray) -> float:
        # rollout에서는 dropout off가 좋음
        self.actor.eval()

        s = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a01 = self.actor(s).squeeze(0).cpu().numpy()  # (act_dim,)
        a01 = np.clip(a01, 0.0, 1.0)

        # exploration은 환경에만 적용
        a01_noisy = a01.copy()
        if np.random.rand() < self.epsilon:
            a01_noisy = np.random.uniform(0.0, 1.0, size=a01.shape)
        a01_noisy = np.clip(a01_noisy + np.random.normal(0.0, self.noise_std, size=a01.shape), 0.0, 1.0)

        a01_scalar = float(a01_noisy[0])
        act_real = a01_scalar * self.A_MAX

        # 메모리에는 "pure a01" 저장 (critic과 일관성 유지)
        self._last_obs = obs_np.copy()
        self._last_a01 = float(a01[0])
        return act_real

    def step_end(self, next_obs: np.ndarray, done: bool):
        if done and getattr(self.memory, "size", 0) > 0:
            last_idx = (self.memory.ptr - 1) % self.memory.capacity
            self.memory.dones[last_idx, 0] = 1.0
        self._last_obs = np.asarray(next_obs, dtype=np.float32)

    # ---------- 학습 스텝 ----------
    def _soft_update(self, net: nn.Module, tgt: nn.Module, tau: float):
        with torch.no_grad():
            for p, tp in zip(net.parameters(), tgt.parameters()):
                tp.copy_(tp * (1.0 - tau) + p * tau)

    @staticmethod
    def _set_eval_no_dropout(m: nn.Module):
        """
        Transformer 안정화 핵심:
        update(학습) 중에도 dropout을 끄고(deterministic) gradient는 유지.
        model.eval()은 grad를 막지 않음.
        """
        m.eval()

    def _critic_step(self, s, a01, r, sn, dn):
        # ====== [NEW] critic/actor target 계산 시 dropout OFF ======
        self._set_eval_no_dropout(self.actor_tgt)
        self._set_eval_no_dropout(self.critic_tgt)
        self._set_eval_no_dropout(self.critic)

        with torch.no_grad():
            a01_next = self.actor_tgt(sn)
            q_next   = self.critic_tgt(sn, a01_next)
            y        = r + self.cfg.gamma * (1.0 - dn) * q_next

            # ====== [NEW] Q target clipping (선택) ======
            if self.cfg.q_target_clip and self.cfg.q_target_clip > 0:
                y = torch.clamp(y, -self.cfg.q_target_clip, self.cfg.q_target_clip)

        q = self.critic(s, a01)

        # ====== [NEW] Huber loss로 outlier 완화 ======
        if self.cfg.use_huber:
            loss = F.smooth_l1_loss(q, y, beta=self.cfg.huber_beta)
        else:
            loss = F.mse_loss(q, y)

        self.opt_critic.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
        self.opt_critic.step()
        return float(loss.item())

    @contextlib.contextmanager
    def freeze_critic_params(self):
        req_flags = [p.requires_grad for p in self.critic.parameters()]
        for p in self.critic.parameters():
            p.requires_grad_(False)
        try:
            yield
        finally:
            for p, f in zip(self.critic.parameters(), req_flags):
                p.requires_grad_(f)

    def _actor_step(self, s):
        # ====== [NEW] actor 업데이트도 dropout OFF (deterministic policy gradient 안정) ======
        self._set_eval_no_dropout(self.actor)
        self._set_eval_no_dropout(self.critic)

        with self.freeze_critic_params():
            a01 = self.actor(s)
            q   = self.critic(s, a01)

            # ====== [FIX] 배치 인접 차이 smooth 제거(시퀀스가 아닌 batch 차원이라 의미 없음) ======
            # 대체: action L2 (너무 크면 0.5로 몰리니 coef 작게)
            act_reg = (a01 ** 2).mean()

            loss = -q.mean() + self.cfg.action_l2_coef * act_reg

            self.opt_actor.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            self.opt_actor.step()

        return float(loss.item())

    # ---------- 에피소드 종료 시 업데이트 ----------
    def update(self):
        if len(self.memory) < self.cfg.batch_size:
            out = (0.0, 0.0, 0.0)
            self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)
            self.begin_episode()
            return out

        s, a, r, sn, dn = self.memory.sample(self.cfg.batch_size)

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
