# Algorithm/RolloutMemory.py
from typing import List
import numpy as np
import torch

class RolloutMemory:
    """
    간단한 리플레이 버퍼
    - store(state, action_real, reward, next_state, done=False)
    - sample(batch_size) -> (s, a_real, r, s_next, done)
    """
    def __init__(self, capacity: int, obs_dim: int, act_dim: int = 1, device: torch.device = torch.device("cpu")):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = device

        self.states = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions_real = np.zeros((self.capacity, self.act_dim), dtype=np.float32)  # kW 단위 실액션
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def store(self,
              state: np.ndarray,
              action_real: float,
              reward: float,
              next_state: np.ndarray,
              done: bool = False) -> None:
        """(s, a_real[kW], r, s', done) 저장"""
        i = self.ptr

        # 상태/다음상태는 1D 벡터로 강제
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        sn = np.asarray(next_state, dtype=np.float32).reshape(-1)
        if s.shape[0] != self.obs_dim or sn.shape[0] != self.obs_dim:
            raise ValueError(f"state/next_state dim mismatch: expected {self.obs_dim}, got {s.shape[0]}/{sn.shape[0]}")

        self.states[i] = s
        # action_real 을 (act_dim,)으로 저장
        a = np.asarray([action_real], dtype=np.float32) if self.act_dim == 1 else np.asarray(action_real, dtype=np.float32)
        self.actions_real[i] = a.reshape(self.act_dim)
        self.rewards[i, 0] = np.float32(reward)
        self.next_states[i] = sn
        self.dones[i, 0] = np.float32(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """무작위 미니배치 샘플링 (torch.Tensor로 반환)"""
        if self.size == 0:
            raise RuntimeError("Replay buffer is empty.")
        bs = min(int(batch_size), self.size)
        idx = np.random.randint(0, self.size, size=bs)

        s = torch.tensor(self.states[idx], device=self.device)
        a_real = torch.tensor(self.actions_real[idx], device=self.device)
        r = torch.tensor(self.rewards[idx], device=self.device)
        sn = torch.tensor(self.next_states[idx], device=self.device)
        d = torch.tensor(self.dones[idx], device=self.device)

        return s, a_real, r, sn, d

    def __len__(self) -> int:
        return self.size

    def clear(self) -> None:
        self.ptr = 0
        self.size = 0