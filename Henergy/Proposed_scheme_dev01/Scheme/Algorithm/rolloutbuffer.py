# Algorithm/RolloutBuffer.py
from typing import List
import numpy as np
import torch

class RolloutBuffer:
    """에피소드 하나의 on-policy trajectory 저장소"""
    def __init__(self, device: torch.device):
        self.device = device
        self.clear()

    def store(self, logp, value, entropy, reward, action_real, real):
        # logp/value/entropy는 텐서(grad 유지), reward는 float → 텐서로 저장
        self.logps.append(logp)
        self.values.append(value)
        self.entropies.append(entropy)
        self.rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
        self.actions_real.append(float(action_real))  # kW
        self.reals.append(float(real))                # kW

    def get_tensors(self):
        # 학습 시 한 번에 stack
        return (
            torch.stack(self.logps),
            torch.stack(self.values),
            torch.stack(self.entropies),
            torch.stack(self.rewards),
        )

    def mae(self) -> float:
        # 액션(kW) vs 실제 발전량(kW)의 평균절대오차
        if not self.actions_real:
            return float("nan")
        a = np.array(self.actions_real)
        r = np.array(self.reals)
        return float(np.mean(np.abs(a - r)))
    
    def mse(self) -> float:
        """액션(kW) vs 실제 발전량(kW)의 평균제곱오차 (단위: kW^2)"""
        if not self.actions_real:
            return float("nan")
        a = np.array(self.actions_real, dtype=np.float32)
        r = np.array(self.reals, dtype=np.float32)
        return float(np.mean((a - r) ** 2))

    def clear(self):
        self.logps:   List[torch.Tensor] = []
        self.values:  List[torch.Tensor] = []
        self.entropies: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.actions_real: List[float] = []
        self.reals:  List[float] = []