# Environment/vppbidenv.py
from typing import Tuple
from collections import deque
import numpy as np
import pandas as pd

EPS = 1e-6

class VPPBidEnv:
    """
    기존 환경(참고). 관측: [pv_prev_norm, clear_ghi_norm, ghi_norm, doy_sin_01, doy_cos_01]
    """
    def __init__(self, df: pd.DataFrame, episode_len: int = 96, lambda_ramp: float = 0.01, seed: int = 0):
        required = ["timestamp", "pv", "clear_ghi", "ghi", "doy_sin", "doy_cos"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"필수 컬럼 없음: {missing}")

        self.df = df.sort_values("timestamp").reset_index(drop=True).copy()
        self.episode_len = episode_len
        self.lambda_ramp = lambda_ramp
        self.rng = np.random.RandomState(seed)

        # 상한 스케일러(이상치 완화)
        self.A_MAX = float(np.percentile(self.df["pv"], 99))
        self.pv_std = float(self.df["pv"].std() + 1e-6)

        # 추가: GHI 스케일러(99퍼센타일 기반)
        self.CLEAR_GHI_MAX = float(max(np.percentile(self.df["clear_ghi"], 99), EPS))
        self.GHI_MAX       = float(max(np.percentile(self.df["ghi"], 99), EPS))

        self.t0 = 0
        self.ptr = 0
        self.prev_action = 0.0

    def _base_obs(self, idx: int) -> np.ndarray:
        row = self.df.iloc[idx]
        pv_prev = self.df.iloc[max(idx - 1, 0)]["pv"]

        # 정규화:
        # - pv_prev: [0, A_MAX] -> [0, 1] 근사
        # - clear_ghi, ghi: 99백분위수로 나눔 (이상치 완화)
        # - doy_sin, doy_cos: [-1,1] -> [0,1] 매핑
        clear_ghi_norm = float(row["clear_ghi"]) / self.CLEAR_GHI_MAX
        ghi_norm       = float(row["ghi"])       / self.GHI_MAX
        doy_sin_01     = 0.5 * (float(row["doy_sin"]) + 1.0)
        doy_cos_01     = 0.5 * (float(row["doy_cos"]) + 1.0)

        return np.array([
            float(pv_prev) / max(self.A_MAX, EPS),
            np.clip(clear_ghi_norm, 0.0, 1.0),
            np.clip(ghi_norm, 0.0, 1.0),
            np.clip(doy_sin_01, 0.0, 1.0),
            np.clip(doy_cos_01, 0.0, 1.0),
        ], dtype=np.float32)

    @property
    def base_obs_dim(self) -> int:
        return 5

    def reset(self) -> np.ndarray:
        max_start = len(self.df) - self.episode_len - 2
        if max_start < 2:
            raise ValueError("데이터 길이가 episode_len 대비 너무 짧음.")
        self.t0 = int(self.rng.randint(1, max(2, max_start)))
        self.ptr = 0
        self.prev_action = 0.0
        return self._base_obs(self.t0)

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        t_idx = self.t0 + self.ptr
        real = float(self.df.iloc[t_idx]["pv"])
        act  = float(np.clip(action, 0.0, self.A_MAX))

        scale = (self.A_MAX ** 2)
        err   = (act - real)
        ramp  = (act - self.prev_action)
        reward = -(err * err) / scale - self.lambda_ramp * (ramp * ramp) / scale

        self.prev_action = act
        self.ptr += 1
        done = self.ptr >= self.episode_len
        nxt_idx = self.t0 + self.ptr if not done else min(self.t0 + self.ptr, len(self.df)-1)
        next_obs = self._base_obs(nxt_idx)
        info = {"real": real, "action": act, "err": err}
        return next_obs, float(reward), bool(done), info


# =========================
# POMDP용 스택 관측 환경
# =========================
class VPPBidPOMDPEnv(VPPBidEnv):
    """
    POMDP 환경: 최근 H개의 관측 + 직전 액션(정규화)을 함께 관측으로 제공
      o_t = concat( [o_base_{t-H+1}, ..., o_base_t], [a_{t-1}/A_MAX] )
    선택: 관측 노이즈도 부여 가능(obs_noise_std).
    """
    def __init__(self, df: pd.DataFrame,
                 episode_len: int = 96,
                 lambda_ramp: float = 0.01,
                 seed: int = 0,
                 history_len: int = 4,
                 obs_noise_std: float = 0.0):
        super().__init__(df, episode_len, lambda_ramp, seed)
        self.H = int(history_len)
        self.obs_noise_std = float(obs_noise_std)
        self._hist = deque(maxlen=self.H)

    @property
    def obs_dim(self) -> int:
        # H * base_obs_dim + 1(직전 액션 정규화)
        return self.H * self.base_obs_dim + 1

    def _observe(self, idx: int) -> np.ndarray:
        o = super()._base_obs(idx).copy()
        if self.obs_noise_std > 0:
            noise = np.random.normal(0.0, self.obs_noise_std, size=o.shape).astype(np.float32)
            o += noise
        return o

    def _stacked_obs(self) -> np.ndarray:
        # 히스토리가 부족하면 가장 첫 관측으로 패딩
        if len(self._hist) < self.H:
            pad = [self._hist[0]] * (self.H - len(self._hist))
            seq = list(pad) + list(self._hist)
        else:
            seq = list(self._hist)
        o_stack = np.concatenate(seq, axis=0).astype(np.float32)       # (H*base_obs_dim,)
        a_prev = np.array([self.prev_action / self.A_MAX], np.float32) # (1,)
        return np.concatenate([o_stack, a_prev], axis=0)               # (H*D + 1,)

    def reset(self) -> np.ndarray:
        _ = super().reset()
        self._hist.clear()
        first = self._observe(self.t0)
        # 초기 H 프레임을 같은 값으로 채워 시작
        for _ in range(self.H):
            self._hist.append(first)
        return self._stacked_obs()

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        t_idx = self.t0 + self.ptr
        real = float(self.df.iloc[t_idx]["pv"])
        act  = float(np.clip(action, 0.0, self.A_MAX))

        scale = (self.A_MAX ** 2)
        err   = (real - act)
        ramp  = (act - self.prev_action)
        reward = -(err * err) / scale - self.lambda_ramp * (ramp * ramp) / scale

        self.prev_action = act
        self.ptr += 1
        done = self.ptr >= self.episode_len
        nxt_idx = self.t0 + self.ptr if not done else min(self.t0 + self.ptr, len(self.df)-1)

        # 히스토리 갱신
        self._hist.append(self._observe(nxt_idx))

        next_obs = self._stacked_obs()
        info = {"real": real, "action": act, "err": err}
        return next_obs, float(reward), bool(done), info