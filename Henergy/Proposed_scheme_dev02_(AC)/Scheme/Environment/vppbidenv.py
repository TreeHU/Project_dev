# vppbidenv.py
from typing import Tuple
import numpy as np
import pandas as pd

class VPPBidEnv:
    """
    Model-Free 데이터 환경.
    관측: [pv_prev_norm, ghi, time-embeds]
    행동: 1시간 출력발전 [0, A_MAX] (연속)
    보상: -(오차^2 + lambda_ramp * 램프^2) / A_MAX^2
    """
    def __init__(self, df: pd.DataFrame, episode_len: int = 96, lambda_ramp: float = 0.01, seed: int = 0):
        required = ["timestamp", "pv", "clear_ghi", "ghi",
                    "doy_sin", "doy_cos"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"환경 구성에 필요한 컬럼이 없습니다: {missing}\n"
                f"data_generator.py가 만든 병합시트(기본 'data15m')를 전달했는지 확인."
            )

        self.df = df.sort_values("timestamp").reset_index(drop=True).copy()
        self.episode_len = episode_len
        self.lambda_ramp = lambda_ramp
        self.rng = np.random.RandomState(seed)

        self.A_MAX = float(np.percentile(self.df["pv"], 99))  # 이상치 완화 상한 #상위 1%만 잘라 안정적인 상한을 줌
        self.pv_std = float(self.df["pv"].std() + 1e-6)

        self.t0 = 0
        self.ptr = 0
        self.prev_action = 0.0

    def _get_obs(self, idx: int) -> np.ndarray:
        row = self.df.iloc[idx]
        pv_prev = self.df.iloc[max(idx - 1, 0)]["pv"]
        obs = np.array([
            pv_prev / self.A_MAX,   # pv_prev 정규화
            row["clear_ghi"],
            row["ghi"],
            row["doy_sin"], row["doy_cos"],
        ], dtype=np.float32)
        return obs

    @property
    def obs_dim(self) -> int:
        # [timestamp, pv, clear_ghi, ghi, doy_sin, doy_cos] = 6
        return 5

    def reset(self) -> np.ndarray:
        max_start = len(self.df) - self.episode_len - 2
        if max_start < 2:
            raise ValueError("에피소드 길이에 비해 데이터가 너무 짧음. episode_len을 줄이거나 더 긴 데이터를 사용 필요.")
        self.t0 = int(self.rng.randint(1, max(2, max_start)))
        self.ptr = 0
        self.prev_action = 0.0
        return self._get_obs(self.t0)

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
        next_obs = self._get_obs(self.t0 + self.ptr if not done else min(self.t0 + self.ptr, len(self.df)-1))
        info = {"real": real, "action": act, "err": err}
        return next_obs, float(reward), bool(done), info
