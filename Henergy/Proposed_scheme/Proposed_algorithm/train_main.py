# train_main.py
import argparse
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd
import torch

import os 
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Algorithm.agent import AgentA2C, AgentConfig

# ------------------------------
# 환경 정의 train_main.py 
# ------------------------------
class VPPBidEnv:
    """
    Model-Free 데이터 환경.
    관측: [pv_prev_norm, ghi, temp_norm, wind_norm, cloud, time-embeds]
    행동: 15분 입찰량(출력발전) [0, A_MAX] (연속)
    보상: -(오차^2 + lambda_ramp * 램프^2) / A_MAX^2
    """
    def __init__(self, df: pd.DataFrame, episode_len: int = 96, lambda_ramp: float = 0.01, seed: int = 0):
        required = ["timestamp", "pv_total", "ghi_sim",
                    "tod_sin", "tod_cos", "doy_sin", "doy_cos"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"환경 구성에 필요한 컬럼이 없습니다: {missing}\n"
                             f"data_generator.py가 만든 병합시트(기본 'data15m')를 전달했는지 확인.")
        self.df = df.sort_values("timestamp").reset_index(drop=True).copy()
        self.episode_len = episode_len
        self.lambda_ramp = lambda_ramp
        self.rng = np.random.RandomState(seed)

        self.A_MAX = float(np.percentile(self.df["pv_total"], 99))
        self.pv_std   = float(self.df["pv_total"].std()   + 1e-6)
        # self.temp_std = float(self.df["temp_sim"].std()   + 1e-6)
        # self.wind_std = float(self.df["wind_sim"].std()   + 1e-6)

        self.t0 = 0
        self.ptr = 0
        self.prev_action = 0.0

    def _get_obs(self, idx: int) -> np.ndarray:
        row = self.df.iloc[idx]
        pv_prev = self.df.iloc[max(idx - 1, 0)]["pv_total"]
        obs = np.array([
            pv_prev / self.A_MAX,
            row["ghi_sim"],
            #(row["temp_sim"] / (self.temp_std * 4)),
            #(row["wind_sim"] / (self.wind_std * 4)),
            #row["cloud_sim"],
            row["tod_sin"], row["tod_cos"],
            row["doy_sin"], row["doy_cos"],
        ], dtype=np.float32)
        return obs

    @property
    def obs_dim(self):
        return 6

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
        real = float(self.df.iloc[t_idx]["pv_total"])
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

# ------------------------------
# 훈련 설정/인자
# ------------------------------
@dataclass
class Config:
    data_path: str = "./Project/Henergy/Proposed_scheme/Data_generator/Data_output/generated_15min_data_pv_ghi.xlsx"  # data_generator.py 출력 파일 경로
    sheet_name: str = "data15m"                   # 병합 데이터 시트명(기본 data15m)
    episodes: int = 100                         # 기본 100
    episode_len: int = 96                         # 15분×96=하루
    lambda_ramp: float = 0.01
    gamma: float = 0.99
    lr: float = 3e-4
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    seed: int = 7

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train A2C agent on 15-min VPP bidding using pre-generated .xlsx")
    p.add_argument("--data_path", type=str, default="./Project/Henergy/Proposed_scheme/Data_generator/Data_output/generated_15min_data_pv_ghi.xlsx",
                   help="data_generator.py가 생성한 .xlsx 경로")
    p.add_argument("--sheet_name", type=str, default="data15m",
                   help="불러올 시트명 (기본: data15m). --separate_sheets 사용 시 'data15m' 또는 병합시트명 지정")
    p.add_argument("--episodes", type=int, default=10000, help="학습 에피소드 수")
    p.add_argument("--episode_len", type=int, default=96, help="에피소드 길이(15분×96=하루)")
    p.add_argument("--lambda_ramp", type=float, default=0.01, help="램프 패널티 가중치")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy_coef", type=float, default=1e-3)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    return Config(**vars(args))

# ------------------------------
# 유틸: 엑셀 로드 & 검증
# ------------------------------
def load_merged_xlsx(path: str, sheet: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f".xlsx 파일을 찾을 수 없습니다: {path}")
    df = pd.read_excel(path, sheet_name=sheet)
    # timestamp → datetime 강제 변환
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    # 필요한 컬럼 존재 확인은 환경에서 다시 한 번 체크
    return df

# ------------------------------
# 메인
# ------------------------------
import os

def main():
    cfg = parse_args()

    # 시드/디바이스
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) .xlsx 로드 (data_generator.py 출력)
    df = load_merged_xlsx(cfg.data_path, cfg.sheet_name)

    # 2) 환경
    env = VPPBidEnv(df, episode_len=cfg.episode_len, lambda_ramp=cfg.lambda_ramp, seed=cfg.seed)
    A_MAX = env.A_MAX

    # 3) 에이전트
    agent_cfg = AgentConfig(
        obs_dim=env.obs_dim,
        hidden_dim=128,
        lr=cfg.lr,
        gamma=cfg.gamma,
        entropy_coef=cfg.entropy_coef,
        value_coef=cfg.value_coef,
        max_grad_norm=cfg.max_grad_norm,
        device=device
    )
    agent = AgentA2C(agent_cfg, A_MAX)

    # 4) 학습 루프(rollout → buffer 적재 → update)
    for ep in range(1, cfg.episodes + 1):
        obs = env.reset()
        agent.begin_episode()

        for _ in range(cfg.episode_len):
            act_real, logp, value, entropy = agent.act(obs)
            next_obs, reward, done, info = env.step(act_real)
            agent.store(logp, value, entropy, reward, act_real, info["real"])
            obs = next_obs
            if done:
                break

        loss, ploss, vloss, ep_reward, ep_mae = agent.update()

        if ep % max(1, cfg.episodes // 20) == 0 or ep == 1:
            print(f"[EP {ep:4d}/{cfg.episodes}] loss={loss:.4f} (π={ploss:.4f}, V={vloss:.4f})  "
                  f"R={ep_reward:.4f}  MAE(kW)={ep_mae:.2f}  A_MAX={A_MAX:.1f}")

    # 5) 최종 저장
    agent.save("vpp_actorcritic_lstm.pt", obs_dim=env.obs_dim)
    print("모델이 저장되었습니다: vpp_actorcritic_lstm.pt")

if __name__ == "__main__":
    main()
