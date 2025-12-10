# train_main.py
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List
import numpy as np
import pandas as pd
import torch

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Algorithm.agent import AgentA2C, AgentConfig
from Environment.vppbidenv import VPPBidEnv   # 

# ------------------------------
# 훈련 설정/인자
# ------------------------------
@dataclass
class Config:
    data_path: str = "./Project/Henergy/Proposed_scheme/Data_generator/Data_output/generated_1hour_data_pv_ghi_clearghi_output_y.xlsx"
    sheet_name: str = "data_with_time_feats"
    episodes: int = 100
    episode_len: int = 96
    lambda_ramp: float = 0.01
    gamma: float = 0.99
    lr: float = 3e-4
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    seed: int = 7

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train A2C agent on 15-min VPP bidding using pre-generated .xlsx")
    p.add_argument("--data_path", type=str, default="./Project/Henergy/Henergy/Proposed_scheme_dev01/Data_generator/Data_output/generated_1hour_data_pv_ghi_clearghi_output_y.xlsx",
                   help="data_generator.py가 생성한 .xlsx 경로")
    p.add_argument("--sheet_name", type=str, default="data_with_time_feats",
                   help="불러올 시트명 (기본: data15m). --separate_sheets 사용 시 병합시트명 지정")
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
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

# ------------------------------
# 메인
# ------------------------------
def main():
    cfg = parse_args()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) .xlsx 로드
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

    # --- 로그 파일 준비 ---
    log_dir = "./Project/Henergy/Henergy/Proposed_scheme_dev01/Logs"
    os.makedirs(log_dir, exist_ok=True)
    log_csv = os.path.join(log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    logs = []   # 메모리 버퍼(매 에피소드 파일로 덮어쓰기 저장)

    # 4) 학습 루프
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

        # 콘솔 로그
        print(f"[EP {ep:4d}/{cfg.episodes}] loss={loss:.4f} (π={ploss:.4f}, V={vloss:.4f}) "
              f"R={ep_reward:.4f}  MAE(kW)={ep_mae:.2f}  A_MAX={A_MAX:.1f}")

        # --- CSV 로그 누적 & 저장 ---
        logs.append({
            "episode": ep,
            "loss_total": float(loss),
            "loss_phi(policy)": float(ploss),
            "loss_V(value)": float(vloss),
            "R": float(ep_reward),
            "MAE_kW": float(ep_mae)
        })
        # 매 에피소드마다 즉시 저장(덮어쓰기)
        pd.DataFrame(logs).to_csv(log_csv, index=False)

    agent.save("vpp_actorcritic_lstm.pt", obs_dim=env.obs_dim)
    print("모델이 저장되었습니다: vpp_actorcritic_lstm.pt")
    print(f"에피소드 로그 CSV: {os.path.abspath(log_csv)}")

if __name__ == "__main__":
    main()
