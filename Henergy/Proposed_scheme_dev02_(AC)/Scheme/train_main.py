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

# === [NEW] 그래프 출력용 ===
import matplotlib
matplotlib.use("Agg")  # 서버/터미널 환경에서 그림 저장만 할 때 권장
import matplotlib.pyplot as plt
# ==========================

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
    p.add_argument("--data_path", type=str, default="./Project/Henergy/Proposed_scheme_dev01/Data_generator/Data_output/generated_1hour_data_pv_ghi_clearghi_output_y.xlsx",
                   help="data_generator.py가 생성한 .xlsx 경로")
    p.add_argument("--sheet_name", type=str, default="data_with_time_feats",
                   help="불러올 시트명 (기본: data15m). --separate_sheets 사용 시 병합시트명 지정")
    p.add_argument("--episodes", type=int, default=100000, help="학습 에피소드 수")
    p.add_argument("--episode_len", type=int, default=24, help="에피소드 길이(60분×24=하루)")
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
        act_dim=1,
        hidden_dim=128,
        lr_actor=cfg.lr,
        lr_critic=cfg.lr,
        gamma=cfg.gamma,
        tau=5e-3,
        batch_size=128,
        memory_capacity=10000,
        epsilon_start=0.3,
        epsilon_decay=0.999,
        epsilon_min=0.05,
        noise_std=0.1,
        max_grad_norm=cfg.max_grad_norm,
        device=device
    )
    agent = AgentA2C(agent_cfg, A_MAX)

    # --- 로그 파일 준비 ---
    log_dir = "./Project/Henergy/Proposed_scheme_dev02_(AC)/Logs"
    os.makedirs(log_dir, exist_ok=True)
    log_csv = os.path.join(log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    logs = []
    fig_dir = "./Project/Henergy/Proposed_scheme_dev02_(AC)/Visualization"

    # 4) 학습 루프
    for ep in range(1, cfg.episodes + 1):
        obs = env.reset()
        agent.begin_episode()

        # ---- 에피소드 통계 누적 변수 ----
        ep_reward_sum = 0.0
        abs_err_sum = 0.0
        step_cnt = 0
        # --------------------------------

        for _ in range(cfg.episode_len):
            act_real = agent.act(obs)
            next_obs, reward, done, info = env.step(act_real)

            # (state, action, reward, next_state) 저장
            agent.memory.store(obs, act_real, reward, next_obs)
            agent.step_end(next_obs, done)

            # ---- MAE/R 누적 (info['err'] 사용) ----
            abs_err_sum += abs(float(info["err"]))  # |act - real|
            ep_reward_sum += float(reward)
            step_cnt += 1

            obs = next_obs
            if done:
                break

        # 배치 업데이트(Actor/Critic)
        loss, ploss, vloss = agent.update()

        # 에피소드 MAE 계산
        mae = abs_err_sum / max(1, step_cnt)

        # 콘솔 로그 (간격 출력)
        if ep % 10 == 0 or ep == 1:
            print(f"[EP {ep:4d}/{cfg.episodes}] loss={loss:.4f} (π={ploss:.4f}, V={vloss:.4f}) "
                  f"R={ep_reward_sum:.4f}  MAE(kW)={mae:.2f}  A_MAX={A_MAX:.1f}")

        # CSV 누적 & 저장
        logs.append({
            "episode": ep,
            "loss_total": float(loss),
            "loss_phi(policy)": float(ploss),
            "loss_V(value)": float(vloss),
            "R": float(ep_reward_sum),
            "MAE_kW": float(mae)
        })
        pd.DataFrame(logs).to_csv(log_csv, index=False)

    # === [NEW] 에피소드 로그로 학습 곡선 그리기 ===
    log_df = pd.DataFrame(logs)
    if len(log_df) > 0:
        fig, axs = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

        # (1) Losses
        axs[0].plot(log_df["episode"], log_df["loss_total"], label="Total Loss")
        axs[0].plot(log_df["episode"], log_df["loss_phi(policy)"], alpha=0.6, label="Policy Loss (π)")
        axs[0].plot(log_df["episode"], log_df["loss_V(value)"], alpha=0.6, label="Value Loss (V)")
        axs[0].set_ylabel("Loss")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        # (2) Return R
        axs[1].plot(log_df["episode"], log_df["R"])
        axs[1].set_ylabel("Episode Return (R)")
        axs[1].grid(True, alpha=0.3)

        # (3) MAE
        axs[2].plot(log_df["episode"], log_df["MAE_kW"])
        axs[2].set_ylabel("MAE (kW)")
        axs[2].set_xlabel("Episode")
        axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(fig_dir, f"train_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(fig_path, dpi=150)
        print(f"학습 곡선 그래프 저장: {os.path.abspath(fig_path)}")
        plt.close(fig)

    # 모델 저장
    agent.save("vpp_actorcritic_lstm.pt", obs_dim=env.obs_dim)
    print("모델이 저장되었습니다: vpp_actorcritic_lstm.pt")
    print(f"에피소드 로그 CSV: {os.path.abspath(log_csv)}")

if __name__ == "__main__":
    main()
