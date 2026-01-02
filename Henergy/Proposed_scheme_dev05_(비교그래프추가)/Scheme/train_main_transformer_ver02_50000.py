# train_main.py
import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import List
import numpy as np
import pandas as pd
import torch
import time

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Algorithm.agent_transformer import AgentA2C, AgentConfig
from Environment.vppbidenv import VPPBidPOMDPEnv

from zoneinfo import ZoneInfo

# === 그래프 출력용 ===
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# ====================


# ------------------------------
# 훈련 설정/인자
# ------------------------------
@dataclass
class Config:
    data_path: str = "./Project/Henergy/Proposed_scheme_dev05_(비교그래프추가)/Data_generator/Data_output/generated_1hour_data_pv_ghi_clearghi_output_y.xlsx"
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

    # === [NEW] efficiency/scaling benchmark ===
    do_scaling_bench: bool = False
    bench_episodes: int = 50
    bench_T_list: str = "8,16,24,48,96"


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train A2C agent on 15-min VPP bidding using pre-generated .xlsx")
    p.add_argument("--data_path", type=str,
                   default="./Project/Henergy/Proposed_scheme_dev05_(비교그래프추가)/Data_generator/Data_output/generated_1hour_data_pv_ghi_clearghi_output_y.xlsx",
                   help="data_generator.py가 생성한 .xlsx 경로")
    p.add_argument("--sheet_name", type=str, default="data_with_time_feats",
                   help="불러올 시트명 (기본: data15m). --separate_sheets 사용 시 병합시트명 지정")
    p.add_argument("--episodes", type=int, default=50000, help="학습 에피소드 수")
    p.add_argument("--episode_len", type=int, default=24, help="에피소드 길이(60분×24=하루)")
    p.add_argument("--lambda_ramp", type=float, default=0.01, help="램프 패널티 가중치")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--entropy_coef", type=float, default=1e-3)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)

    # === [NEW] efficiency/scaling benchmark ===
    p.add_argument("--do_scaling_bench", action="store_true",
                   help="episode_len(T) 스윕 벤치마크 실행 및 스케일링 그래프 저장")
    p.add_argument("--bench_episodes", type=int, default=50,
                   help="각 T에서 벤치마크로 돌릴 에피소드 수")
    p.add_argument("--bench_T_list", type=str, default="8,16,24,48,96",
                   help="벤치마크할 episode_len 리스트(예: '8,16,24,48,96')")

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
# [NEW] 스케일링 벤치마크: episode_len(T)별 update_time / peak_mem 측정
# ------------------------------
def run_scaling_benchmark(
    df: pd.DataFrame,
    base_cfg: Config,
    agent_cfg: AgentConfig,
    A_MAX: float,
    device: torch.device,
    T_list: List[int],
    bench_episodes: int,
    fig_dir: str,
    tag: str,
):
    results = []

    for T in T_list:
        env = VPPBidPOMDPEnv(
            df,
            episode_len=T,
            lambda_ramp=base_cfg.lambda_ramp,
            seed=base_cfg.seed,
            history_len=4,
            obs_noise_std=0.0
        )

        # agent를 새로 생성 (가중치/리플레이/옵티마 상태 초기화)
        agent_cfg_T = AgentConfig(**{**agent_cfg.__dict__, "obs_dim": env.obs_dim, "device": device})
        agent = AgentA2C(agent_cfg_T, A_MAX)

        upd_times = []
        peak_mems = []

        for _ in range(bench_episodes):
            obs = env.reset()
            agent.begin_episode()

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            step_cnt = 0
            for _ in range(T):
                act_real = agent.act(obs)
                next_obs, reward, done, info = env.step(act_real)

                act_real_scalar = act_real / A_MAX
                agent.memory.store(obs, act_real_scalar, reward, next_obs)
                agent.step_end(next_obs, done)

                obs = next_obs
                step_cnt += 1
                if done:
                    break

            t0 = time.perf_counter()
            agent.update()
            t1 = time.perf_counter()
            upd_times.append(t1 - t0)

            if device.type == "cuda":
                peak_mems.append(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
            else:
                peak_mems.append(0.0)

        results.append({
            "T": int(T),
            "bench_episodes": int(bench_episodes),
            "update_time_sec_mean": float(np.mean(upd_times)),
            "update_time_sec_std": float(np.std(upd_times)),
            "peak_mem_mib_mean": float(np.mean(peak_mems)),
            "peak_mem_mib_std": float(np.std(peak_mems)),
        })

    res_df = pd.DataFrame(results)
    os.makedirs(fig_dir, exist_ok=True)

    # 1) update time vs T
    plt.figure(figsize=(7, 4))
    plt.errorbar(res_df["T"], res_df["update_time_sec_mean"], yerr=res_df["update_time_sec_std"], fmt='-o')
    plt.xlabel("Episode length T")
    plt.ylabel("Update time (sec)")
    plt.grid(True, alpha=0.3)
    p1 = os.path.join(fig_dir, f"scaling_update_time_{tag}.png")
    plt.tight_layout()
    plt.savefig(p1, dpi=150)
    plt.close()

    # 2) peak mem vs T
    plt.figure(figsize=(7, 4))
    plt.errorbar(res_df["T"], res_df["peak_mem_mib_mean"], yerr=res_df["peak_mem_mib_std"], fmt='-o')
    plt.xlabel("Episode length T")
    plt.ylabel("Peak GPU memory (MiB)")
    plt.grid(True, alpha=0.3)
    p2 = os.path.join(fig_dir, f"scaling_peak_mem_{tag}.png")
    plt.tight_layout()
    plt.savefig(p2, dpi=150)
    plt.close()

    # CSV 저장
    csv_path = os.path.join(fig_dir, f"scaling_bench_{tag}.csv")
    res_df.to_csv(csv_path, index=False)

    print(f"[SCALING] saved: {os.path.abspath(p1)}")
    print(f"[SCALING] saved: {os.path.abspath(p2)}")
    print(f"[SCALING] csv  : {os.path.abspath(csv_path)}")


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
    now_korea = datetime.now(ZoneInfo("Asia/Seoul"))

    # 2) 환경
    H = 4
    OBS_NOISE = 0.0
    env = VPPBidPOMDPEnv(
        df,
        episode_len=cfg.episode_len,
        lambda_ramp=cfg.lambda_ramp,
        seed=cfg.seed,
        history_len=H,
        obs_noise_std=OBS_NOISE
    )
    A_MAX = env.A_MAX

    # 3) 에이전트
    agent_cfg = AgentConfig(
        obs_dim=env.obs_dim,
        act_dim=1,
        hidden_dim=256,
        lr_actor=cfg.lr,
        lr_critic=cfg.lr,
        gamma=cfg.gamma,
        tau=5e-3,
        batch_size=256,
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
    log_dir = "./Project/Henergy/Proposed_scheme_dev05_(비교그래프추가)/Logs"
    os.makedirs(log_dir, exist_ok=True)
    log_csv = os.path.join(log_dir, f"train_log_transformer_{now_korea.strftime('%Y%m%d_%H%M%S')}.csv")
    logs = []

    fig_dir = "./Project/Henergy/Proposed_scheme_dev05_(비교그래프추가)/Visualization"
    os.makedirs(fig_dir, exist_ok=True)

    # 4) 학습 루프
    for ep in range(1, cfg.episodes + 1):
        ep_t0 = time.perf_counter()  # === [NEW] episode timer start ===
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)  # === [NEW]

        obs = env.reset()
        agent.begin_episode()

        ep_reward_sum = 0.0
        abs_err_sum = 0.0
        step_cnt = 0

        for _ in range(cfg.episode_len):
            act_real = agent.act(obs)
            next_obs, reward, done, info = env.step(act_real)

            act_real_scalar = act_real / A_MAX
            agent.memory.store(obs, act_real_scalar, reward, next_obs)
            agent.step_end(next_obs, done)

            abs_err_sum += abs(float(info["err"]))
            ep_reward_sum += float(reward)
            step_cnt += 1

            obs = next_obs
            if done:
                break

        # === [NEW] update timing ===
        upd_t0 = time.perf_counter()
        loss, ploss, vloss = agent.update()
        upd_t1 = time.perf_counter()
        update_time = upd_t1 - upd_t0

        mae = abs_err_sum / max(1, step_cnt)

        ep_t1 = time.perf_counter()
        ep_time = ep_t1 - ep_t0
        steps_per_sec = step_cnt / max(1e-9, ep_time)

        if device.type == "cuda":
            peak_mem_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        else:
            peak_mem_mib = 0.0

        if ep % 10 == 0 or ep == 1:
            print(
                f"[EP {ep:4d}/{cfg.episodes}] loss={loss:.4f} (π={ploss:.4f}, V={vloss:.4f}) "
                f"R={ep_reward_sum:.4f}  MAE(kW)={mae:.2f}  "
                f"time={ep_time:.3f}s  upd={update_time:.3f}s  "
                f"steps/s={steps_per_sec:.1f}  peak_mem={peak_mem_mib:.0f}MiB"
            )

        logs.append({
            "episode": ep,
            "loss_total": float(loss),
            "loss_phi(policy)": float(ploss),
            "loss_V(value)": float(vloss),
            "R": float(ep_reward_sum),
            "MAE_kW": float(mae),

            # === [NEW] efficiency metrics ===
            "episode_time_sec": float(ep_time),
            "update_time_sec": float(update_time),
            "steps_per_sec": float(steps_per_sec),
            "peak_gpu_mem_mib": float(peak_mem_mib),
        })
        pd.DataFrame(logs).to_csv(log_csv, index=False)

    # === 학습 곡선 + 효율 곡선 그리기 ===
    log_df = pd.DataFrame(logs)
    if len(log_df) > 0:
        fig, axs = plt.subplots(5, 1, figsize=(11, 18), sharex=True)

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
        axs[2].grid(True, alpha=0.3)

        # (4) Throughput
        axs[3].plot(log_df["episode"], log_df["steps_per_sec"])
        axs[3].set_ylabel("Steps/sec")
        axs[3].grid(True, alpha=0.3)

        # (5) Peak GPU memory
        axs[4].plot(log_df["episode"], log_df["peak_gpu_mem_mib"])
        axs[4].set_ylabel("Peak GPU Mem (MiB)")
        axs[4].set_xlabel("Episode")
        axs[4].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(fig_dir, f"train_curves_{now_korea.strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(fig_path, dpi=150)
        print(f"학습 곡선/효율 그래프 저장: {os.path.abspath(fig_path)}")
        plt.close(fig)

    # === [NEW] scaling benchmark (optional) ===
    if cfg.do_scaling_bench:
        T_list = [int(x.strip()) for x in cfg.bench_T_list.split(",") if x.strip()]
        run_scaling_benchmark(
            df=df,
            base_cfg=cfg,
            agent_cfg=agent_cfg,
            A_MAX=A_MAX,
            device=device,
            T_list=T_list,
            bench_episodes=cfg.bench_episodes,
            fig_dir=fig_dir,
            tag=now_korea.strftime('%Y%m%d_%H%M%S'),
        )

    # 모델 저장
    agent.save("vpp_actorcritic_transformer.pt", obs_dim=env.obs_dim)
    print("모델이 저장되었습니다: vpp_actorcritic_transformer.pt")
    print(f"에피소드 로그 CSV: {os.path.abspath(log_csv)}")


if __name__ == "__main__":
    t_start_total = time.perf_counter()
    main()
    t_end_total = time.perf_counter()
    total_elapsed = t_end_total - t_start_total
    print("\n================= Execution Time Summary =================")
    print(f"Total execution time     : {total_elapsed:.2f} sec")
    print("==========================================================\n")
