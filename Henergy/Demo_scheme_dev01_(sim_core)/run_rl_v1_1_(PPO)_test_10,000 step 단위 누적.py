# run_rl_v1_1.py

from __future__ import annotations

from typing import Optional, cast

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from scenarios import (
    BID_PRICE_LOWER_BOUND,
    BID_PRICE_UPPER_BOUND,
    SCENARIO_1_GOV_PLAN,
)
from sim_core_v1_1 import (
    Bid,
    MarketSimulator,
    create_competitor_power_sources,
    prepare_simulation_data,
)
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

#추가 코드 by KH
#===
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from datetime import datetime
from zoneinfo import ZoneInfo
# === [NEW] 그래프 출력용 ===
import matplotlib
matplotlib.use("Agg")  # 서버/터미널 환경에서 그림 저장만 할 때 권장
import matplotlib.pyplot as plt
#===


class KpxBiddingEnv(gym.Env):
    """
    태양·풍력·기타 재생에너지 및 수요/시간 정보를 관측으로 사용하고,
    (입찰전략, 물량비율) 9개 이산 행동으로 입찰가와 입찰량을 동시에 결정하는 환경.
    """

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, scenario: dict):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.scenario = scenario
        self.current_step = 0

        # 정규화를 위한 최대값들
        self._max_demand = float(self.df["forecast_load"].max())
        self._max_hegy = float(self.df["hegy_solar_energy"].max())
        self._max_comp = float(self.df["competitor_solar_energy"].max())
        self._max_wind = float(self.df["wind_energy"].max())
        # 기타 재생에너지 (0일 수도 있으므로 max가 0이면 나눗셈 방지)
        self._max_other = float(self.df["other_renew_energy"].max() or 1.0)

        # 행동: 전략(3) × 비율(3) = 9개
        self.action_space = spaces.Discrete(9)
        self._action_map = {
            0: ("AGGRESSIVE", 1.1),
            1: ("AGGRESSIVE", 1.0),
            2: ("AGGRESSIVE", 0.9),
            3: ("NEUTRAL", 1.1),
            4: ("NEUTRAL", 1.0),
            5: ("NEUTRAL", 0.9),
            6: ("CONSERVATIVE", 1.1),
            7: ("CONSERVATIVE", 1.0),
            8: ("CONSERVATIVE", 0.9),
        }

        # 관측: [sin(hour), cos(hour),
        #        forecast_load,
        #        hegy_solar_energy,
        #        competitor_solar_energy,
        #        wind_energy,
        #        other_renew_energy]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        hour = (pd.to_datetime(row["datetime"]).hour / 24.0) * 2 * np.pi

        # 0 나눗셈 방지용 max 값 보정
        max_demand = self._max_demand if self._max_demand > 0 else 1.0
        max_hegy = self._max_hegy if self._max_hegy > 0 else 1.0
        max_comp = self._max_comp if self._max_comp > 0 else 1.0
        max_wind = self._max_wind if self._max_wind > 0 else 1.0
        max_other = self._max_other if self._max_other > 0 else 1.0

        obs = np.array(
            [
                np.sin(hour),
                np.cos(hour),
                row["forecast_load"] / max_demand,
                row["hegy_solar_energy"] / max_hegy,
                row["competitor_solar_energy"] / max_comp,
                row["wind_energy"] / max_wind,
                row["other_renew_energy"] / max_other,
            ],
            dtype=np.float32,
        )
        # Box 범위에 맞게 클리핑
        return np.clip(obs, -1.0, 1.0)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action: int):
        row = self.df.iloc[self.current_step]
        strat, ratio = self._action_map[int(action)]

        # 전략별 입찰가 결정
        if strat == "AGGRESSIVE":
            price = BID_PRICE_LOWER_BOUND
        elif strat == "NEUTRAL":
            price = (BID_PRICE_LOWER_BOUND + BID_PRICE_UPPER_BOUND) / 2
        else:
            price = BID_PRICE_UPPER_BOUND

        # 물량 비율 적용 (우리 태양광 설비량 기준)
        qty = float(row["hegy_solar_energy"] * ratio)
        my_bid = Bid(
            price=price,
            source_name="에이치에너지_입찰기",
            capacity_kw=qty,
        )

        # 경쟁자 구성 및 시장 시뮬레이션
        competitors = create_competitor_power_sources(row, self.scenario)
        sim = MarketSimulator(competitors)

        demand = float(
            row["forecast_load"] * (1.0 + row["oper_reserve_rate"] / 100.0)
        )
        result = sim.run(demand_kw=demand, my_solar_bid=my_bid)

        # 보상: dispatch_kw × SMP (단순 매출 기준)
        reward = float(result.my_dispatch_kw * result.smp_krw_per_kwh)

        # 다음 스텝으로 진행
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= len(self.df) - 1

        next_obs = (
            self._get_obs()
            if not truncated
            else np.zeros(
                cast(tuple, self.observation_space.shape),
                dtype=np.float32,
            )
        )

        info = {
            "datetime": row["datetime"],
            "demand_kw": demand,
            "my_bid_price": float(price),
            "my_bid_quantity_kw": float(qty),
            "smp_krw_per_kwh": float(getattr(result, "smp_krw_per_kwh", np.nan)),
            "my_dispatch_kw": float(getattr(result, "my_dispatch_kw", np.nan)),
            "bid_strategy": strat,
            "bid_ratio": float(ratio),
        }
        return next_obs, reward, terminated, truncated, info


#추가 코드 by KH
# ==========================
#  A) 학습 중 reward 기록 콜백
# ==========================
class RewardCallback(BaseCallback):
    def __init__(self, window_size: int = 10_000):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = 0.0

        # step 단위 로그
        self.train_smp = []
        self.train_dispatch = []
        self.train_datetime = []

        # === [NEW] 10,000 step 단위 reward 합계 (Train) ===
        self.window_size = int(window_size)
        self._window_sum = 0.0
        self._window_idx = 0
        self.window_rewards = []   # 각 윈도우의 reward 총합
        self.window_steps = []     # (윈도우 끝) 글로벌 step

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        done = bool(self.locals["dones"][0])
        info0 = self.locals["infos"][0] if "infos" in self.locals else {}

        # episode 누적
        self.current_rewards += reward

        # info 기록
        self.train_smp.append(float(info0.get("smp_krw_per_kwh", np.nan)))
        self.train_dispatch.append(float(info0.get("my_dispatch_kw", np.nan)))
        self.train_datetime.append(info0.get("datetime", None))

        # === [NEW] 10,000 step 윈도우 누적 (Train) ===
        self._window_sum += reward
        self._window_idx += 1

        if self._window_idx % self.window_size == 0:
            self.window_rewards.append(self._window_sum)
            self.window_steps.append(int(self.num_timesteps))  # SB3 총 step
            self._window_sum = 0.0

        # episode 종료 처리
        if done:
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0.0

        return True


if __name__ == "__main__":

    # 기존 코드 유지
    df = prepare_simulation_data()
    split = int(len(df) * 0.8)
    train_df, test_df = df[:split], df[split:]

    train_env = DummyVecEnv([lambda: KpxBiddingEnv(train_df, SCENARIO_1_GOV_PLAN)])
    test_env = KpxBiddingEnv(test_df, SCENARIO_1_GOV_PLAN)

    reward_callback = RewardCallback()

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
    )

    # ==========================
    #  B) 학습
    # ==========================
    model.learn(total_timesteps=200_000, progress_bar=True, callback=reward_callback)

    # ==========================
    #  C) 테스트 수행
    # ==========================
    obs, _ = test_env.reset()
    terminated = truncated = False

    rewards = []
    actions = []
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))

        obs, r, terminated, truncated, _ = test_env.step(int(action))
        rewards.append(float(r))  # <- float로 명시

    total_reward = float(np.nansum(rewards))
    print(f"[TEST] steps={len(rewards)}, total_reward={total_reward:,.0f}")

    # === [NEW] Test: 10,000 step 단위 reward 총합 계산 ===
    test_window_size = 10_000
    test_window_rewards = []
    test_window_steps = []

    running_sum = 0.0
    for i, r in enumerate(rewards, start=1):  # i = 1..len(rewards)
        running_sum += r
        if i % test_window_size == 0:
            test_window_rewards.append(running_sum)
            test_window_steps.append(i)  # 윈도우 끝 step
            running_sum = 0.0

    # (선택) 마지막 잔여 구간도 포함하고 싶으면 아래 주석 해제
    if (len(rewards) % test_window_size) != 0:
        test_window_rewards.append(running_sum)
        test_window_steps.append(len(rewards))

    # ==========================
    #  D) 학습 결과 가시화
    # ==========================

    # 저장 경로 설정
    now_korea = datetime.now(ZoneInfo("Asia/Seoul"))
    fig_dir = "./Project/Henergy/Demo_scheme_dev01_(sim_core)/Visualization"
    os.makedirs(fig_dir, exist_ok=True)  # <- 폴더 없으면 생성(안전)

    fig_path1 = os.path.join(fig_dir, f"Training Episode Rewards_{now_korea.strftime('%Y%m%d_%H%M%S')}.png")
    fig_path2 = os.path.join(fig_dir, f"Test Episode - Reward per Step_{now_korea.strftime('%Y%m%d_%H%M%S')}.png")
    fig_path3 = os.path.join(fig_dir, f"Test Episode - Cumulative Reward_{now_korea.strftime('%Y%m%d_%H%M%S')}.png")
    fig_path4 = os.path.join(fig_dir, f"Action Distribution (Test Episode)_{now_korea.strftime('%Y%m%d_%H%M%S')}.png")
    fig_path5 = os.path.join(fig_dir, f"Train - my_dispatch_kw_{now_korea.strftime('%Y%m%d_%H%M%S')}.png")

    # === [NEW] Test 10,000-step reward sum 그래프 저장 경로 ===
    fig_path7 = os.path.join(fig_dir, f"Test - Reward Sum per 10000 Steps_{now_korea.strftime('%Y%m%d_%H%M%S')}.png")

    # 1) 학습 중 episode reward
    plt.figure(figsize=(10, 4))
    plt.plot(reward_callback.episode_rewards)
    plt.title("Training Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(fig_path1, dpi=150)

    # 2) 테스트 reward 시계열
    plt.figure(figsize=(10, 4))
    plt.plot(rewards)
    plt.title("Test Episode - Reward per Step")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(fig_path2, dpi=150)

    # 3) 누적 reward 곡선
    cumulative = np.cumsum(rewards)
    plt.figure(figsize=(10, 4))
    plt.plot(cumulative)
    plt.title("Test Episode - Cumulative Reward")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.savefig(fig_path3, dpi=150)

    # 4) 행동(action) 선택 분포 히스토그램
    plt.figure(figsize=(10, 4))
    plt.hist(actions, bins=9, range=(0, 9), edgecolor="black")
    plt.title("Action Distribution (Test Episode)")
    plt.xlabel("Action ID (0~8)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(fig_path4, dpi=150)

    # (옵션) train dispatch 샘플링 플롯을 원하면 아래처럼 추가 가능
    # stride = 10
    # dispatch_series = np.array(reward_callback.train_dispatch, dtype=float)[::stride]
    # plt.figure(figsize=(12, 4))
    # plt.plot(dispatch_series)
    # plt.title("Training - my_dispatch_kw (sampled)")
    # plt.xlabel(f"Step (every {stride} steps)")
    # plt.ylabel("my_dispatch_kw")
    # plt.grid(True)
    # plt.savefig(fig_path5, dpi=150)

    # === [NEW] 7) 테스트: 10,000 step 단위 reward 총합 그래프 ===
    plt.figure(figsize=(12, 4))
    plt.plot(test_window_steps, test_window_rewards)
    plt.title("Test - Reward Sum per 10,000 Steps")
    plt.xlabel("Step (end of window)")
    plt.ylabel("Reward Sum (10,000 steps)")
    plt.grid(True)
    plt.savefig(fig_path7, dpi=150)

    print("학습/테스트 그래프 저장 완료")
