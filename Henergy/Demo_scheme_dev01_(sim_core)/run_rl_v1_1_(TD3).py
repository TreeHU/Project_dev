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

# ====== 변경: PPO -> TD3 ======
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise

#추가 코드 by KH
#===
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from datetime import datetime
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#===


class KpxBiddingEnv(gym.Env):
    """
    관측: 재생에너지/수요/시간 정보
    행동(TD3용 연속형):
      a[0] in [-1,1] -> 입찰가 [LOWER, UPPER]
      a[1] in [-1,1] -> 물량비율 [0.9, 1.1]
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
        self._max_other = float(self.df["other_renew_energy"].max() or 1.0)

        # ====== 변경: Discrete(9) -> Box(2,) ======
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # 관측: [sin(hour), cos(hour), forecast_load, hegy_solar, comp_solar, wind, other]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

        # ratio 범위(기존 0.9/1.0/1.1에서 연속으로 확장)
        self._ratio_min = 0.9
        self._ratio_max = 1.1

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        hour = (pd.to_datetime(row["datetime"]).hour / 24.0) * 2 * np.pi

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
        return np.clip(obs, -1.0, 1.0)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    # ====== 변경: 연속 행동 -> price/ratio로 스케일링 ======
    def _scale_action_to_price_ratio(self, action: np.ndarray) -> tuple[float, float]:
        a = np.clip(np.array(action, dtype=np.float32), -1.0, 1.0)

        # [-1,1] -> [LOWER, UPPER]
        price = float(
            BID_PRICE_LOWER_BOUND
            + (a[0] + 1.0) * 0.5 * (BID_PRICE_UPPER_BOUND - BID_PRICE_LOWER_BOUND)
        )

        # [-1,1] -> [ratio_min, ratio_max]
        ratio = float(self._ratio_min + (a[1] + 1.0) * 0.5 * (self._ratio_max - self._ratio_min))

        return price, ratio

    def step(self, action):
        row = self.df.iloc[self.current_step]

        price, ratio = self._scale_action_to_price_ratio(action)

        qty = float(row["hegy_solar_energy"] * ratio)
        my_bid = Bid(
            price=price,
            source_name="에이치에너지_입찰기",
            capacity_kw=qty,
        )

        competitors = create_competitor_power_sources(row, self.scenario)
        sim = MarketSimulator(competitors)

        demand = float(row["forecast_load"] * (1.0 + row["oper_reserve_rate"] / 100.0))
        result = sim.run(demand_kw=demand, my_solar_bid=my_bid)

        reward = float(result.my_dispatch_kw * result.smp_krw_per_kwh)

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= len(self.df) - 1

        next_obs = (
            self._get_obs()
            if not truncated
            else np.zeros(cast(tuple, self.observation_space.shape), dtype=np.float32)
        )

        info = {
            "datetime": row["datetime"],
            "demand_kw": demand,
            "my_bid_price": float(price),
            "my_bid_quantity_kw": float(qty),
            "smp_krw_per_kwh": float(getattr(result, "smp_krw_per_kwh", np.nan)),
            "my_dispatch_kw": float(getattr(result, "my_dispatch_kw", np.nan)),
            "bid_ratio": float(ratio),
            "raw_action": np.array(action, dtype=np.float32),
        }
        return next_obs, reward, terminated, truncated, info


# ==========================
#  A) 학습 중 reward 기록 콜백
# ==========================
class RewardCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.current_rewards = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        done = self.locals["dones"][0]

        self.current_rewards += reward

        if done:
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0.0

        return True


if __name__ == "__main__":

    df = prepare_simulation_data()
    split = int(len(df) * 0.8)
    train_df, test_df = df[:split], df[split:]

    train_env = DummyVecEnv([lambda: KpxBiddingEnv(train_df, SCENARIO_1_GOV_PLAN)])
    test_env = KpxBiddingEnv(test_df, SCENARIO_1_GOV_PLAN)

    reward_callback = RewardCallback()

    # ====== TD3 권장: action noise ======
    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # ====== 변경: PPO -> TD3 ======
    model = TD3(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=256,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
    )

    model.learn(total_timesteps=1_000_000, progress_bar=True, callback=reward_callback)

    # ==========================
    #  C) 테스트 수행
    # ==========================
    obs, _ = test_env.reset()
    terminated = truncated = False

    rewards = []
    actions_price = []
    actions_ratio = []

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)   # action shape (2,)
        obs, r, terminated, truncated, info = test_env.step(action)

        rewards.append(r)
        actions_price.append(info["my_bid_price"])
        actions_ratio.append(info["bid_ratio"])

    total_reward = float(np.nansum(rewards))
    print(f"[TEST] steps={len(rewards)}, total_reward={total_reward:,.0f}")

    # ==========================
    #  D) 학습 결과 가시화
    # ==========================
    now_korea = datetime.now(ZoneInfo("Asia/Seoul"))
    fig_dir = "./Project/Henergy/Demo_scheme_dev01_(sim_core)/Visualization"
    os.makedirs(fig_dir, exist_ok=True)

    fig_path1 = os.path.join(fig_dir, f"Training Episode Rewards_{now_korea.strftime('%Y%m%d_%H%M%S')}.png")
    fig_path2 = os.path.join(fig_dir, f"Test Episode - Reward per Step_{now_korea.strftime('%Y%m%d_%H%M%S')}.png")
    fig_path3 = os.path.join(fig_dir, f"Test Episode - Cumulative Reward_{now_korea.strftime('%Y%m%d_%H%M%S')}.png")
    fig_path4 = os.path.join(fig_dir, f"Test Episode - Bid Price Distribution_{now_korea.strftime('%Y%m%d_%H%M%S')}.png")
    fig_path5 = os.path.join(fig_dir, f"Test Episode - Bid Ratio Distribution_{now_korea.strftime('%Y%m%d_%H%M%S')}.png")

    # 1) 학습 중 episode reward
    plt.figure(figsize=(10,4))
    plt.plot(reward_callback.episode_rewards)
    plt.title("Training Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(fig_path1, dpi=150)

    # 2) 테스트 reward 시계열
    plt.figure(figsize=(10,4))
    plt.plot(rewards)
    plt.title("Test Episode - Reward per Step")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(fig_path2, dpi=150)

    # 3) 누적 reward 곡선
    cumulative = np.cumsum(rewards)
    plt.figure(figsize=(10,4))
    plt.plot(cumulative)
    plt.title("Test Episode - Cumulative Reward")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.savefig(fig_path3, dpi=150)

    # 4) 테스트 입찰가 분포
    plt.figure(figsize=(10,4))
    plt.hist(actions_price, bins=30, edgecolor="black")
    plt.title("Test Episode - Bid Price Distribution")
    plt.xlabel("Bid Price")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(fig_path4, dpi=150)

    # 5) 테스트 비율 분포
    plt.figure(figsize=(10,4))
    plt.hist(actions_ratio, bins=30, edgecolor="black")
    plt.title("Test Episode - Bid Ratio Distribution")
    plt.xlabel("Bid Ratio")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(fig_path5, dpi=150)

    print("학습/테스트 그래프 저장 완료")
