# run_rl_v1_1.py (PPO 직접 구현 버전)

from __future__ import annotations
from typing import Optional, cast, Dict, Any, List, Tuple

import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

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

# ---- plotting (save only) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime
from zoneinfo import ZoneInfo

# ---- PyTorch ----
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# =========================================================
# 1) Environment (동일)
# =========================================================
class KpxBiddingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, scenario: dict):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.scenario = scenario
        self.current_step = 0

        self._max_demand = float(self.df["forecast_load"].max() or 1.0)
        self._max_hegy = float(self.df["hegy_solar_energy"].max() or 1.0)
        self._max_comp = float(self.df["competitor_solar_energy"].max() or 1.0)
        self._max_wind = float(self.df["wind_energy"].max() or 1.0)
        self._max_other = float(self.df["other_renew_energy"].max() or 1.0)

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

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.current_step]
        hour = (pd.to_datetime(row["datetime"]).hour / 24.0) * 2 * np.pi

        obs = np.array(
            [
                np.sin(hour),
                np.cos(hour),
                row["forecast_load"] / (self._max_demand if self._max_demand > 0 else 1.0),
                row["hegy_solar_energy"] / (self._max_hegy if self._max_hegy > 0 else 1.0),
                row["competitor_solar_energy"] / (self._max_comp if self._max_comp > 0 else 1.0),
                row["wind_energy"] / (self._max_wind if self._max_wind > 0 else 1.0),
                row["other_renew_energy"] / (self._max_other if self._max_other > 0 else 1.0),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action: int):
        row = self.df.iloc[self.current_step]
        strat, ratio = self._action_map[int(action)]

        if strat == "AGGRESSIVE":
            price = BID_PRICE_LOWER_BOUND
        elif strat == "NEUTRAL":
            price = (BID_PRICE_LOWER_BOUND + BID_PRICE_UPPER_BOUND) / 2
        else:
            price = BID_PRICE_UPPER_BOUND

        qty = float(row["hegy_solar_energy"] * ratio)
        my_bid = Bid(price=price, source_name="에이치에너지_입찰기", capacity_kw=qty)

        competitors = create_competitor_power_sources(row, self.scenario)
        sim = MarketSimulator(competitors)

        demand = float(row["forecast_load"] * (1.0 + row["oper_reserve_rate"] / 100.0))
        result = sim.run(demand_kw=demand, my_solar_bid=my_bid)

        reward = float(result.my_dispatch_kw * result.smp_krw_per_kwh)

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= len(self.df) - 1

        next_obs = self._get_obs() if not truncated else np.zeros(cast(tuple, self.observation_space.shape), dtype=np.float32)

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


# =========================================================
# 2) PPO 네트워크 (Actor-Critic)
# =========================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)     # logits
        self.v = nn.Linear(hidden, 1)              # value

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value

    def get_action(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value, dist.entropy()

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy()
        return logp, entropy, value


# =========================================================
# 3) Rollout Buffer + GAE
# =========================================================
@torch.no_grad()
def compute_gae(rewards, dones, values, next_value, gamma=0.99, lam=0.95):
    """
    rewards: (T,)
    dones: (T,)  True if episode ended at step t
    values: (T,)
    next_value: scalar
    returns, advantages: (T,)
    """
    T = len(rewards)
    adv = torch.zeros(T, dtype=torch.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t].float()
        next_v = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return returns, adv


# =========================================================
# 4) PPO Agent (직접 구현)
# =========================================================
class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.device = torch.device(device)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.net = ActorCritic(obs_dim, n_actions).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=learning_rate)

        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.gamma = gamma
        self.lam = gae_lambda
        self.clip = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # logging buffers (학습 중 시각화용)
        self.train_episode_rewards: List[float] = []
        self._ep_reward_acc = 0.0
        self.train_smp: List[float] = []
        self.train_dispatch: List[float] = []

    @torch.no_grad()
    def predict(self, obs_np: np.ndarray, deterministic: bool = True) -> int:
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, _ = self.net(obs)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
        return int(action.item())

    def learn(self, env: gym.Env, total_timesteps: int, progress_bar: bool = True):
        obs, _ = env.reset()
        timesteps = 0

        while timesteps < total_timesteps:
            # ---- collect rollout (n_steps) ----
            obs_buf = []
            act_buf = []
            logp_buf = []
            rew_buf = []
            done_buf = []
            val_buf = []

            for _ in range(self.n_steps):
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                action_t, logp_t, value_t, _ = self.net.get_action(obs_t)

                action = int(action_t.item())
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

                # logging (학습 중 dispatch/smp)
                self.train_smp.append(float(info.get("smp_krw_per_kwh", np.nan)))
                self.train_dispatch.append(float(info.get("my_dispatch_kw", np.nan)))

                # episode reward logging
                self._ep_reward_acc += float(reward)
                if done:
                    self.train_episode_rewards.append(self._ep_reward_acc)
                    self._ep_reward_acc = 0.0
                    next_obs, _ = env.reset()

                obs_buf.append(obs)
                act_buf.append(action)
                logp_buf.append(float(logp_t.item()))
                rew_buf.append(float(reward))
                done_buf.append(done)
                val_buf.append(float(value_t.item()))

                obs = next_obs
                timesteps += 1
                if timesteps >= total_timesteps:
                    break

            # bootstrap value
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                _, next_v = self.net(obs_t)

            # tensors
            obs_tensor = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=self.device)
            act_tensor = torch.tensor(np.array(act_buf), dtype=torch.int64, device=self.device)
            old_logp = torch.tensor(np.array(logp_buf), dtype=torch.float32, device=self.device)
            rewards = torch.tensor(np.array(rew_buf), dtype=torch.float32, device=self.device)
            dones = torch.tensor(np.array(done_buf), dtype=torch.bool, device=self.device)
            values = torch.tensor(np.array(val_buf), dtype=torch.float32, device=self.device)

            returns, adv = compute_gae(
                rewards=rewards,
                dones=dones,
                values=values,
                next_value=next_v.squeeze(0),
                gamma=self.gamma,
                lam=self.lam,
            )

            # normalize advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            # ---- PPO update ----
            n = obs_tensor.shape[0]
            idx = np.arange(n)

            for _epoch in range(self.n_epochs):
                np.random.shuffle(idx)
                for start in range(0, n, self.batch_size):
                    mb_idx = idx[start : start + self.batch_size]
                    mb_obs = obs_tensor[mb_idx]
                    mb_act = act_tensor[mb_idx]
                    mb_old_logp = old_logp[mb_idx]
                    mb_adv = adv[mb_idx]
                    mb_ret = returns[mb_idx]

                    new_logp, entropy, v_pred = self.net.evaluate_actions(mb_obs, mb_act)

                    ratio = torch.exp(new_logp - mb_old_logp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = 0.5 * (mb_ret - v_pred).pow(2).mean()
                    entropy_loss = -entropy.mean()

                    loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                    self.opt.step()

            if progress_bar:
                # 아주 단순 진행 출력
                print(f"[TRAIN] timesteps={timesteps}/{total_timesteps}, episodes={len(self.train_episode_rewards)}")

        return self


# =========================================================
# 5) main: 학습 + 테스트 + 가시화
# =========================================================
if __name__ == "__main__":

    # 1) CSV -> dataframe
    df = prepare_simulation_data()
    split = int(len(df) * 0.8)
    train_df, test_df = df[:split], df[split:]

    # 2) env
    train_env = KpxBiddingEnv(train_df, SCENARIO_1_GOV_PLAN)
    test_env = KpxBiddingEnv(test_df, SCENARIO_1_GOV_PLAN)

    # 3) PPO (직접 구현)
    agent = PPOAgent(
        obs_dim=7,
        n_actions=9,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        device="cpu",  # CUDA 가능하면 "cuda"
        seed=0,
    )

    # 4) 학습
    total_timesteps = 2_000_000
    agent.learn(train_env, total_timesteps=total_timesteps, progress_bar=True)

    # 5) 테스트
    obs, _ = test_env.reset()
    terminated = truncated = False
    rewards = []
    actions = []
    test_dispatch = []
    test_smp = []

    while not (terminated or truncated):
        action = agent.predict(obs, deterministic=True)
        actions.append(int(action))
        obs, r, terminated, truncated, info = test_env.step(int(action))
        rewards.append(float(r))
        test_dispatch.append(float(info.get("my_dispatch_kw", np.nan)))
        test_smp.append(float(info.get("smp_krw_per_kwh", np.nan)))

    total_reward = float(np.nansum(rewards))
    print(f"[TEST] steps={len(rewards)}, total_reward={total_reward:,.0f}")

    # 6) 가시화 저장
    now_korea = datetime.now(ZoneInfo("Asia/Seoul"))
    fig_dir = "./Project/Henergy/Demo_scheme_dev01_(sim_core)/Visualization"
    os.makedirs(fig_dir, exist_ok=True)

    ts = now_korea.strftime("%Y%m%d_%H%M%S")
    fig_path1 = os.path.join(fig_dir, f"Training Episode Rewards_{ts}.png")
    fig_path2 = os.path.join(fig_dir, f"Train - my_dispatch_kw_{ts}.png")
    fig_path3 = os.path.join(fig_dir, f"Train - smp_krw_per_kwh_{ts}.png")
    fig_path4 = os.path.join(fig_dir, f"Test Episode - Reward per Step_{ts}.png")
    fig_path5 = os.path.join(fig_dir, f"Test Episode - Cumulative Reward_{ts}.png")
    fig_path6 = os.path.join(fig_dir, f"Action Distribution (Test Episode)_{ts}.png")
    fig_path7 = os.path.join(fig_dir, f"Test - dispatch & smp_{ts}.png")

    # training episode reward
    plt.figure(figsize=(10, 4))
    plt.plot(agent.train_episode_rewards)
    plt.title("Training Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(fig_path1, dpi=150)

    # train dispatch/smp (sampled)
    stride = 10
    train_dispatch = np.array(agent.train_dispatch, dtype=float)[::stride]
    train_smp = np.array(agent.train_smp, dtype=float)[::stride]

    plt.figure(figsize=(12, 4))
    plt.plot(train_dispatch)
    plt.title(f"Training - my_dispatch_kw (every {stride} steps)")
    plt.xlabel("Index")
    plt.ylabel("my_dispatch_kw")
    plt.grid(True)
    plt.savefig(fig_path2, dpi=150)

    plt.figure(figsize=(12, 4))
    plt.plot(train_smp)
    plt.title(f"Training - smp_krw_per_kwh (every {stride} steps)")
    plt.xlabel("Index")
    plt.ylabel("smp_krw_per_kwh")
    plt.grid(True)
    plt.savefig(fig_path3, dpi=150)

    # test reward
    plt.figure(figsize=(10, 4))
    plt.plot(rewards)
    plt.title("Test Episode - Reward per Step")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(fig_path4, dpi=150)

    # cumulative reward
    plt.figure(figsize=(10, 4))
    plt.plot(np.cumsum(rewards))
    plt.title("Test Episode - Cumulative Reward")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.grid(True)
    plt.savefig(fig_path5, dpi=150)

    # action distribution
    plt.figure(figsize=(10, 4))
    plt.hist(actions, bins=9, range=(0, 9), edgecolor="black")
    plt.title("Action Distribution (Test Episode)")
    plt.xlabel("Action ID (0~8)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(fig_path6, dpi=150)

    # test dispatch & smp (two lines)
    plt.figure(figsize=(12, 4))
    plt.plot(test_dispatch, label="my_dispatch_kw")
    plt.plot(test_smp, label="smp_krw_per_kwh")
    plt.title("Test - Dispatch and SMP")
    plt.xlabel("Step")
    plt.grid(True)
    plt.legend()
    plt.savefig(fig_path7, dpi=150)

    print("[OK] PPO 직접 구현 학습/테스트/가시화 저장 완료")
