"""
VPP Day-Ahead/Intraday 15-min Bidding with LSTM Actor-Critic (Model-Free)
- Environment: 모사 전력 시장 (데이터 기반)
- Agent: HENERGY 社 VPP (Actor-Critic, LSTM)
- State: LSTM Latent (입력: 과거 PV, 임의 생성 기상, 시간 임베딩)
- Transition: Model-Free (다음 시점 데이터로 전이, 수학적 전이모델 無)
- Action: 15분 단위 입찰 발전량 (연속, 0 ~ A_MAX)
- Reward: 실제 발전량과 출력 발전량(입찰량) 차이의 음의 제곱 + 램프 패널티

실행 예:
    python vpp_actorcritic.py --excel_path ./pv_15min.xlsx --sheet_name Sheet1 --episodes 100
"""

import os
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Union

def resample_impute(df: pd.DataFrame, ts_col: str, freq_minutes: int) -> pd.DataFrame:

    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col])
    out = out.sort_values(ts_col).set_index(ts_col)
    rule = f"{freq_minutes}T"
    out = out.resample(rule).mean()
    out = out.interpolate(limit_direction="both")
    out = out.reset_index()
    return out

def _find_col(candidates, columns):
    cols_lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name.lower() in cols_lower:
            return cols_lower[name.lower()]
    return None

def load_pv_15min(excel_path: Optional[str], sheet_name: Optional[str], freq_minutes: int = 15) -> pd.DataFrame:
    """
    15분 PV 총발전량 시계열을 로드. 파일이 없거나 형식이 맞지 않으면 모의 데이터를 생성.
    반환 컬럼: ['timestamp', 'pv_total']
    """
    if excel_path and os.path.exists(excel_path):
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            ts_col = _find_col(["timestamp", "time", "date", "datetime"], df.columns)
            pv_col = _find_col(["pv_total", "sum_energy", "pv", "generation", "energy"], df.columns) 
            if ts_col is None or pv_col is None:
                raise ValueError("엑셀에 timestamp/pv_total 열을 찾을 수 없습니다.")
            df = df[[ts_col, pv_col]].rename(columns={ts_col: "timestamp", pv_col: "pv_total"})
            df = resample_impute(df, "timestamp", freq_minutes)
            return df[["timestamp", "pv_total"]]
        except Exception as e:
            print(f"엑셀 로드 실패: {e} → 모의 PV 데이터 생성으로 대체합니다. ")
    print("모의 pv 15분 데이터를 생성합니다.")
    periods = 96 * 30
    idx = pd.date_range("2025-07-09", periods=periods, freq=f"{freq_minutes}T")

    # 일중 일사량 곡선(사인 반주기) + 랜덤 구름(베르누이 AR-like)
    t = np.arange(len(idx))
    tod = (idx.hour * 60 + idx.minute) / (24 * 60)  # [0,1)
    irradiance = np.clip(np.sin(np.pi * tod), 0, 1)  # 밤 0, 정오 1
    clouds = np.clip(np.random.beta(2, 5, len(idx)) * 0.7, 0, 1)  # 0(쾌청)~1(흐림)
    pv = 3000 * irradiance * (1 - 0.6 * clouds)  # kW급 임의 스케일
    pv += np.random.normal(0, 40, len(idx))       # 측정 노이즈
    pv = np.clip(pv, 0, None)

    return pd.DataFrame({"timestamp": idx, "pv_total": pv})


def make_synthetic_weather(index: Union[pd.Series, pd.DatetimeIndex, np.ndarray, list]) -> pd.DataFrame:
    """
    15분 간격 임의 기상 생성.
    어떤 형태로 들어와도 DatetimeIndex 로 변환해 안전하게 .hour/.minute 사용.
    """
    idx = pd.DatetimeIndex(index)  # 핵심: 내부에서 강제 캐스팅
    n = len(idx)

    # 일중/연중 주기
    tod = (idx.hour * 60 + idx.minute) / (24 * 60)  # [0,1)
    doy = idx.dayofyear                               # 1..365(또는 366)

    # 일사량 유사(낮에만 양)
    ghi = np.clip(np.sin(np.pi * tod), 0, 1)

    # 운량 AR(1) 느낌의 랜덤
    cloud = np.zeros(n)
    for i in range(n):
        eps = np.random.normal(0, 0.08)
        prev = cloud[i-1] if i else 0.3
        cloud[i] = np.clip(0.85 * prev + 0.15 * np.random.rand() + eps, 0, 1)

    # 기온: 연중/일중 변동 + 노이즈
    temp = (12
            + 10 * np.sin(2 * np.pi * (doy / 365.0))
            + 7 * np.sin(2 * np.pi * tod)
            + np.random.normal(0, 1.2, n))

    # 풍속: 양수, 밤에 조금 강(임의)
    wind = np.clip(2.0 + 2.0 * (1 - ghi) + np.random.normal(0, 0.6, n), 0, None)

    return pd.DataFrame({
        "ghi_sim": ghi.astype(np.float32),
        "temp_sim": temp.astype(np.float32),
        "wind_sim": wind.astype(np.float32),
        "cloud_sim": cloud.astype(np.float32),
    })

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    tod = (ts.dt.hour * 60 + ts.dt.minute) / (24 * 60)
    doy = ts.dt.dayofyear / 365.0
    df["tod_sin"] = np.sin(2 * np.pi * tod)
    df["tod_cos"] = np.cos(2 * np.pi * tod)
    df["doy_sin"] = np.sin(2 * np.pi * doy)
    df["doy_cos"] = np.cos(2 * np.pi * doy)
    return df

class VPPBidEnv:
    """
    Model-Free 데이터 환경.
    - step 마다 t→t+1로 진행하며, 실제 pv_total[t]를 기반으로 보상을 계산
    - 관측: [pv_prev_norm, ghi, temp_norm, wind_norm, cloud, time-embeds]
    - 행동: 15분 입찰량(출력발전) [0, A_MAX] (연속)
    """
    def __init__(self, df: pd.DataFrame, episode_len: int = 96, lambda_ramp: float = 0.01, seed: int = 0):
        assert "pv_total" in df.columns
        self.df = df.reset_index(drop=True).copy()
        self.episode_len = episode_len
        self.lambda_ramp = lambda_ramp
        self.rng = np.random.RandomState(seed)

        self.A_MAX = float(np.percentile(self.df["pv_total"], 99))  # 이상치 완화 상한
        self.pv_std = float(self.df["pv_total"].std() + 1e-6)
        self.temp_std = float(self.df["temp_sim"].std() + 1e-6)
        self.wind_std = float(self.df["wind_sim"].std() + 1e-6)

        self.t0 = 0
        self.ptr = 0
        self.prev_action = 0.0

    def _get_obs(self, idx: int) -> np.ndarray:
        row = self.df.iloc[idx]
        pv_prev = self.df.iloc[max(idx - 1, 0)]["pv_total"]
        obs = np.array([
            pv_prev / self.A_MAX,            # pv_prev_norm
            row["ghi_sim"],                  # [0,1]
            (row["temp_sim"] / (self.temp_std * 4)),  # 약한 정규화
            (row["wind_sim"] / (self.wind_std * 4)),
            row["cloud_sim"],
            row["tod_sin"], row["tod_cos"],
            row["doy_sin"], row["doy_cos"],
        ], dtype=np.float32)
        return obs
    
    @property
    def obs_dim(self):
        return 9
    
    def reset(self) -> np.ndarray:
        max_start = len(self.df) - self.episode_len - 2
        self.t0 = int(self.rng.randint(1, max(2, max_start)))
        self.ptr = 0
        self.prev_action = 0.0
        return self._get_obs(self.t0)
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        """
        action: [0, A_MAX] 범위의 실수 
        """
        t_idx = self.t0 + self.ptr
        real = float(self.df.iloc[t_idx]["pv_total"])
        act = float(np.clip(action, 0.0, self.A_MAX))

        scale = (self.A_MAX ** 2)
        err = (act - real)
        ramp = (act - self.prev_action)
        reward = -(err * err) / scale - self.lambda_ramp * (ramp * ramp) / scale

        self.prev_action = act
        self.ptr += 1
        done = self.ptr >= self.episode_len
        next_obs = self._get_obs(self.t0 + self.ptr)
        info = {"real": real, "action": act, "err": err}
        return next_obs, float(reward), bool(done), info
     

class BetaPolicyHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ab = self.fc(z)
        alpha_raw, beta_raw = ab[..., 0], ab[..., 1]
        alpha = self.softplus(alpha_raw) + 1.0
        beta = self.softplus(beta_raw) + 1.0
        return alpha, beta
    
class RecurrentActorCritic(nn.Module):
    """
    공유 인코더 → LSTM → (정책 Beta-Head, 가치 V-Head)
    - 입력: obs_t (배치 x obs_dim)
    - LSTM이 Latent Vector를 생성(요구사항의 'State: Latent Vector')
    """
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, batch_first=True)
        self.policy = BetaPolicyHead(hidden_dim)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def init_state(self, batch_size: int = 1):
        h = torch.zeros(1, batch_size, self.lstm.hidden_size)
        c = torch.zeros(1, batch_size, self.lstm.hidden_size)
        return (h, c)

    def forward(self, obs: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        # obs: (B, 1, obs_dim)
        feat = self.encoder(obs.squeeze(1))          # (B, 128)
        lstm_in = feat.unsqueeze(1)                  # (B, 1, 128)
        z, new_state = self.lstm(lstm_in, state)     # z: (B, 1, H)
        z = z.squeeze(1)                             # (B, H)
        alpha, beta = self.policy(z)
        value = self.value(z).squeeze(-1)            # (B,)
        return alpha, beta, value, new_state


def beta_log_prob(alpha: torch.Tensor, beta: torch.Tensor, a01: torch.Tensor):
    # a01 in (0,1); Beta PDF 로그
    from torch.distributions import Beta
    dist = Beta(alpha, beta)
    lp = dist.log_prob(torch.clamp(a01, 1e-6, 1 - 1e-6))
    return lp, dist

@dataclass
class Config:
    excel_path: Optional[str] = None
    sheet_name: Optional[str] = None
    episodes: int = 100
    episode_len: int = 96
    lambda_ramp: float = 0.01
    gamma: float = 0.99
    lr: float = 3e-4
    entropy_coef: float = 0.001
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    seed: int = 7

def train(cfg: Config):
    # 데이터 준비
    pv_df = load_pv_15min(cfg.excel_path, cfg.sheet_name, 15)
    wx = make_synthetic_weather(pd.to_datetime(pv_df["timestamp"]))
    df = pd.concat([pv_df.reset_index(drop=True), wx.reset_index(drop=True)], axis=1)
    df = add_time_features(df)

    env = VPPBidEnv(df, episode_len=cfg.episode_len, lambda_ramp=cfg.lambda_ramp, seed=cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RecurrentActorCritic(obs_dim=env.obs_dim, hidden_dim=128).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    A_MAX = env.A_MAX

    for ep in range(1, cfg.episodes + 1):
        obs = env.reset()
        h, c = model.init_state(batch_size=1)
        h, c = h.to(device), c.to(device)

        logps, values, rewards, entropies = [], [], [], []
        actions01, actions_real = [], []

        for t in range(cfg.episode_len):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).view(1, 1, -1)
            alpha, beta, value, (h, c) = model(obs_t, (h, c))

            # 샘플링: Beta in (0,1), 실액션 = a01 * A_MAX
            a01 = torch.distributions.Beta(alpha, beta).rsample()
            logp, dist = beta_log_prob(alpha, beta, a01)
            entropy = dist.entropy()

            act_real = (a01.squeeze().item()) * A_MAX
            next_obs, reward, done, info = env.step(act_real)

            # 저장
            logps.append(logp.squeeze())
            entropies.append(entropy.squeeze())
            values.append(value.squeeze())
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
            actions01.append(a01.squeeze())
            actions_real.append(act_real)

            obs = next_obs
            if done:
                break

        # Returns / Advantage
        with torch.no_grad():
            G = torch.zeros(1, device=device)
        returns = []
        for r in reversed(rewards):
            G = r + cfg.gamma * G
            returns.append(G)
        returns = list(reversed(returns))
        returns = torch.stack(returns)
        values_t = torch.stack(values)

        advantages = returns - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        # 손실
        logps_t = torch.stack(logps)
        entropies_t = torch.stack(entropies)

        policy_loss = -(advantages.detach() * logps_t).mean() - cfg.entropy_coef * entropies_t.mean()
        value_loss = cfg.value_coef * (returns.detach() - values_t).pow(2).mean()
        loss = policy_loss + value_loss

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        opt.step()

        # 로그
        ep_reward = sum([r.item() for r in rewards])
        if ep % max(1, cfg.episodes // 20) == 0 or ep == 1:
            mae = float(np.mean([abs(a - info.get("real", 0.0)) for a in actions_real])) if actions_real else float("nan")
            print(f"[EP {ep:4d}/{cfg.episodes}] loss={loss.item():.4f}  R={ep_reward:.4f}  "
                  f"MAE(kW)≈{mae:.2f}  A_MAX={A_MAX:.1f}")

    # 최종 저장
    torch.save({"model_state_dict": model.state_dict(),
                "A_MAX": A_MAX,
                "obs_dim": env.obs_dim}, "vpp_actorcritic_lstm.pt")
    print("모델이 저장되었습니다: vpp_actorcritic_lstm.pt")



# ------------------------------
# 진입점
# ------------------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--excel_path", type=str, default=None, help="./Project/EM/Data/Excel/sum_energy_15min.xlsx")
    p.add_argument("--sheet_name", type=str, default=None, help="Sheet1")
    p.add_argument("--episodes", type=int, default=500000, help="학습 에피소드 수 (요구사항 4: 기본 100)")
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


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)