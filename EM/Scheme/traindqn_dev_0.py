# ============================================================
# VPP Strategic Bidding (End-to-End) — PPO with Two-Stream FE
# - 1년치 15분 단위 임의 데이터 생성
# - 두 스트림 피처 추출(CNN/Residual + LSTM)
# - 다구간 입찰(가격,수량) 3쌍 연속 액션
# - 보상: gamma_clear * q_offer - cost_rate * q_offer
# - 전이: 모델프리(다음 시점 실제 데이터로 상태 갱신)
# ============================================================

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass

# --------------------------
# 0) 전역 설정
# --------------------------
SEED = 7
np.random.seed(SEED)
torch.manual_seed(SEED)

# 시간/데이터 스펙
MINUTES = 15
STEPS_PER_DAY = (24 * 60) // MINUTES  # 96
DAYS = 365
TOTAL_STEPS = DAYS * STEPS_PER_DAY    # 35,040 (윤년 미고려)
LOOKBACK = 96                          # 상태 입력용 과거 시퀀스 길이(1일)
EPISODE_LEN = 96                       # 1 에피소드 길이(기본 1일)
N_EPISODES = 100                       
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 시장/설비 파라미터
Q_MAX = 14.0                           # 최대 발전가능(kW) — 논문 예시 규모
PRICE_MIN, PRICE_MAX = -50.0, 200.0    # clearing/입찰 가격 범위(가상)
COST_RATE = 10.0                       # 단위 전력 당 변동비(가상, $/kWh)

# PPO 파라미터(간단 버전)
GAMMA = 0.99
LR = 3e-4
PPO_EPOCHS = 8
BATCH_SIZE = 2048
CLIP_EPS = 0.2
ENTROPY_COEF = 0.00
VF_COEF = 0.5

# -----------------------------------------------------------
# 1) 1년치 Synthetic 데이터 생성기
# -----------------------------------------------------------
@dataclass
class MarketData:
    price: np.ndarray       # (T,)
    irr: np.ndarray         # 일사량 proxy (0~1000)
    wind: np.ndarray        # 풍속 proxy (m/s)
    temp: np.ndarray        # 기온 proxy (°C)
    gen_potential: np.ndarray  # 재생에너지 가용 발전량(0~Q_MAX)
    cost_rate: float

def make_synthetic_market(seed=SEED) -> MarketData:
    rng = np.random.default_rng(seed)
    t = np.arange(TOTAL_STEPS)

    # 가격 시계열(일/주기+잡음+드문 음수가격 스파이크)
    daily = np.sin(2*np.pi*(t % STEPS_PER_DAY)/STEPS_PER_DAY - np.pi/3)
    weekly = np.sin(2*np.pi*t/(STEPS_PER_DAY*7))
    price = 90 + 25*daily + 12*weekly + rng.normal(0, 8, size=TOTAL_STEPS)
    spikes = rng.binomial(1, 0.01, TOTAL_STEPS)  # 1% 확률로 음수 스파이크
    price = price - spikes * rng.uniform(30, 80, size=TOTAL_STEPS)
    price = np.clip(price, PRICE_MIN, PRICE_MAX)

    # 날씨 proxy
    daylight = np.maximum(0.0, np.sin(2*np.pi*(t % STEPS_PER_DAY)/STEPS_PER_DAY))
    irr = 1000 * daylight + rng.normal(0, 60, TOTAL_STEPS)   # W/m^2 proxy
    irr = np.clip(irr, 0, 1000)

    wind = np.abs(8 + 3*np.sin(2*np.pi*t/(STEPS_PER_DAY*3)) + rng.normal(0, 2, TOTAL_STEPS))
    temp = 15 + 10*np.sin(2*np.pi*t/(STEPS_PER_DAY*365)) + 5*np.sin(2*np.pi*t/STEPS_PER_DAY) + rng.normal(0, 1.5, TOTAL_STEPS)

    # 재생에너지 가용발전량(0~Q_MAX)
    pv = (irr/1000.0) * (Q_MAX * 0.75) + rng.normal(0, 0.4, TOTAL_STEPS)
    wind_p = ((wind/15.0)**3) * (Q_MAX * 0.35) + rng.normal(0, 0.2, TOTAL_STEPS)
    gen_potential = np.clip(pv*0.7 + wind_p*0.3, 0, Q_MAX)

    return MarketData(
        price=price.astype(np.float32),
        irr=irr.astype(np.float32),
        wind=wind.astype(np.float32),
        temp=temp.astype(np.float32),
        gen_potential=gen_potential.astype(np.float32),
        cost_rate=float(COST_RATE),
    )

MARKET = make_synthetic_market()

# -----------------------------------------------------------
# 2) 환경 (모델프리): 다음 시점 데이터로 전이
#    상태: 두 스트림용 시계열 (에너지/날씨, 가격)
#    액션: (가격,수량)×3 연속값
# -----------------------------------------------------------
class VPPBiddingEnv:
    def __init__(self, market: MarketData, lookback=LOOKBACK, episode_len=EPISODE_LEN):
        self.m = market
        self.lookback = lookback
        self.episode_len = episode_len
        self.ptr = 0
        self.t0 = 0
        self.t1 = 0

    def _slice(self, k):
        sl = slice(k - self.lookback + 1, k + 1)
        E = np.stack([self.m.gen_potential[sl], self.m.irr[sl], self.m.wind[sl], self.m.temp[sl]], axis=0)  # (4, L)
        P = self.m.price[sl]  # (L,)
        return E, P

    def reset(self):
        # 임의의 시작점(lookback 이후, episode_len 여유 고려)
        self.t0 = np.random.randint(self.lookback, TOTAL_STEPS - self.episode_len - 1)
        self.ptr = self.t0
        self.t1 = self.t0 + self.episode_len
        E, P = self._slice(self.ptr)
        obs = {
            "energy_stream": E.copy(),
            "price_stream": P.copy(),
            "clear_price": float(self.m.price[self.ptr]),
            "gen_cap": float(self.m.gen_potential[self.ptr]),
        }
        return obs

    def step(self, action):
        """
        action: tensor/ndarray shape (6,)
        interpreted as [p1,p2,p3, q1,q2,q3]
        """
        p = np.array(action[:3], dtype=np.float32)
        q = np.array(action[3:], dtype=np.float32)

        # 제약 반영: 가격 오름차순, 수량 0~Q_MAX, 수량 비감소(선택)
        p = np.sort(p)
        q = np.clip(q, 0.0, Q_MAX)
        q = np.maximum.accumulate(q)

        clear = float(self.m.price[self.ptr])
        cap = float(self.m.gen_potential[self.ptr])

        # 낙찰 수량(q_offer)
        q_offer = 0.0
        for i in range(3):
            if p[i] <= clear:
                q_offer = q[i]
        # 가용 발전량 한도 내
        q_offer = float(min(q_offer, cap))

        # 비용 및 보상
        C_t = COST_RATE * q_offer
        reward = clear * q_offer - C_t

        # 다음 상태로 전이(모델프리: 실제 다음 시점 데이터 사용)
        self.ptr += 1
        done = self.ptr >= self.t1
        E, P = self._slice(self.ptr)
        obs = {
            "energy_stream": E.copy(),
            "price_stream": P.copy(),
            "clear_price": float(self.m.price[self.ptr]),
            "gen_cap": float(self.m.gen_potential[self.ptr]),
        }
        return obs, reward, done, {"q_offer": q_offer, "clearing_price": clear}

# -----------------------------------------------------------
# 3) 두 스트림 피처 추출기 + 정책/가치망
# -----------------------------------------------------------
class ResidBlock1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=1),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.net(x) + x)

class EnergyCNN(nn.Module):
    """에너지/날씨 스트림용 1D-CNN + Residual"""
    def __init__(self, in_ch=4, latent=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, padding=2),
            nn.GELU(),
            ResidBlock1D(32),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.GELU(),
            ResidBlock1D(64),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(64, latent)

    def forward(self, x):  # x: (B,4,L)
        h = self.stem(x)           # (B,64,L)
        h = self.head(h).squeeze(-1)  # (B,64)
        return self.proj(h)        # (B,latent)

class PriceLSTM(nn.Module):
    """가격 스트림용 LSTM (더블 레이어)"""
    def __init__(self, in_dim=1, hidden=64, latent=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden,
                            num_layers=num_layers, batch_first=True)
        self.proj = nn.Linear(hidden, latent)

    def forward(self, x):  # x: (B,L,1)
        out, _ = self.lstm(x)
        h = out[:, -1, :]     # 마지막 타임스텝 hidden
        return self.proj(h)   # (B,latent)

class PolicyNet(nn.Module):
    def __init__(self, latent=64):
        super().__init__()
        self.energy = EnergyCNN(in_ch=4, latent=latent)
        self.price = PriceLSTM(in_dim=1, hidden=64, latent=latent)
        self.core = nn.Sequential(
            nn.Linear(latent*2, 128),
            nn.GELU(),
        )
        # 액션(가격3, 수량3) 평균과 로그표준편차
        self.mu = nn.Linear(128, 6)
        self.log_std = nn.Parameter(torch.full((6,), -0.5))
        # 가치함수
        self.v = nn.Linear(128, 1)

    def forward(self, obs):
        # obs dict -> tensor
        E = torch.tensor(obs["energy_stream"], dtype=torch.float32, device=DEVICE)  # (4,L)
        P = torch.tensor(obs["price_stream"], dtype=torch.float32, device=DEVICE)    # (L,)
        E = E.unsqueeze(0)  # (1,4,L)
        P = P.view(1, -1, 1)  # (1,L,1)

        fe1 = self.energy(E)     # (1,latent)
        fe2 = self.price(P)      # (1,latent)
        h = self.core(torch.cat([fe1, fe2], dim=-1))  # (1,128)

        mu = self.mu(h).squeeze(0)              # (6,)
        std = torch.exp(self.log_std)           # (6,)
        v = self.v(h).squeeze(0)                # ()
        return mu, std, v

    @staticmethod
    def squash_and_scale(act_raw):
        """
        원시 샘플 act_raw ~ Normal(mu,std)
        -> tanh로 [-1,1], 범위 스케일링하여 실제 액션(가격, 수량) 생성
        """
        a = torch.tanh(act_raw)
        p_raw = a[:3]
        q_raw = a[3:]

        # 가격 스케일링
        p = (p_raw + 1.0) * 0.5 * (PRICE_MAX - PRICE_MIN) + PRICE_MIN  # [PRICE_MIN, PRICE_MAX]
        # 수량 스케일링
        q = (q_raw + 1.0) * 0.5 * Q_MAX                                # [0, Q_MAX]

        # 가격 오름차순, 수량 비감소
        p_sorted, _ = torch.sort(p)
        q_sorted, _ = torch.sort(q)
        return torch.cat([p_sorted, q_sorted], dim=0)  # (6,)

# -----------------------------------------------------------
# 4) PPO 에이전트(간단 구현)
# -----------------------------------------------------------
class PPOAgent:
    def __init__(self):
        self.net = PolicyNet().to(DEVICE)
        self.opt = optim.Adam(self.net.parameters(), lr=LR)

        self.buf_obs_E = []
        self.buf_obs_P = []
        self.buf_actions = []
        self.buf_logp = []
        self.buf_rewards = []
        self.buf_dones = []
        self.buf_values = []

    def policy(self, obs):
        with torch.no_grad():
            mu, std, v = self.net(obs)
            dist = torch.distributions.Normal(mu, std)
            a_raw = dist.sample()
            logp = dist.log_prob(a_raw).sum()
            a = self.net.squash_and_scale(a_raw)  # (6,)
        return a.cpu().numpy(), float(logp.cpu()), float(v.cpu())

    def store(self, obs, action, logp, reward, done, value):
        self.buf_obs_E.append(obs["energy_stream"].copy())
        self.buf_obs_P.append(obs["price_stream"].copy())
        self.buf_actions.append(action.copy())
        self.buf_logp.append(logp)
        self.buf_rewards.append(reward)
        self.buf_dones.append(done)
        self.buf_values.append(value)

    def finish_and_learn(self, next_value=0.0):
        # 1) 리턴/어드밴티지 계산 (MC + bootstrap)
        rewards = np.array(self.buf_rewards, dtype=np.float32)
        dones = np.array(self.buf_dones, dtype=np.bool_)
        values = np.array(self.buf_values + [next_value], dtype=np.float32)

        returns = []
        G = values[-1]
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values[:-1])):
            G = r + GAMMA * G * (1.0 - float(d))
            returns.append(G)
        returns = list(reversed(returns))
        returns_t = torch.tensor(returns, device=DEVICE)
        values_t = torch.tensor(values[:-1], device=DEVICE)
        adv = returns_t - values_t
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 2) 텐서화
        E = torch.tensor(np.array(self.buf_obs_E), dtype=torch.float32, device=DEVICE)  # (T,4,L)
        P = torch.tensor(np.array(self.buf_obs_P), dtype=torch.float32, device=DEVICE)  # (T,L)
        A = torch.tensor(np.array(self.buf_actions), dtype=torch.float32, device=DEVICE) # (T,6)
        old_logp = torch.tensor(np.array(self.buf_logp), dtype=torch.float32, device=DEVICE)

        # 3) PPO 학습 루프
        T = E.size(0)
        idx = np.arange(T)
        for _ in range(PPO_EPOCHS):
            np.random.shuffle(idx)
            for s in range(0, T, BATCH_SIZE):
                j = idx[s:s+BATCH_SIZE]
                if len(j) == 0: continue

                # 배치 obs 구성
                batch = {
                    "energy_stream": E[j],           # (B,4,L)
                    "price_stream": P[j],            # (B,L)
                }
                # 네트 통과(배치)
                mu_list, std, v_list = [], None, []
                for b in range(len(j)):
                    mu, std_now, v = self.net({
                        "energy_stream": batch["energy_stream"][b].cpu().numpy(),
                        "price_stream" : batch["price_stream"][b].cpu().numpy(),
                    })
                    mu_list.append(mu.unsqueeze(0))
                    v_list.append(v.unsqueeze(0))
                    if std is None: std = std_now
                mu = torch.cat(mu_list, dim=0)     # (B,6)
                v = torch.cat(v_list, dim=0).squeeze(-1)  # (B,)

                # 분포/로그확률 (squash 이전 raw에 대해)
                dist = torch.distributions.Normal(mu, torch.exp(self.net.log_std))
                # 역으로 raw를 알 수 없어 simple surrogate 사용(교육용):
                # 여기서는 squash 전후 불일치의 이론적 보정은 생략(데모용)
                # -> 실제 연구/운영에서는 TanhGaussian w/ change-of-variables 권장
                logp = dist.log_prob(A[j]).sum(dim=-1)

                ratio = torch.exp(logp - old_logp[j])
                surr1 = ratio * adv[j]
                surr2 = torch.clamp(ratio, 1.0-CLIP_EPS, 1.0+CLIP_EPS) * adv[j]
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss  = torch.mean((returns_t[j] - v)**2)
                entropy     = torch.mean(dist.entropy().sum(dim=-1))

                loss = policy_loss + VF_COEF*value_loss - ENTROPY_COEF*entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()

        # 4) 버퍼 비우기
        self.buf_obs_E.clear(); self.buf_obs_P.clear()
        self.buf_actions.clear(); self.buf_logp.clear()
        self.buf_rewards.clear(); self.buf_dones.clear(); self.buf_values.clear()

# -----------------------------------------------------------
# 5) 학습 루프
# -----------------------------------------------------------
def to_tensor_obs(obs):
    # 환경 dict -> 정책 forward용(np 기반)으로 그대로 사용
    return obs

def train(n_episodes=N_EPISODES):
    env = VPPBiddingEnv(MARKET, lookback=LOOKBACK, episode_len=EPISODE_LEN)
    agent = PPOAgent()

    for ep in range(1, n_episodes+1):
        obs = env.reset()
        ep_ret, ep_q = 0.0, 0.0
        done = False

        while not done:
            a, logp, v = agent.policy(to_tensor_obs(obs))
            next_obs, r, done, info = env.step(a)

            agent.store(obs, a, logp, r, done, v)
            ep_ret += r
            ep_q   += info["q_offer"]
            obs = next_obs

        # 마지막 스텝 bootstrap value
        with torch.no_grad():
            _, _, v_last = agent.net(to_tensor_obs(obs))
        agent.finish_and_learn(next_value=float(v_last.cpu()))

        if ep % 5 == 0:
            print(f"[Episode {ep:4d}] return={ep_ret:8.2f} | offered_energy={ep_q:7.2f} kWh")

if __name__ == "__main__":
    train(n_episodes=N_EPISODES)