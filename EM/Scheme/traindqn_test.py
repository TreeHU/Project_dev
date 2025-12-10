# State Transition 추가

import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Multi‐Stage VPP Environment
class MultiStageVPPEnv:
    def __init__(self,
                 horizon=12,      # 6시~18시 12시간
                 block=3,         # 3시간 블록, Stage 내의 시간 단위
                 soc_max=1.0,     # State Of Charge (SOC)
                 charge_cost=1.0, # Lamba_elec : Electricity price for usage of companies
                 b_ch_eff = 0.9,       # ESS 충전 효율 v_c
                 b_dc_eff = 0.9,       # ESS 방전 효율 v_d
                 incentive_rate=3.0,
                 tax_fare_rate = 1.137, # β: Indicator for tax combined fare (β 1:137 in the Korean market)
                 lamda_elec=1): # λinc: Balancing incentive at time h (3 KRW/kWh in Korea market)
        self.H = horizon # Day-ahead Market 에서 예측 Time Frame
        self.block = block # Time Frame 묶음.
        self.num_blocks = horizon // block # Time Frame 묶음 갯수
        self.num_stages = 1 + self.num_blocks # Stage = {1,2,3,4,5}
        self.soc_max       = soc_max
        self.charge_cost   = charge_cost
        self.b_ch_eff         = b_ch_eff
        self.b_dc_eff         = b_dc_eff
        self.incentive_rate = incentive_rate
        self.tax_fare_rate = tax_fare_rate
        self.lamda_elec = lamda_elec
        self.reset()

    def reset(self):
        # --- Stage1 forecast & market price ---
        self.forecast_pv  = np.random.uniform(0.5, 1.0, size=self.H) # 발전량 예측
        self.forecast_dr  = np.random.uniform(0.5, 1.0, size=self.H) # 수요량 예측
        self.smp_forecast = np.random.uniform(50, 100, size=self.H)  # 가격 예측
        # --- 당일 actual (forecast + noise) ---
        self.actual_pv    = self.forecast_pv + np.random.normal(0, 0.1, size=self.H) # Intraday market 에서의 실제 발전량, Paper 상에는 (PV 예측량 + 오차량)
        self.actual_dr    = self.forecast_dr + np.random.normal(0, 0.15, size=self.H) # Intraday market 에서의 실제 수요량, Paper 상에는 (DR 예측량 + 오차량)
        # ESS 초기 완전충전
        self.soc   = self.soc_max # Paper 상에서 ESS 의 초기 상태는 완전 충전을 가정함.
        self.stage = 1 # 초기 Stage
        return self.state()

    def state(self):
        if self.stage == 1:
            return np.concatenate([self.forecast_pv, self.forecast_dr]).astype(np.float32)
        else:
            i0 = (self.stage - 2)*self.block
            pv = self.actual_pv[i0:i0+self.block]
            dr = self.actual_dr[i0:i0+self.block]
            return np.concatenate([[self.soc], pv, dr]).astype(np.float32)

    def state_element_size(self): # State Size 출력
        return 2*self.H if self.stage==1 else (1 + 4*self.block)

    def action_element_size(self): # Action Size 출력
        return self.H if self.stage == 1 else 4 * self.block

    def step(self, action):
        done = False

        if self.stage == 1:
            q = np.asarray(action, dtype=np.float32)   # shape = (H,)
            reward = float(q @ self.smp_forecast)
            self.prev_bid = q.copy()

        else:
            # Stage 2–5: 당일 운영
            a = np.asarray(action, dtype=np.float32)      # shape = (4*block,)
            # decision variables 분리
            b_ch    = a[0             : self.block]       # b^{c,ch}(h)
            b_dc    = a[self.block     : 2*self.block]    # b^{c,dc}(h)
            I_plus  = a[2*self.block   : 3*self.block]    # I⁺(h)
            I_minus = a[3*self.block   : 4*self.block]    # I⁻(h)

            # 현재 블록의 시간 인덱스 시작점
            i0 = (self.stage - 2) * self.block

            # --- A₁: Revenue Adjustment ---
            # f_t 에서 A₁ = Σₕ λ_SMP(h)·[q(h) − (Z(h)+ΔZ(h) − Σ v^c(h))]
            # 여기서는 Σv^c=0, 실제 공급량 = actual_pv + actual_dr 로 가정
            A1 = 0.0
            for i in range(self.block):
                h = i0 + i
                lam_smp = self.smp_forecast[h]
                qh      = self.prev_bid[h]
                supply  = self.actual_pv[h] + self.actual_dr[h]
                A1 += lam_smp * (qh - supply)

            # --- A₂: Balancing Incentive Penalty ---
            # A₂ = Σₕ λ_inc(h)·q(h)·[1 − I⁺(h) − I⁻(h)]
            # I⁺/I⁻가 1일 때만 보상, 아니면 페널티로 작용
            A2 = 0.0
            for i in range(self.block):
                h = i0 + i
                lam_inc = self.incentive_rate          # λ_inc(h)
                qh      = self.prev_bid[h]
                A2 += lam_inc * qh * (1.0 - I_plus[i] - I_minus[i])

            # --- A₃: ESS 운영비용 & 전력사용 보상 ---
            # A₃ = β·Σₕ λ_elec(h)·[G(h)+ΔG(h) − b^{dc}(h) + b^{ch}(h)]
            A3 = 0.0
            for i in range(self.block):
                h = i0 + i
                # 주석처리함 lamda_elec = 1로 두기 위해...
                #lam_elec = self.smp_forecast[h]        # λ_elec(h), 예시로 SMP 재사용
                lam_elec = self.lamda_elec
                # 여기서 G+ΔG = actual_dr
                calc_usage = self.actual_dr[h] - b_dc[i] + b_ch[i] # 충·방전 동작을 반영한 “순 전력 수요”
                A3 += self.tax_fare_rate * lam_elec * calc_usage

            # --- SoC 업데이트 (물리 제약) ---
            # e(h+1) = e(h) + b^{ch} − b^{dc}
            delta = np.sum(b_ch * self.b_ch_eff) - np.sum(b_dc / self.b_dc_eff)
            self.soc = np.clip(self.soc + delta, 0.0, self.soc_max)

            # 최종 보상: A₁ − A₂ + A₃
            reward = A1 - A2 + A3

        # 다음 스테이지로 진행
        if self.stage >= self.num_stages:
            done = True
            next_state = None
        else:
            self.stage += 1
            next_state = self.state()

        return next_state, reward, done, {}

# 2. DQN 네트워크 & ReplayBuffer (이산 액션 예시)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)
    def push(self, s,a,r,ns,d):
        self.buf.append((s,a,r,ns,d))
    def sample(self,bz):
        batch = random.sample(self.buf,bz)
        S,A,R,NS,D = zip(*batch)
        NS = [s if s is not None else np.zeros_like(S[0]) for s in NS]
        return np.vstack(S), A, R, np.vstack(NS), D
    def __len__(self):
        return len(self.buf)

# 3. 학습 함수
def train_dqn(env, episodes=200, batch_size=32, gamma=0.99, lr=1e-3,
              eps_start=1.0, eps_end=0.01, eps_decay=0.995,
              target_update=20, prefill=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 네트워크 & 타깃 네트워크
    net1 = DQN(input_dim=2*env.H,        output_dim=env.H).to(device)
    tgt1 = DQN(input_dim=2*env.H,        output_dim=env.H).to(device)
    net2 = DQN(input_dim=1 + 2*env.block, output_dim=4*env.block).to(device)
    tgt2 = DQN(input_dim=1 + 2*env.block, output_dim=4*env.block).to(device)
    tgt1.load_state_dict(net1.state_dict())
    tgt2.load_state_dict(net2.state_dict())

    opt1 = optim.Adam(net1.parameters(), lr=lr)
    opt2 = optim.Adam(net2.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 2) 리플레이 버퍼
    buf1, buf2 = ReplayBuffer(), ReplayBuffer()

    eps = eps_start
    hist_reward, eps_history = [], []

    for ep in range(1, episodes+1):
        state = env.reset()
        ep_reward = 0.0

        # --- Stage 1 action 선택 & 실행 ---
        s1 = state
        if random.random() < eps:
            q_max = 1.0
            a1 = np.random.uniform(0, q_max, size=env.H)
        else:
            with torch.no_grad():
                a1 = net1(torch.tensor(s1,device=device)
                          .float().unsqueeze(0)).cpu().numpy().squeeze()
            a1 = np.clip(a1, 0.0, None)

        ns, r, done, _ = env.step(a1)
        buf1.push(s1, a1, r, ns, done)
        ep_reward += r

        # Stage1 DQN 업데이트
        if len(buf1) >= batch_size:
            S, A, R, NS, D = buf1.sample(batch_size)
            S_v  = torch.tensor(S, device=device).float()
            A_v  = torch.tensor(A, device=device).float()   # continuous actions
            R_v  = torch.tensor(R, device=device).float()
            #NS_v = torch.tensor(NS,device=device).float()
            D_v  = torch.tensor(D, device=device).float()

            # Q(S,A) 추정치: net1(S) · A (내적)
            # 여기선 continuous action: treat net1 outputs as Q-values per hour,
            # so total Q = sum(net1(S)*A), 유연하게 설계할 수 있습니다.
            q_pred = (net1(S_v) * A_v).sum(dim=1)


            # 다음 상태의 최대 Q
            #q_next = tgt1(NS_v).max(dim=1)[0]
            #q_target = R_v + gamma * q_next * (1 - D_v)

            q_target = R_v

            loss1 = loss_fn(q_pred, q_target.detach())
            opt1.zero_grad()
            loss1.backward()
            opt1.step()

        # --- Stage 2–5 루프 ---
        while not done:
            s2 = ns
            i0 = (env.stage - 2)*env.block

            # 1) ESS 충·방전량 결정
            if random.random() < eps:
                ch_max, dc_max = 0.5, 0.5
                b_ch = np.random.uniform(0, ch_max, size=env.block)
                b_dc = np.random.uniform(0, dc_max, size=env.block)
            else:
                with torch.no_grad():
                    out = net2(torch.tensor(s2,device=device)
                              .float().unsqueeze(0)).cpu().numpy().squeeze()
                b_ch = np.clip(out[0:env.block],    0.0, None)
                b_dc = np.clip(out[env.block:2*env.block], 0.0, None)

            # 2) 인센티브 이진변수 결정
            I_plus  = np.zeros(env.block, dtype=np.float32)
            I_minus = np.zeros(env.block, dtype=np.float32)
            for i in range(env.block):
                h = i0 + i
                qh     = env.prev_bid[h]
                actual = env.actual_pv[h] + env.actual_dr[h]
                err    = actual - qh
                rel_err = abs(err)/max(qh,1e-6)
                if err >= 0 and rel_err <= 0.08:
                    I_plus[i] = 1.0
                elif err < 0 and rel_err <= 0.08:
                    I_minus[i] = 1.0

            # 3) 하나의 action 벡터로 합치기
            a2 = np.concatenate([b_ch, b_dc, I_plus, I_minus]).astype(np.float32)

            ns, r, done, _ = env.step(a2)
            buf2.push(s2, a2, r, ns, done)
            ep_reward += r

            # Stage2–5 DQN 업데이트
            if len(buf2) >= batch_size:
                S2, A2, R2, NS2, D2 = buf2.sample(batch_size)
                S2_v = torch.tensor(S2,device=device).float()
                A2_v = torch.tensor(A2,device=device).float()
                R2_v = torch.tensor(R2,device=device).float()
                NS2_v= torch.tensor(NS2,device=device).float()
                D2_v = torch.tensor(D2,device=device).float()

                # 예시: net2(S2) 반환 4*block 차원 Q-values,
                #       action A2_v 와 곱해 스칼라 Q 예측
                q2_pred = (net2(S2_v) * A2_v).sum(dim=1)
                q2_next = tgt2(NS2_v).max(dim=1)[0]
                q2_target = R2_v + gamma * q2_next * (1 - D2_v)

                loss2 = loss_fn(q2_pred, q2_target.detach())
                opt2.zero_grad()
                loss2.backward()
                opt2.step()

        # 타깃 네트워크 동기화
        if ep % target_update == 0:
            tgt1.load_state_dict(net1.state_dict())
            tgt2.load_state_dict(net2.state_dict())

        hist_reward.append(ep_reward)
        eps_history.append(eps)
        eps = max(eps_end, eps * eps_decay)

        if ep % 10 == 0:
            print(f"Ep {ep:3d}, Reward {ep_reward:.2f}, Eps {eps:.3f}")

    return net1, net2, hist_reward, eps_history

# 4. 실행 및 결과 플롯
if __name__ == "__main__":
    env = MultiStageVPPEnv()
    net1, net2, rewards, epsilons  = train_dqn(env, episodes=10000, prefill=500)

    plt.figure()
    plt.plot(rewards)
    plt.title("Episode Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig('./Project/EM/Simulation/Reward/Reward_epi_%d.png'%(10000))

    # ε 변화 그래프
    plt.figure()
    plt.plot(epsilons)
    plt.title("Epsilon Decay over Time")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True)
    #plt.savefig('savefig_default.png')


