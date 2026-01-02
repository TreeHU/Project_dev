# profile_nets.py
import os, sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# 각 네트워크 import 경로는 프로젝트에 맞게 수정
from Algorithm.network_lstm import ActorNet as LSTMActor, CriticNet as LSTMCritic
from Algorithm.network_transformer import ActorNet as TrActor, CriticNet as TrCritic
from Algorithm.network_mamba import ActorNet as MbActor, CriticNet as MbCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 아래 값은 프로젝트에 맞게 조정 ----
D = 128   # env.obs_dim으로 바꾸세요
A = 1
T = 1     # (B,T,D)로 시퀀스 처리면 조정
B = 1     # FLOPs는 보통 B=1 기준으로 비교 (원하면 batch_size로 바꿔도 됨)
HIDDEN = 256
# --------------------------------------

def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def _try_fvcore_flops(model: torch.nn.Module, inputs: tuple):
    """
    fvcore 기반 FLOPs 계산 (가장 안정적).
    설치: pip install fvcore
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, inputs).total()
        return float(flops)
    except Exception as e:
        return None

def _format_num(n: float) -> str:
    if n is None:
        return "N/A"
    if n >= 1e12:
        return f"{n/1e12:.3f} TFLOPs"
    if n >= 1e9:
        return f"{n/1e9:.3f} GFLOPs"
    if n >= 1e6:
        return f"{n/1e6:.3f} MFLOPs"
    if n >= 1e3:
        return f"{n/1e3:.3f} KFLOPs"
    return f"{n:.1f} FLOPs"

def run_actor(model, name):
    model = model.to(device).eval()

    # Actor 입력: (B,T,D) 또는 (B,D) 모두 가능하지만,
    # 여기선 일반적으로 (B,T,D)로 넣고 내부에서 처리하게 둠.
    x = torch.randn(B, T, D, device=device)

    params = _count_params(model)
    flops = _try_fvcore_flops(model, (x,))

    print(f"[Actor] {name}")
    print(f"  Params: {params/1e6:.3f} M")
    print(f"  FLOPs : {_format_num(flops)}  (B={B}, T={T}, D={D})")
    print("")

def run_critic(model, name):
    model = model.to(device).eval()

    # Critic 입력: s와 a01
    s = torch.randn(B, T, D, device=device)

    # a01 shape는 네 Critic forward가 (B,A) 또는 (B,T,A) 둘다 처리하므로
    # 시퀀스 정합을 위해 (B,T,A)로 맞추는 편이 안전
    a01 = torch.randn(B, T, A, device=device)

    params = _count_params(model)
    flops = _try_fvcore_flops(model, (s, a01))

    print(f"[Critic] {name}")
    print(f"  Params: {params/1e6:.3f} M")
    print(f"  FLOPs : {_format_num(flops)}  (B={B}, T={T}, D={D}, A={A})")
    print("")

if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- LSTM ----
    run_actor(LSTMActor(D, A, HIDDEN), "LSTM Actor")
    run_critic(LSTMCritic(D, A, HIDDEN), "LSTM Critic")

    # ---- Mamba ----
    run_actor(MbActor(D, A, HIDDEN), "Mamba Actor")
    run_critic(MbCritic(D, A, HIDDEN), "Mamba Critic")

    # ---- Transformer ----
    run_actor(TrActor(D, A, HIDDEN), "Transformer Actor")
    run_critic(TrCritic(D, A, HIDDEN), "Transformer Critic")
