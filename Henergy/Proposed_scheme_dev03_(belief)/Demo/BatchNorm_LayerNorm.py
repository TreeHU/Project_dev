# Re-run only the small-batch (B=1) part with error handling and eval-mode demonstration
import torch

# Recreate modules consistent with previous cell
bn = torch.nn.BatchNorm1d(num_features=3, affine=False, eps=1e-5)
ln = torch.nn.LayerNorm(normalized_shape=3, elementwise_affine=False, eps=1e-5)
bn.train(); ln.train()

X1 = torch.tensor([[1.0, 2.0, 3.0]])  # (1,3)

print("---- Small-batch case (B=1) ----")
print("X1:\n", X1)

# Training-time BN with B=1 raises an error (needs >1 per-channel)
try:
    Y_bn_1_train = bn(X1.clone())
    print("BatchNorm1d output (B=1, training):\n", Y_bn_1_train)
except Exception as e:
    print("BatchNorm1d (B=1, training) ERROR:", e)

# Switch BN to eval so it uses running stats (but they are still zeros at init)
bn.eval()
Y_bn_1_eval = bn(X1.clone())
print("BatchNorm1d output (B=1, eval - using running stats):\n", Y_bn_1_eval)

# LayerNorm works in training with B=1
ln.train()
Y_ln_1 = ln(X1.clone())
print("LayerNorm output (B=1, training):\n", Y_ln_1)