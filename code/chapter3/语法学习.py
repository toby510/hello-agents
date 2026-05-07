import torch
import torch.nn as nn

# 模拟你的场景
d_model = 64
W_q = nn.Linear(d_model, d_model)

# 输入
Q = torch.randn(1, 9, d_model)
print(f"原始 Q: {Q.shape}")  # [1, 9, 64]

# 步骤1: W_q(Q) 返回什么？
temp = W_q(Q)
print(f"W_q(Q) 返回: {type(temp)}, shape: {temp.shape}")
# <class 'torch.Tensor'>, shape: torch.Size([1, 9, 64])

# 步骤2: split_heads 接收什么？
def split_heads(x):
    print(f"split_heads 接收: {type(x)}, shape: {x.shape}")
    # 确实是张量！
    batch, seq, d = x.size()
    return x.view(batch, seq, 8, 8).transpose(1, 2)

result = split_heads(temp)
print(f"最终: {result.shape}")  # [1, 8, 9, 8]