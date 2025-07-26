import torch
from torch import nn
import math

class MemoryFuse(nn.Module):
    """Debug version: force memory to impact hidden_states directly.
    After verifying gradients become non‑zero, we can revert to gated fusion."""

    def __init__(self, hidden_size, memory_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size

        # 强初始化 memory（确保其对前向有强影响）
        self.memory = nn.Parameter(torch.ones(memory_size, hidden_size))
        nn.init.normal_(self.memory, mean=0.5, std=0.1)

        # 线性映射层（QK）用于注意力计算
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)

        # Dropout 与归一化
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

        # 调试记录
        self.last_weights = None
        self.last_context = None

    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, hidden]
        Q = self.q_proj(hidden_states)                      # [B, S, H]
        K = self.k_proj(self.memory)                        # [M, H]
        scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.hidden_size)  # [B, S, M]

        weights = torch.softmax(scores, dim=-1)             # [B, S, M]

        memory_context = torch.matmul(weights, self.memory) # [B, S, H]

        fused = hidden_states + self.dropout(memory_context)
        out = self.norm(fused)
        
        with torch.no_grad():
            self.memory.copy_(memory_context.mean(dim=1))

        self.last_weights = weights.detach()
        self.last_context = memory_context.detach()
        return out