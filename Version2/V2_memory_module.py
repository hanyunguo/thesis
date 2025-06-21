import torch
from torch import nn

class MemoryFuse(nn.Module):
    def __init__(self, hidden_size, memory_size, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        # Parameterized memory bank
        self.memory = nn.Parameter(torch.randn(memory_size, hidden_size) * 0.02)
        # Projections for attention
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        # Fusion and gating
        self.fuse = nn.Linear(hidden_size * 2, hidden_size)
        self.gate = nn.Linear(hidden_size * 2, hidden_size)
        # Activation, dropout, normalization
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, hidden]
        # 1) Compute Q, K, V
        Q = self.q_proj(hidden_states)             # [b, s, h]
        K = self.k_proj(self.memory)               # [m, h]
        V = self.v_proj(self.memory)               # [m, h]
        # 2) Attention scores and context
        scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.hidden_size)  # [b, s, m]
        weights = torch.softmax(scores, dim=-1)                                 # [b, s, m]
        context = torch.matmul(weights, V)                                      # [b, s, h]
        # 3) Fuse hidden + context
        fuse_input = torch.cat([hidden_states, context], dim=-1)                # [b, s, 2h]
        fused = self.act(self.fuse(fuse_input))                                 # [b, s, h]
        fused = self.dropout(fused)
        # 4) Gated combination
        gate_values = torch.sigmoid(self.gate(fuse_input))                     # [b, s, h]
        gated = gate_values * hidden_states + (1 - gate_values) * fused         # [b, s, h]
        # 5) Residual + LayerNorm
        return self.norm(hidden_states + gated)                                 # [b, s, h]
