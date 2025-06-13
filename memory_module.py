import torch
from torch import nn

class MemoryFuse(nn.Module):
    def __init__(self, hidden_size, memory_size):
        super().__init__()
        # 可学习的“记忆矩阵”
        self.memory = nn.Parameter(torch.zeros(memory_size, hidden_size))
        # 融合层：把 token 向量和 memory 拼接后再线性映射回 hidden_size
        self.fuse = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, hidden_states):
        # hidden_states: [batch, seq_len, hidden_size]
        bsz, seq_len, h = hidden_states.size()
        # 取记忆矩阵第一行 (或者你希望的方式) 并扩展到 seq_len
        mem0 = self.memory[:1, :].unsqueeze(0)                # [1,1,h]
        mem0 = mem0.expand(bsz, seq_len, h)                   # [bsz, seq_len, h]
        # 拼接并融合
        fused = self.fuse(torch.cat([hidden_states, mem0], dim=-1))
        return fused  # [bsz, seq_len, h]
