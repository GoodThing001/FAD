"""BERT-like transformer encoder reproducing gkag145 (NAR 2026).

Architecture (from the paper's "Encoder model" section):
  - embedding d_model = 128, fixed cosine positional encoding from source code
  - 6 transformer encoder blocks
  - multi-head self-attention: h = 4 heads, d_attention = 64 per head
    (the paper doubles the per-head dim vs. Vaswani's d_model/h, i.e.
     d_attention = 2 * d_model / h), total attention width 4 * 64 = 256
  - FFN: 128 -> 384 (GELU) -> 128, both layers dropout 10%
  - post-LN (Vaswani/Devlin style), dropout 10%

The sequence embedding is the [CLS] token hidden state after the last block.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CosinePositionalEncoding(nn.Module):
    """Fixed non-learnable cosine positional encoding, matching the paper's code.

    pe[pos, j] = cos(pi * (pos/(j+1) + (j+1))), pos=0..max_len-1, j=0..d_model-1.
    (NOT the standard Vaswani sinusoid.)
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)      # (max_len, 1)
        d = torch.arange(1, d_model + 1, dtype=torch.float).unsqueeze(0)  # (1, d_model)
        pe = torch.cos((pos / d + d) * math.pi)                           # (max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))                        # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    """Self-attention with per-head dim d_attention (not d_model/h)."""

    def __init__(self, d_model: int = 128, h: int = 4, d_attention: int = 64, dropout: float = 0.1):
        super().__init__()
        self.h = h
        self.d_attention = d_attention
        inner = h * d_attention
        self.q = nn.Linear(d_model, inner)
        self.k = nn.Linear(d_model, inner)
        self.v = nn.Linear(d_model, inner)
        self.out = nn.Linear(inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = d_attention ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, s, _ = x.shape
        q = self.q(x).view(b, s, self.h, self.d_attention).transpose(1, 2)
        k = self.k(x).view(b, s, self.h, self.d_attention).transpose(1, 2)
        v = self.v(x).view(b, s, self.h, self.d_attention).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # mask: (b, s) True for PAD positions -> -inf
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, s, -1)
        return self.out(out)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int = 128, h: int = 4, d_attention: int = 64,
                 ffn_dim: int = 384, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, h, d_attention, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.ln1(x + self.attn(x, mask))
        x = self.ln2(x + self.ffn(x))
        return x


class SequenceEncoder(nn.Module):
    """Transformer encoder; returns (token_hidden, cls_embedding)."""

    def __init__(self, vocab_size: int, d_model: int = 128, h: int = 4,
                 d_attention: int = 64, ffn_dim: int = 384, n_blocks: int = 6,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = CosinePositionalEncoding(d_model, max_len)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(d_model, h, d_attention, ffn_dim, dropout)
             for _ in range(n_blocks)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor, pad_mask: torch.Tensor | None = None,
                return_hidden: bool = False):
        """token_ids: (b, s) long.

        Returns [CLS] embedding (b, d_model); if return_hidden, returns
        (full_hidden (b, s, d_model), cls_embedding (b, d_model)).
        """
        x = self.embedding(token_ids)
        x = self.pos(x)
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        cls = x[:, 0]  # [CLS] token embedding
        return (x, cls) if return_hidden else cls


def default_encoder(vocab_size: int) -> SequenceEncoder:
    return SequenceEncoder(vocab_size)
