"""Regression surrogate + deep ensemble reproducing gkag145 (NAR 2026).

The paper's regression model is a 9x256 MLP (GELU, no dropout, layer-norm every
3 hidden layers) on the 128-d [CLS] embedding, outputting the four expression
levels. For FAD/GBSA we output a single continuous value and use MSE (the
paper's Bernoulli KL divergence applies to [0,1] expression probabilities).

The deep ensemble is n independently-initialized heads on the frozen pretrained
encoder; prediction mean is mu and the std across members is the epistemic
uncertainty sigma.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import SequenceEncoder


class RegressionHead(nn.Module):
    def __init__(self, d_model: int = 128, hidden: int = 256, n_layers: int = 9,
                 out_dim: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(n_layers):
            in_d = d_model if i == 0 else hidden
            layers.append(nn.Linear(in_d, hidden))
            layers.append(nn.GELU())
            if (i + 1) % 3 == 0:
                layers.append(nn.LayerNorm(hidden))
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.net(emb)


class DeepEnsemble(nn.Module):
    """n regression heads sharing a (frozen) pretrained encoder."""

    def __init__(self, encoder: SequenceEncoder, n: int = 30,
                 head_kwargs: dict | None = None):
        super().__init__()
        head_kwargs = head_kwargs or {}
        self.encoder = encoder
        self.n = n
        self.heads = nn.ModuleList([RegressionHead(d_model=encoder.d_model, **head_kwargs)
                                    for _ in range(n)])

    def embed(self, token_ids: torch.Tensor, pad_mask: torch.Tensor | None = None,
              train_dropout: bool = False) -> torch.Tensor:
        prev = self.encoder.training
        if train_dropout:
            self.encoder.train()  # encoder dropout active -> member diversity
        else:
            self.encoder.eval()
        emb = self.encoder(token_ids, pad_mask)
        self.encoder.train(prev)
        return emb

    def predict(self, token_ids: torch.Tensor, pad_mask: torch.Tensor | None = None):
        """Returns (mu, sigma, preds) where preds is (n, b, out).

        sigma is the (unbiased, ddof=1) sample std across ensemble members.
        """
        emb = self.embed(token_ids, pad_mask, train_dropout=False)
        preds = torch.stack([h(emb) for h in self.heads], dim=0)
        mu = preds.mean(dim=0)
        sigma = preds.std(dim=0, unbiased=True)
        return mu, sigma, preds

    def member_parameters(self):
        """Parameters of the heads only (encoder is frozen)."""
        return [p for h in self.heads for p in h.parameters()]
