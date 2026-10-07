"""Pretraining objectives reproducing gkag145 (NAR 2026): MLM + triplet margin loss.

MLM: 15% of non-special tokens are selected; 80% -> [MASK], 10% -> random
3-mer, 10% unchanged. Cross-entropy is computed on masked positions only via a
position-wise single-layer FFN head.

Triplet: anchor = [CLS] embedding of the unmasked sequence; positive = [CLS]
embedding of the masked version of the same sequence; negative = [CLS] embedding
of the masked version of the hard negative (the in-batch sequence whose unmasked
embedding is nearest to the anchor). Margin = 1.0.

Total loss = MLM loss + triplet loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tokenizer as tok
from .data import tokenize_batch
from .encoder import SequenceEncoder

# ids of the 64 real 3-mer tokens (for random-token replacement)
_MER_IDS = torch.tensor([i for i in range(tok.VOCAB_SIZE) if i not in tok.MASKABLE_EXCLUDE])


def make_pad_mask(token_ids: torch.Tensor) -> torch.Tensor:
    return token_ids == tok.PAD_ID


def mask_tokens(token_ids: torch.Tensor, mask_prob: float = 0.15,
                generator: torch.Generator | None = None):
    """Return (masked_ids, labels). labels = -100 for positions without a target."""
    device = token_ids.device
    labels = token_ids.clone()
    b, s = token_ids.shape
    special = torch.zeros_like(token_ids, dtype=torch.bool)
    for sp in tok.MASKABLE_EXCLUDE:
        special |= token_ids == sp

    # Generate on CPU (generator must be a CPU generator) then move to device.
    prob = torch.rand(b, s, generator=generator).to(device)
    rnd = torch.rand(b, s, generator=generator).to(device)
    selected = (prob < mask_prob) & (~special)

    # 80% [MASK], 10% random, 10% unchanged
    to_mask = selected & (rnd < 0.8)
    to_random = selected & (rnd >= 0.8) & (rnd < 0.9)

    masked = token_ids.clone()
    masked[to_mask] = tok.MASK_ID
    random_ids = _MER_IDS[
        torch.randint(len(_MER_IDS), (int(to_random.sum().item()),), generator=generator)
    ].to(device)
    masked[to_random] = random_ids

    labels[~selected] = -100
    return masked, labels


class MLMHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc(hidden)


class PretrainWrapper(nn.Module):
    """Encoder + MLM head; computes MLM loss and triplet loss."""

    def __init__(self, encoder: SequenceEncoder, vocab_size: int, margin: float = 1.0):
        super().__init__()
        self.encoder = encoder
        self.mlm_head = MLMHead(encoder.d_model, vocab_size)
        self.triplet = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, token_ids: torch.Tensor, pad_mask: torch.Tensor | None = None,
                generator: torch.Generator | None = None):
        masked_ids, labels = mask_tokens(token_ids, generator=generator)

        anchor = self.encoder(token_ids, pad_mask)                          # (b, d)
        hidden, pos = self.encoder(masked_ids, pad_mask, return_hidden=True)

        # Hard negative: nearest unmasked embedding in-batch (exclude self).
        d = torch.cdist(anchor, anchor)
        d = d.masked_fill(torch.eye(d.size(0), device=d.device, dtype=torch.bool), float("inf"))
        neg_idx = d.argmin(dim=1)

        # Negative = masked version of the hard-negative sequence.
        neg_ids, neg_labels = mask_tokens(token_ids[neg_idx], generator=generator)
        neg_pad = pad_mask[neg_idx] if pad_mask is not None else None
        neg_hidden, neg = self.encoder(neg_ids, neg_pad, return_hidden=True)

        # MLM loss on masked positions only, for both positive and negative.
        logits = self.mlm_head(hidden)
        mlm_pos = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100)
        neg_logits = self.mlm_head(neg_hidden)
        mlm_neg = F.cross_entropy(neg_logits.transpose(1, 2), neg_labels, ignore_index=-100)

        trip_loss = self.triplet(anchor, pos, neg)
        return mlm_pos + mlm_neg, trip_loss, (masked_ids, labels)


def batched_indices(n: int, batch_size: int, gen: torch.Generator):
    perm = torch.randperm(n, generator=gen)
    for i in range(0, n, batch_size):
        yield perm[i : i + batch_size]


def pretrain(encoder: SequenceEncoder, seqs, device, epochs, batch_size, lr, gamma, seed):
    """Pretrain the encoder with MLM + triplet loss. Returns the trained encoder."""
    wrapper = PretrainWrapper(encoder, tok.VOCAB_SIZE).to(device)
    opt = torch.optim.Adam(wrapper.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    ids, mask, _ = tokenize_batch(seqs)
    ids, mask = ids.to(device), mask.to(device)
    for ep in range(epochs):
        wrapper.train()
        tot_mlm = tot_trip = 0.0
        for idxs in batched_indices(len(ids), batch_size, gen):
            x, m = ids[idxs], mask[idxs]
            opt.zero_grad()
            mlm, trip, _ = wrapper(x, m, generator=gen)
            loss = mlm + trip
            loss.backward()
            opt.step()
            tot_mlm += mlm.item() * len(idxs)
            tot_trip += trip.item() * len(idxs)
        for g in opt.param_groups:
            g["lr"] *= gamma
        print(f"  pretrain ep {ep+1}/{epochs}  mlm={tot_mlm/len(ids):.4f}  trip={tot_trip/len(ids):.4f}")
    return wrapper
