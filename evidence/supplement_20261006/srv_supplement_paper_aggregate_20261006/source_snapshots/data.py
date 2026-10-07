"""Data loading and batching for the FAD aptamer GBSA dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from . import tokenizer as tok


def load_fad(data_path: str, target: str = "gbsa"):
    df = pd.read_csv(data_path)
    seqs = df["Sequence"].astype(str).tolist()
    y = df[target].to_numpy(dtype=np.float32)
    return seqs, y


def tokenize_batch(seqs: list[str], y: np.ndarray | None = None):
    """Tokenize a list of sequences into a padded batch.

    Returns (token_ids (b, s), pad_mask (b, s) True=PAD, labels (b,) or None).
    """
    ids = [tok.tokenize(s) for s in seqs]
    maxlen = max(len(t) for t in ids)
    batch = torch.full((len(ids), maxlen), tok.PAD_ID, dtype=torch.long)
    mask = torch.ones((len(ids), maxlen), dtype=torch.bool)
    for i, t in enumerate(ids):
        batch[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        mask[i, : len(t)] = False
    labels = torch.tensor(y, dtype=torch.float32) if y is not None else None
    return batch, mask, labels
