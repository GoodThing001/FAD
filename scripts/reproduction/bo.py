"""Bayesian optimization components reproducing gkag145 (NAR 2026).

Acquisition is UCB in its quantile form: with a deep ensemble, u(x) is the
(1-alpha) empirical quantile of the n member predictions (maximization). For a
minimization target (low GBSA) we also provide LCB = alpha-quantile.

Kriging Believer batches by, after each member is selected from the candidate
pool, appending its predicted mean to the training set and refitting the
surrogate (encoder frozen) before selecting the next member, which enforces
batch diversity. Selection is exhaustive over the (compact) pool, as in the
paper.
"""

from __future__ import annotations

import torch


def ucb_quantile(preds: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """preds: (n_members, n_candidates, out) -> u(x) per candidate (maximize)."""
    return torch.quantile(preds, 1.0 - alpha, dim=0)


def lcb_quantile(preds: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """Lower confidence bound for minimization targets (low GBSA)."""
    return torch.quantile(preds, alpha, dim=0)


def acquisition(preds: torch.Tensor, alpha: float = 0.05, maximize: bool = True) -> torch.Tensor:
    return ucb_quantile(preds, alpha) if maximize else lcb_quantile(preds, alpha)


def kriging_believer(ensemble, train_fn, train_ids, train_mask, train_labels,
                     pool_ids, pool_mask, n_batch: int, alpha: float = 0.05,
                     maximize: bool = True):
    """Select n_batch candidates from the pool via Kriging Believer.

    train_fn(cur_ids, cur_mask, cur_labels) retrains the ensemble heads
    (encoder frozen) on the augmented data and is invoked after each member.
    """
    cur_ids, cur_mask, cur_labels = train_ids, train_mask, train_labels
    selected: list[int] = []
    for _ in range(n_batch):
        mu, sigma, preds = ensemble.predict(pool_ids, pool_mask)
        u = acquisition(preds, alpha, maximize).reshape(-1).clone()
        for i in selected:
            u[i] = float("-inf") if maximize else float("inf")
        best = int(torch.argmax(u) if maximize else torch.argmin(u))
        selected.append(best)
        # Kriging Believer: augment training with the predicted mean of best.
        cur_ids = torch.cat([cur_ids, pool_ids[best : best + 1]], dim=0)
        cur_mask = torch.cat([cur_mask, pool_mask[best : best + 1]], dim=0)
        cur_labels = torch.cat([cur_labels, mu.reshape(-1)[best : best + 1]], dim=0)
        train_fn(cur_ids, cur_mask, cur_labels)
    return selected
