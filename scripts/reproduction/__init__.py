"""Reproduction of gkag145 (NAR 2026) deep batch Bayesian optimization.

Modules: tokenizer (3-mer), encoder (BERT-like transformer), pretrain
(MLM + triplet), surrogate (MLP head + deep ensemble), bo (UCB + Kriging
Believer), and data loading for the FAD aptamer GBSA dataset.
"""
