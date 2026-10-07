"""3-mer tokenizer reproducing gkag145 (NAR 2026) / DNABERT-style tokenization.

The paper tokenizes each construct by overlapping 3-mer tokens over the RNA
alphabet {A, C, G, U} and adds [CLS]/[SEP]/[PAD] (plus [MASK] for MLM).
Sequence length L (bases) maps to L - 2 3-mer tokens, plus [CLS] and [SEP].
"""

from __future__ import annotations

from itertools import product

BASES = "ACGU"
SPECIAL_TOKENS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]

# 4^3 = 64 3-mers, lexicographic order.
_MERS = ["".join(t) for t in product(BASES, repeat=3)]

# token -> id
VOCAB = {tok: i for i, tok in enumerate(SPECIAL_TOKENS + _MERS)}
ID_TO_TOKEN = {i: tok for tok, i in VOCAB.items()}

PAD_ID = VOCAB["[PAD]"]
CLS_ID = VOCAB["[CLS]"]
SEP_ID = VOCAB["[SEP]"]
MASK_ID = VOCAB["[MASK]"]
UNK_ID = VOCAB["[UNK]"]

SPECIAL_IDS = {VOCAB[t] for t in SPECIAL_TOKENS}
MASKABLE_EXCLUDE = {PAD_ID, CLS_ID, SEP_ID, MASK_ID, UNK_ID}

VOCAB_SIZE = len(VOCAB)


def normalize(seq: str) -> str:
    """Uppercase and treat T as U (RNA)."""
    return str(seq).upper().replace("T", "U")


def tokenize(seq: str, add_special: bool = True) -> list[int]:
    """Convert an RNA sequence to overlapping 3-mer token ids.

    add_special=True prepends [CLS] and appends [SEP], matching the paper's
    sequence representation for the encoder (embedding read from [CLS]).
    """
    seq = normalize(seq)
    n = len(seq)
    ids = []
    for i in range(n - 2):  # overlapping 3-mers, stride 1
        mer = seq[i : i + 3]
        ids.append(VOCAB.get(mer, UNK_ID))
    if add_special:
        ids = [CLS_ID] + ids + [SEP_ID]
    return ids


def detokenize(ids: list[int]) -> str:
    """Best-effort inverse for logging (special tokens as <...>)."""
    return " ".join(ID_TO_TOKEN.get(i, "<unk>") for i in ids)
