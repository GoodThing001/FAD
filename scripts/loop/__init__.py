"""Candidate closed-loop framework (mutate -> surrogate -> select -> label).

Interface contract v0.1.0. Target = standalone GBSA (the weighted
0.961*Gap2 + 0.039*gbsa label is intentionally not used).

See docs/项目记录/GBSA代理接口与候选闭环实施方案_20260930.md for the
current implementation plan, and
scripts/loop/interface.py for the policy decision record.
"""

from .interface import (INTERFACE_VERSION, MUT_POS, SEQ_LEN, TARGET_COLUMN,
                        Prediction, TargetSpec, WT_141)

__all__ = ["INTERFACE_VERSION", "MUT_POS", "SEQ_LEN", "TARGET_COLUMN",
           "Prediction", "TargetSpec", "WT_141"]
