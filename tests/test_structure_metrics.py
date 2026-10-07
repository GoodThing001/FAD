import numpy as np
import pytest

from scripts.reproduction.structure_metrics import target_structure_metrics


def test_unpaired_diagonal_is_not_one_minus_row_sum():
    # Target: positions 1/2 are paired, 3 is unpaired. Every row sums to 1.
    p = np.array([[.2, .8, 0], [.8, .2, 0], [0, 0, 1.]])
    defect, probabilities = target_structure_metrics(p, [1, 0, -1], [1, 3])
    assert defect == pytest.approx(.4)
    assert probabilities == pytest.approx([.8, 1.])


def test_exact_target_has_zero_defect():
    p = np.array([[0, 1., 0], [1., 0, 0], [0, 0, 1.]])
    assert target_structure_metrics(p, [1, 0, -1], [1, 2, 3])[0] == 0


def test_all_unpaired_target_uses_all_diagonal_entries():
    p = np.array([[.25, .75], [.75, .25]])
    assert target_structure_metrics(p, [-1, -1], [1, 2])[0] == pytest.approx(1.5)
