# tests/test_pss_mask_norm.py
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.physical_signal_source import valid_edge_mask, row_normalize_l1

def test_valid_edge_mask_structure():
    m = valid_edge_mask(65)
    assert m[0, 1:].all() and not m[:, 0].any()       # None->content yes, into-None no
    assert not np.diag(m).any()                        # no diagonal
    assert m[2, 1] and not m[1, 2]                     # content lower-tri only
    assert m.sum() == 64 + 64*63//2                    # 64 None-edges + content lower-tri

def test_row_normalize_l1_sums_to_one_on_valid():
    B = np.zeros((65, 65)); B[3, 1] = 2.0; B[3, 2] = 2.0
    m = valid_edge_mask(65)
    Bn = row_normalize_l1(B, m)
    assert abs(Bn[3, 1] + Bn[3, 2] - 1.0) < 1e-6       # row 3 valid mass -> 1
    assert Bn[:, 0].sum() == 0                          # invalid zeroed

def test_row_normalize_l1_ignores_nonfinite_invalid_entries_and_uses_absolute_mass():
    m = valid_edge_mask(5)
    B = np.zeros((5, 5))
    B[3, 1] = -2.0
    B[3, 2] = 1.0
    B[3, 3] = np.nan       # invalid diagonal must not contaminate row mass
    B[3, 4] = np.inf       # invalid upper-triangle entry must be zeroed

    Bn = row_normalize_l1(B, m)

    np.testing.assert_allclose(Bn[3, [1, 2]], [-2.0 / 3.0, 1.0 / 3.0])
    assert np.all(Bn[~m] == 0.0)
    assert np.isfinite(Bn).all()

def test_row_normalize_l1_keeps_zero_valid_mass_rows_zero():
    m = valid_edge_mask(5)
    B = np.zeros((5, 5))
    B[2, 2] = np.nan
    B[2, 4] = -np.inf

    Bn = row_normalize_l1(B, m)

    np.testing.assert_array_equal(Bn[2], np.zeros(5))
