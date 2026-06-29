import pathlib, sys
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analyses.p5_utility_controller import soft_pref, pairwise_loss


def test_soft_pref_follows_best_order():
    # two candidates; the much-lower-NLL one dominates -> P matches its order
    sA = np.array([0, 1, 2] + list(range(3, 64)))     # i before j ascending
    sB = sA[::-1].copy()
    P = soft_pref([sA, sB], [0.1, 10.0], T=0.5)        # sA hugely better
    assert P[0, 1] > 0.9 and P[1, 0] < 0.1             # 0 before 1 (per sA)


def test_pairwise_loss_lower_when_scores_match_teacher():
    P = np.full((64, 64), 0.5); P[0, 1] = 1.0; P[1, 0] = 0.0
    z_good = torch.zeros(64); z_good[0] = 5.0          # 0 scored well above 1
    z_bad = torch.zeros(64); z_bad[1] = 5.0
    assert float(pairwise_loss(z_good, P)) < float(pairwise_loss(z_bad, P))
