"""Frozen Pillar-3 carrier sets (from step-10000 tau_table.npz, method C-D+L).

Tiers per (seed, layer) by |tau|:
  strong  |tau| >= 0.95
  weak    0.60 <= |tau| < 0.95
  null    the 3 smallest-|tau| heads, EXCLUDING any weak/strong carrier.

L0 has no strong head in any seed -> it is a weak UPSTREAM SOURCE, never a strong
carrier. seed42 has no strong head anywhere -> natural negative contrast.

These sets are frozen and used as-is by the path-patching interventions; there is
no runtime head selection. ``verify_against_tau_table`` re-derives them from the
real tau table so drift is caught.
"""
import numpy as np

CARRIER_SETS = {
    2: {
        0: {"strong": [],          "weak": [1, 6, 7],    "null": [5, 2, 3]},
        1: {"strong": [0, 3, 5, 7], "weak": [1, 2],       "null": [4, 6]},
        2: {"strong": [],          "weak": [3],          "null": [1, 6, 5]},
        3: {"strong": [],          "weak": [5],          "null": [7, 1, 6]},
    },
    42: {
        0: {"strong": [], "weak": [0, 3, 5, 6, 7], "null": [4, 1, 2]},
        1: {"strong": [], "weak": [2, 6],          "null": [4, 1, 3]},
        2: {"strong": [], "weak": [3, 5],          "null": [7, 0, 6]},
        3: {"strong": [], "weak": [2],             "null": [3, 4, 0]},
    },
    123: {
        0: {"strong": [],          "weak": [6, 7],       "null": [3, 4, 1]},
        1: {"strong": [0, 5, 6, 7], "weak": [1, 4],       "null": [2, 3]},
        2: {"strong": [],          "weak": [4],          "null": [0, 7, 3]},
        3: {"strong": [],          "weak": [0, 1, 3, 4],  "null": [5, 7, 2]},
    },
}


def load_carrier_sets(seed: int) -> dict:
    return CARRIER_SETS[int(seed)]


def verify_against_tau_table(seed, tau_npz_path, strong=0.95, weak_lo=0.60):
    """Re-derive tiers from the real tau table and assert they match the frozen config."""
    d = np.load(tau_npz_path, allow_pickle=True)
    methods = list(d["methods"])
    mi = methods.index("C-D+L")
    tau = d["tau"][:, :, mi]  # (L, H) signed
    cfg = load_carrier_sets(seed)
    for L in range(tau.shape[0]):
        a = np.abs(tau[L])
        exp_strong = sorted(int(h) for h in range(8) if a[h] >= strong)
        exp_weak = sorted(int(h) for h in range(8) if weak_lo <= a[h] < strong)
        assert sorted(cfg[L]["strong"]) == exp_strong, (seed, L, "strong", exp_strong)
        assert sorted(cfg[L]["weak"]) == exp_weak, (seed, L, "weak", exp_weak)
        cand = sorted(
            (h for h in range(8) if h not in exp_strong and h not in exp_weak),
            key=lambda h: a[h],
        )
        assert cfg[L]["null"] == cand[:3], (seed, L, "null", cand[:3])
