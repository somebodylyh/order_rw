import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, "analyses")
from analyze_handoff_circuit import load_trajectory, handoff_timing


def _write_step(root, step, tau, comp01):
    d = root / "raw" / f"step_{step:06d}"
    d.mkdir(parents=True)
    np.savez(d / "tau_table.npz", tau=tau, phys0_rank=np.zeros_like(tau, dtype=np.int64),
             prefix8=np.zeros_like(tau, dtype=np.int64),
             methods=np.array(["C-D+L", "L"]), bs_mean=np.int64(16))
    np.savez(d / "composition.npz", **{"0_1": comp01})


def test_load_and_timing(tmp_path):
    L, H, M = 2, 2, 2
    # step 0: L0 head0 carries; step 200: L1 head0 carries; comp rises
    tau0 = np.zeros((L, H, M)); tau0[0, 0] = 1.0
    tau1 = np.zeros((L, H, M)); tau1[1, 0] = 1.0
    _write_step(tmp_path, 0, tau0, np.zeros((H, H, 3)))
    _write_step(tmp_path, 200, tau1, np.ones((H, H, 3)))
    traj = load_trajectory(tmp_path)
    assert traj["tau"].shape == (2, L, H, M)
    assert list(traj["steps"]) == [0, 200]
    t = handoff_timing(traj)
    # layer 0 peaks earlier than layer 1
    assert t["layer_peak_step"][0] <= t["layer_peak_step"][1]
