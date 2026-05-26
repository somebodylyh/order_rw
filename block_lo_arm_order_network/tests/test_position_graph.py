import numpy as np, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from position_graph import text_position_graph, image_manhattan_graph


def test_text_position_graph_decays_with_index_distance():
    B = text_position_graph(N=4, tau=1.0)
    assert B.shape == (4, 4)
    assert np.allclose(np.diag(B), 0.0)
    assert np.allclose(B, B.T)                       # symmetric
    assert B[0, 1] > B[0, 2] > B[0, 3]               # monotone decay in |u-v|
    assert np.isclose(B[0, 1], np.exp(-1.0))


def test_image_manhattan_graph_uses_grid_distance():
    B = image_manhattan_graph(side=8, tau=1.0)       # 64 patches, 8x8 raster
    assert B.shape == (64, 64)
    assert np.allclose(np.diag(B), 0.0)
    assert np.allclose(B, B.T)
    # patch 0=(0,0), 1=(0,1) manh 1; 8=(1,0) manh 1; 9=(1,1) manh 2
    assert np.isclose(B[0, 1], np.exp(-1.0))
    assert np.isclose(B[0, 8], np.exp(-1.0))
    assert np.isclose(B[0, 9], np.exp(-2.0))
