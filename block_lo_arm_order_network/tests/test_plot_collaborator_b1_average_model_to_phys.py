import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_collaborator_b1_average_model_to_phys import _resolve_perm_orientation


def test_resolve_perm_orientation_uses_clean_convention():
    ckpt = {"data_permutation": {"convention": "clean_phys_to_model"}}

    assert _resolve_perm_orientation(ckpt, "auto") == "phys_to_model"


def test_resolve_perm_orientation_preserves_explicit_choice():
    ckpt = {"data_permutation": {"convention": "clean_phys_to_model"}}

    assert _resolve_perm_orientation(ckpt, "model_to_phys") == "model_to_phys"


def test_resolve_perm_orientation_defaults_to_collaborator_orientation():
    ckpt = {"data_permutation": {}}

    assert _resolve_perm_orientation(ckpt, "auto") == "model_to_phys"
