import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_wandb_train_payload_contains_loss_alpha_lr_and_step():
    from train_clean_aogpt import wandb_train_payload

    payload = wandb_train_payload(
        step=20001,
        train_loss=4.205864,
        alpha=0.0002,
        lr=6.89057647e-4,
    )

    assert payload == {
        "step": 20001,
        "train_loss": 4.205864,
        "alpha": 0.0002,
        "lr": 6.89057647e-4,
    }


def test_wandb_eval_payload_flattens_required_and_optional_metrics():
    from train_clean_aogpt import wandb_eval_payload

    metrics = {
        "val_train_objective": {"loss_token_avg": 3.482409},
        "val_ori_l2r_block": {"loss_token_avg": 3.332232},
        "val_ar_l2r": {"loss_token_avg": 3.332232},
        "val_model_order": {"loss_token_avg": 4.459385},
        "val_unstructured_order": {"loss_token_avg": 4.414729},
        "val_rw_order": {"loss_token_avg": 3.800303},
        "val_beta_order": {"loss_token_avg": 3.482409},
        "val_direct_order": {"loss_token_avg": 3.400001},
    }

    payload = wandb_eval_payload(
        step=60000,
        metrics=metrics,
        train_loss=3.189834,
        alpha=1.0,
        lr=1e-4,
    )

    assert payload["step"] == 60000
    assert payload["train_loss"] == 3.189834
    assert payload["alpha"] == 1.0
    assert payload["lr"] == 1e-4
    assert payload["val_train_objective"] == 3.482409
    assert payload["val_ori_l2r_block"] == 3.332232
    assert payload["val_ar_l2r"] == 3.332232
    assert payload["val_model_order"] == 4.459385
    assert payload["val_unstructured_order"] == 4.414729
    assert payload["val_rw_order"] == 3.800303
    assert payload["val_beta_order"] == 3.482409
    assert payload["val_direct_order"] == 3.400001
    assert math.isnan(payload["val_cdl_order"])
