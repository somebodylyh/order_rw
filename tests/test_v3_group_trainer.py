import json
import pathlib
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "block_lo_arm_order_network")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


class _Perm:
    block_perm_phys_to_model = torch.tensor([2, 0, 3, 1])
    inv_perm_model_to_phys = torch.tensor([1, 3, 0, 2])


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.seen_orders = []
        self.forward_count = 0
        self.optimizer = None

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        class CountingAdamW(torch.optim.AdamW):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.step_calls = 0

            def step(self, *args, **kwargs):
                self.step_calls += 1
                return super().step(*args, **kwargs)

        self.optimizer = CountingAdamW(
            [{"params": [self.weight], "weight_decay": weight_decay}],
            lr=learning_rate,
            betas=betas,
        )
        return self.optimizer

    def forward_fn(self, idx, orders, return_token_loss=False):
        assert return_token_loss
        assert self.training
        self.forward_count += 1
        self.seen_orders.append(orders.detach().cpu().clone())
        token_losses = (self.weight - 0.25).square().expand(idx.shape[0], idx.shape[1])
        return None, token_losses.mean(), token_losses


class _GBeta(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.tensor([4.0, 1.0, 3.0, 2.0]))


class _OrderHead(torch.nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.gbeta = _GBeta()
        self.group_means = []

    def scores(self, A, per_sample):
        assert per_sample is False
        mean = A.mean()
        self.group_means.append(float(mean))
        modulation = mean * torch.tensor([1.0, 0.0, 0.0, 0.0])
        return self.gbeta.scores.unsqueeze(0) + modulation.unsqueeze(0)


class _Wrap(torch.nn.Module):
    last = None

    def __init__(self, backbone, order_head, clean_perm, device="cpu"):
        super().__init__()
        self.backbone = backbone
        self.order_head = order_head
        self.clean_perm = clean_perm
        self.heldout_chunks = None
        self.extract_calls = 0
        _Wrap.last = self

    def extract_B(self, idx_batch, probe):
        self.extract_calls += 1
        self.backbone.eval()  # mirrors the production attention extractor
        values = torch.arange(idx_batch.shape[0], dtype=torch.float32).view(-1, 1, 1)
        return values.expand(-1, 4, 4)

    def token_orders_from_model_blocks(self, order_model_bn):
        # N=4, block_len=1 equivalent of MODEL -> PHYSICAL -> MODEL-token.
        model = torch.as_tensor(order_model_bn, dtype=torch.long)
        phys = self.clean_perm.inv_perm_model_to_phys[model]
        return self.clean_perm.block_perm_phys_to_model[phys]


class _Source:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.heldout = torch.full((batch_size, 4), 99, dtype=torch.long)
        self.calls = []

    def batch(self, global_step, micro_step):
        self.calls.append((global_step, micro_step))
        return (
            torch.arange(self.batch_size * 4).reshape(self.batch_size, 4)
            + 10 + global_step + micro_step
        )


def _patch_training(monkeypatch, module, batch_size):
    model = _Backbone()
    source = _Source(batch_size)
    meta = {
        "args": {
            "lr": 1e-3,
            "min_lr": 1e-4,
            "lr_decay_steps": 50000,
            "warmup_iters": 0,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.99,
            "grad_clip": 1.0,
            "batch_size": batch_size,
            "grad_accum": 2,
            "seed": 123,
        },
        "global_step": 10000,
        "optimizer": None,
    }
    monkeypatch.setattr(module, "N", 4)
    monkeypatch.setattr(module, "BLOCK_LEN", 1)
    monkeypatch.setattr(module, "OrderHeadModule", _OrderHead)
    monkeypatch.setattr(module, "AOGPTWithOrderHead", _Wrap)
    monkeypatch.setattr(
        module,
        "_load_training_context",
        lambda *_args, **_kwargs: (model, source, _Perm(), torch.device("cpu"), meta),
    )
    monkeypatch.setattr(
        module,
        "random_probe_token_orders",
        lambda b, seed, global_step, device: torch.zeros((b, 4), dtype=torch.long),
    )
    return model, source


def _patch_training_factory(monkeypatch, module, batch_size, backbone_cls=_Backbone):
    models, sources = [], []
    meta = {
        "args": {
            "lr": 1e-3, "min_lr": 1e-4, "lr_decay_steps": 50000,
            "warmup_iters": 0, "weight_decay": 0.1, "beta1": 0.9,
            "beta2": 0.99, "grad_clip": 1.0, "batch_size": batch_size,
            "grad_accum": 2, "seed": 123,
        },
        "global_step": 10000,
        "optimizer": None,
    }

    def loader(*_args, **_kwargs):
        model, source = backbone_cls(), _Source(batch_size)
        models.append(model); sources.append(source)
        return model, source, _Perm(), torch.device("cpu"), dict(meta)

    monkeypatch.setattr(module, "N", 4)
    monkeypatch.setattr(module, "BLOCK_LEN", 1)
    monkeypatch.setattr(module, "OrderHeadModule", _OrderHead)
    monkeypatch.setattr(module, "AOGPTWithOrderHead", _Wrap)
    monkeypatch.setattr(module, "_load_training_context", loader)
    monkeypatch.setattr(
        module, "random_probe_token_orders",
        lambda b, seed, global_step, device: torch.zeros((b, 4), dtype=torch.long),
    )
    return models, sources


def test_module_imports():
    import analyses.v3_group_trainer  # noqa: F401


def test_token_loss_adapter_supports_new_and_probe_contracts():
    from analyses.v3_group_trainer import _forward_with_token_losses

    idx = torch.zeros((2, 3), dtype=torch.long)
    order = torch.arange(3).repeat(2, 1)

    class New:
        def forward_fn(self, idx, order, return_token_loss=False):
            assert return_token_loss
            token = torch.full(idx.shape, 2.0)
            return None, token.mean(), token

    class Probe:
        def forward_fn(self, idx, order, return_probe_data=False):
            assert return_probe_data
            token = torch.full(idx.shape, 3.0)
            return None, token.mean(), {"loss_per_step": token}

    for model, expected_contract, expected in (
        (New(), "return_token_loss", 2.0),
        (Probe(), "return_probe_data.loss_per_step", 3.0),
    ):
        loss, token, contract = _forward_with_token_losses(model, idx, order)
        assert contract == expected_contract
        assert loss.item() == expected
        assert torch.all(token == expected)


def test_token_loss_adapter_exact_production_fallback_and_explicit_failure():
    from analyses.v3_group_trainer import _forward_with_token_losses
    from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT, AOGPTConfig

    model = AOGPT(AOGPTConfig(
        block_size=4, vocab_size=16, n_layer=1, n_head=2, n_embd=16,
        dropout=0.0, bias=False,
    ))
    idx = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    order = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    loss, token, contract = _forward_with_token_losses(model, idx, order)
    assert contract == "derived_from_logits_and_model.shuffle"
    assert token.shape == idx.shape
    assert torch.allclose(loss, token.mean(), rtol=1e-6, atol=1e-7)

    class Unsupported:
        def forward_fn(self, idx, order):
            return None, torch.tensor(0.0)

    with pytest.raises(TypeError, match="no shuffle"):
        _forward_with_token_losses(Unsupported(), idx, order)


def test_frozen_gbeta_is_deterministic_argsort_and_never_samples(monkeypatch, tmp_path):
    import analyses.v3_group_trainer as trainer

    model, _ = _patch_training(monkeypatch, trainer, batch_size=4)
    monkeypatch.setattr(
        trainer, "_sample_pl_order", lambda *_args, **_kwargs: pytest.fail("frozen arm sampled PL")
    )
    res = trainer.train_arm(
        "unused.pt", "frozen_gbeta", n_steps=1, batch_size=4, m=4,
        out_dir=tmp_path, tag="tag",
    )
    assert model.seen_orders[0][0].tolist() == [0, 2, 3, 1]
    assert res["orderhead_param_delta"] == 0.0


def test_joint_arm_passes_tau_and_routes_gradients(monkeypatch, tmp_path):
    import analyses.v3_group_trainer as trainer

    _patch_training(monkeypatch, trainer, batch_size=4)
    seen_tau = []
    seen_first_score = []

    def fake_sample(scores, tau):
        seen_tau.append(tau)
        seen_first_score.append(float(scores[0]))
        dist = torch.distributions.Categorical(logits=scores / tau)
        # A differentiable log-prob/entropy with a deterministic legal order.
        order = torch.argsort(scores.detach(), descending=True)
        return order, dist.log_prob(torch.tensor(0)), dist.entropy()

    monkeypatch.setattr(trainer, "_sample_pl_order", fake_sample)
    res = trainer.train_arm(
        "unused.pt", "joint_group", n_steps=2, batch_size=4, m=2, tau=0.37,
        out_dir=tmp_path, tag="tag",
    )
    assert seen_tau == [0.37] * 8
    # Group-means are 0.5 / 2.5 with an initial score delta of ~2.0.
    # Parameter updates cause slight drift; verify the pattern, not exact values.
    for left, right in zip(seen_first_score[::2], seen_first_score[1::2]):
        assert right - left == pytest.approx(2.0, abs=0.1)
    assert res["pg_only_backbone_grad"] == 0.0
    assert res["orderhead_param_delta"] > 0.0
    assert res["backbone_param_delta"] > 0.0
    assert set(res["log"][0]) >= {
        "lm_loss", "pg", "entropy", "orderhead_grad_norm", "backbone_grad_norm"
    }
    assert res["log"][0]["orderhead_grad_norm"] > 0.0
    assert res["log"][0]["backbone_grad_norm"] > 0.0


def test_exact_sampler_and_grad_accum_microstep_semantics(monkeypatch, tmp_path):
    import analyses.v3_group_trainer as trainer
    from clean_training_protocol import sample_stream_batch

    stream = np.arange(200, dtype=np.uint16)
    gather = torch.tensor([3, 2, 1, 0])
    source = trainer._ContinuousBatchSource(
        stream=stream, gather=gather, heldout=torch.empty((0, 4), dtype=torch.long),
        batch_size=2, block_size=4, seed=9, start_step=0,
    )
    expected = sample_stream_batch(stream, 2, 4, 9, 7, 1)[:, gather]
    assert torch.equal(source.batch(7, 1), expected)

    model, fake_source = _patch_training(monkeypatch, trainer, batch_size=2)
    res = trainer.train_arm(
        "unused.pt", "l2r", n_steps=1, batch_size=2,
        out_dir=tmp_path, tag="accum",
    )
    assert fake_source.calls == [(10000, 0), (10000, 1)]
    assert model.forward_count == 2
    assert model.optimizer.step_calls == 1
    assert len(res["log"]) == 1
    assert res["grad_accum"] == 2


def test_one_attention_extraction_per_microbatch_not_per_group(monkeypatch, tmp_path):
    import analyses.v3_group_trainer as trainer

    _patch_training(monkeypatch, trainer, batch_size=4)
    res = trainer.train_arm(
        "unused.pt", "joint_group", n_steps=1, batch_size=4, m=2,
        out_dir=tmp_path, tag="extract-once",
    )
    assert _Wrap.last.extract_calls == 2  # checkpoint grad_accum=2
    assert _Wrap.last.order_head.group_means == [0.5, 2.5, 0.5, 2.5]
    assert len(res["log"]) == 1


def test_l2r_uses_physical_frame_conversion(monkeypatch, tmp_path):
    import analyses.v3_group_trainer as trainer

    model, _ = _patch_training(monkeypatch, trainer, batch_size=2)
    trainer.train_arm(
        "unused.pt", "l2r", n_steps=1, batch_size=2, out_dir=tmp_path, tag="tag",
    )
    # physical [0,1,2,3] -> model blocks [2,0,3,1], block_len=1
    assert model.seen_orders[0][0].tolist() == [2, 0, 3, 1]


def test_invalid_grouping_fails_before_training(monkeypatch, tmp_path):
    import analyses.v3_group_trainer as trainer

    _patch_training(monkeypatch, trainer, batch_size=5)
    with pytest.raises(ValueError, match="divisible"):
        trainer.train_arm(
            "unused.pt", "joint_group", n_steps=1, batch_size=5, m=2,
            out_dir=tmp_path, tag="tag",
        )


def test_checkpoint_batch_conflict_and_optimizer_restore_fail_closed(monkeypatch, tmp_path):
    import analyses.v3_group_trainer as trainer

    _patch_training(monkeypatch, trainer, batch_size=2)
    with pytest.raises(ValueError, match="conflicts with checkpoint"):
        trainer.train_arm(
            "unused.pt", "l2r", n_steps=1, batch_size=4,
            out_dir=tmp_path, tag="mismatch",
        )

    model = _Backbone()
    with pytest.raises(RuntimeError, match="exact restore failed"):
        trainer._build_optimizer(
            model, None, False,
            {"args": {}, "optimizer": {"invalid": "state"}},
            None, 3e-4, torch.device("cpu"),
        )


def test_evaluator_gets_eval_mode_and_training_mode_is_restored(monkeypatch, tmp_path):
    import analyses.v3_group_trainer as trainer

    model, _ = _patch_training(monkeypatch, trainer, batch_size=2)
    calls = []

    def evaluator(step, callback_model, wrap):
        calls.append((step, callback_model.training, wrap.heldout_chunks.clone()))
        return {"metric": 1.25}

    res = trainer.train_arm(
        "unused.pt", "l2r", n_steps=1, batch_size=2,
        out_dir=tmp_path, tag="tag", eval_steps=(1,), evaluator=evaluator,
    )
    assert calls[0][0:2] == (1, False)
    assert torch.all(calls[0][2] == 99)
    assert model.training is True
    assert res["evals"] == [{"step": 1, "metric": 1.25}]
    assert (tmp_path / "tag" / "l2r" / "train.json").exists()


def test_eval_mode_is_restored_even_if_evaluator_raises(monkeypatch, tmp_path):
    import analyses.v3_group_trainer as trainer

    model, _ = _patch_training(monkeypatch, trainer, batch_size=2)

    def evaluator(*_args):
        raise RuntimeError("probe failed")

    with pytest.raises(RuntimeError, match="probe failed"):
        trainer.train_arm(
            "unused.pt", "l2r", n_steps=1, batch_size=2,
            out_dir=tmp_path, tag="tag", eval_steps=(1,), evaluator=evaluator,
        )
    assert model.training is True


def test_interruption_resume_matches_continuous_training(monkeypatch, tmp_path):
    import analyses.v3_group_trainer as trainer

    _patch_training_factory(monkeypatch, trainer, batch_size=4)
    full = trainer.train_arm(
        "unused.pt", "joint_group", n_steps=2, batch_size=4, m=2,
        out_dir=tmp_path / "full", tag="run",
    )
    trainer.train_arm(
        "unused.pt", "joint_group", n_steps=1, batch_size=4, m=2,
        out_dir=tmp_path / "split", tag="run",
    )
    resume_path = tmp_path / "split" / "run" / "joint_group" / "checkpoints" / "latest.pt"
    resumed = trainer.train_arm(
        "unused.pt", "joint_group", n_steps=2, batch_size=4, m=2,
        out_dir=tmp_path / "split", tag="run", resume_from=resume_path,
    )
    assert resumed["log"] == full["log"]
    assert resumed["evals"] == full["evals"]
    full_ckpt = torch.load(
        tmp_path / "full" / "run" / "joint_group" / "checkpoints" / "latest.pt",
        map_location="cpu", weights_only=False,
    )
    resumed_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
    for key, tensor in full_ckpt["backbone"].items():
        assert torch.equal(tensor, resumed_ckpt["backbone"][key])
    for key, tensor in full_ckpt["orderhead"].items():
        assert torch.equal(tensor, resumed_ckpt["orderhead"][key])


def test_nonfinite_stops_before_step_and_writes_finite_report(monkeypatch, tmp_path):
    import analyses.v3_group_trainer as trainer

    class NonFiniteBackbone(_Backbone):
        def forward_fn(self, idx, orders, return_token_loss=False):
            assert return_token_loss
            self.forward_count += 1
            token = (self.weight * 0.0 + torch.tensor(float("nan"))).expand_as(idx)
            return None, token.mean(), token

    _patch_training_factory(
        monkeypatch, trainer, batch_size=2, backbone_cls=NonFiniteBackbone
    )
    res = trainer.train_arm(
        "unused.pt", "l2r", n_steps=1, batch_size=2,
        out_dir=tmp_path, tag="bad",
    )
    assert res["nan"] is True
    assert res["backbone_param_delta"] == 0.0
    assert res["failure"] == {
        "component": "lm_loss", "global_step": 10000,
        "local_step": 0, "micro_step": 0,
    }
    report = tmp_path / "bad" / "l2r" / "failure.json"
    parsed = json.loads(report.read_text())
    assert parsed == res["failure"]
    assert "NaN" not in report.read_text() and "Infinity" not in report.read_text()
    assert (tmp_path / "bad" / "l2r" / "checkpoints" / "latest.pt").exists()
