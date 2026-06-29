import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from extract_real_attention import extract_attention_for_sequence


class FakeYchModel:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        idx,
        mode,
        orders,
        return_probe_data,
        return_last_attention=False,
        return_all_attentions=False,
    ):
        self.calls.append(
            {
                "return_last_attention": return_last_attention,
                "return_all_attentions": return_all_attentions,
            }
        )
        T = idx.shape[1] + 1
        all_attentions = [
            torch.full((1, 2, T, T), 0.2),
            torch.full((1, 2, T, T), 0.4),
            torch.full((1, 2, T, T), 0.6),
        ]
        return None, None, {"all_attentions": all_attentions}


def test_ych_extraction_requests_all_layer_attentions():
    model = FakeYchModel()
    block_seqs = torch.arange(4).view(2, 2)

    A, _ = extract_attention_for_sequence(
        model,
        block_seqs,
        device="cpu",
        model_source="ych",
        num_blocks=2,
        M=1,
    )

    assert model.calls == [
        {"return_last_attention": False, "return_all_attentions": True}
    ]
    assert A.shape == (2, 2)
