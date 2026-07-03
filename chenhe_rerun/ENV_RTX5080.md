# RTX 5080 Environment

This environment note is for the cleaned WikiText103 / language-only workflow.

RTX 5080 is Blackwell (`sm_120`) and should use a PyTorch build with CUDA 12.8
or newer. Do not copy the old RTX 4090/X1 environment verbatim.

## System Requirements

- Linux is recommended.
- NVIDIA driver: recent 570+ driver.
- A system CUDA Toolkit is not required for normal PyTorch training if the
  PyTorch CUDA wheel is installed.
- Python: 3.11 is recommended.

## Create Environment

```bash
conda create -n aogpt5080 python=3.11 -y
conda activate aogpt5080
python -m pip install -U pip setuptools wheel
```

Install PyTorch CUDA first:

```bash
python -m pip install \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

Install project dependencies:

```bash
python -m pip install \
  numpy==1.26.4 \
  wandb matplotlib tqdm \
  datasets tiktoken \
  pandas scipy seaborn PyYAML packaging pytest
```

## Verify CUDA And Blackwell Support

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("arch list:", torch.cuda.get_arch_list())
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (12, 0)
assert any("120" in arch for arch in torch.cuda.get_arch_list())
PY
```

## Project Assets

Keep or regenerate:

```text
data/wikitext103
config/WikiText103
Report/language/wikitext103
```

## Smoke Tests

```bash
python tests/test_aogpt_random_orders.py
python tests/test_aogpt_attention_export.py
python train.py config/WikiText103/seq256/permute/block64/random.py \
  --max_iters=1 --eval_interval=1 --eval_iters=1 --wandb_log=False --compile=False
```
