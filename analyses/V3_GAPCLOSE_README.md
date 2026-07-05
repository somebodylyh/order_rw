# V3 gap-closing experiment — run guide (portable)

The **naive gap-closing experiment**: continue from the same aligned gβ parent
checkpoint and change **only** whether gβ is updated —
`frozen_gbeta` (control) vs `joint_group` (unfrozen, PG on own-order LM loss).

Headline metric = `own_over_l2r = L2R_transfer_NLL − own_order_NLL` (both under
**physical/text L2R**), i.e. does unfreezing gβ close the persistent deployment
gap. Guards: `tau_to_l2r` (physical frame — revert-to-L2R?), `tau_to_init`
(policy diffusion), entropy / adv_std / score_norm.

## Files to copy to the other machine (NOT in git — large)

- **20k aligned parent**: `block_lo_arm_order_network/probe_results/gbeta_K1000_from10k_L1H7_seed123/ckpt_step20000.pt`
- **gβ readout ckpt** (`GBETA_CKPT`): `reports/uniform_label_free_v1/nodewise_K1000.pt`
- WikiText103 token bins referenced by the parent ckpt's `args` (`train_bin`/`val_bin`).

The drivers default to repo-relative paths; keep the same relative layout or pass
`--ckpt`.

## Wiring smoke first (CPU, minutes)

```
python analyses/v3_gap_closing_smoke.py --n-steps 3 --batch-size 32
```
Asserts PG reaches gβ, PG stays off backbone, metrics logged, no NaN.

## Full joint run (GPU)

Pilot horizon 20k→25k (what we ran):
```
CUDA_VISIBLE_DEVICES=0 python analyses/v3_gapclose_pilot.py \
    --n-steps 5000 --eval-every 1000 --device cuda --seed 0
```

**Full decisive run** = 20k→60k (`--n-steps 40000`) × multiple seeds:
```
for s in 0 1 2; do
  CUDA_VISIBLE_DEVICES=0 python analyses/v3_gapclose_pilot.py \
      --n-steps 40000 --eval-every 2000 --device cuda --seed $s
done
```
Conservative naive-PG hyperparams (defaults): `lr_orderhead=5e-5`, `beta=0`
(no entropy pressure), `tau=1.0`, EMA baseline + adv clip. Each seed writes to
`runs/v3_gapclose_pilot_20k25k_lr5e5_noent_seed<s>/pilot_report.json`.

## Read results

`pilot_report.json` holds the per-eval `frozen_own_over_l2r` / `joint_own_over_l2r`
trajectories + `gap_closed_vs_frozen`. Per-arm `evals` also carry `tau_to_l2r`
(physical), `tau_to_init`, entropy.
