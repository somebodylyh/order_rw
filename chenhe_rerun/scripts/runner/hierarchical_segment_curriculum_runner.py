"""
Purpose:
Run a stage-by-stage curriculum where each benchmark stage folds the current
top-k non-conflicting unit pairs, locks those merged units, and uses the
result as the unit vocabulary for the next stage.

Typical usage:
python scripts/runner/hierarchical_segment_curriculum_runner.py \
  config/WikiText103/seq256/permute/block1/segment_curriculum.py
"""

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_csv_list(raw_value, cast_fn):
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    return [cast_fn(item) for item in values]


def parse_optional_csv_list(raw_value, cast_fn):
    if raw_value is None or not str(raw_value).strip():
        return []
    return parse_csv_list(raw_value, cast_fn)


def parse_optional_bool(raw_value):
    if raw_value is None or not str(raw_value).strip():
        return None
    value = str(raw_value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {raw_value!r}")


def stage_value(stage_values, fallback_value, stage_idx):
    if stage_values:
        return stage_values[int(stage_idx) - 1]
    return fallback_value


def run_command(cmd, cwd):
    print("[runner] exec:")
    print("  " + " ".join(shlex.quote(str(part)) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def count_final_units(results_json):
    results_path = Path(results_json)
    with open(results_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    final_units = payload.get("final_units", [])
    if final_units is None:
        return 0
    return len(final_units)


def results_json_complete(results_json):
    try:
        return count_final_units(results_json) > 0
    except Exception:
        return False


def write_frozen_policy_results(source_results, target_results, stage_idx, ratio, threshold):
    source_path = Path(source_results)
    target_path = Path(target_results)
    if not results_json_complete(source_path):
        raise ValueError(f"cannot freeze policy from incomplete results: {source_path}")
    with open(source_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    freeze_meta = {
        "enabled": True,
        "stage": int(stage_idx),
        "segment_guided_ratio": float(ratio),
        "threshold": float(threshold),
        "source_results": str(source_path),
        "note": (
            "Policy recovery was skipped because segment_guided_ratio reached "
            "the full-guidance threshold; this stage reuses the previous policy."
        ),
    }
    payload["runner_policy_freeze"] = freeze_meta
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (target_path.parent / "policy_freeze_meta.json").write_text(
        json.dumps(freeze_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return freeze_meta


def write_no_recovery_results(target_results, stage_idx, ratio, num_blocks=0):
    target_path = Path(target_results)
    num_blocks = int(num_blocks)
    if num_blocks > 0:
        final_units = [{"segment": [int(idx)]} for idx in range(num_blocks)]
    else:
        final_units = [{"segment": [0]}]
    payload = {
        "meta": {
            "stage": int(stage_idx),
            "segment_guided_ratio": float(ratio),
            "curriculum_recovery_mode": "none",
            "note": (
                "No recovery was run for this stage. The results file only marks "
                "the pause/resume boundary; train.py receives segment_guided_ratio=0 "
                "in the intended random-only ablation."
            ),
        },
        "final_units": final_units,
        "top_pairs": [],
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload["meta"]


def checkpoint_iter(ckpt_path):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return None
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    value = checkpoint.get("iter_num")
    if value is None:
        return None
    return int(value)


def load_runner_config_from_argv(argv):
    config_ns = {}
    filtered = []
    config_path = None
    for arg in argv:
        if "=" not in arg and not arg.startswith("--") and config_path is None:
            config_path = Path(arg)
        else:
            filtered.append(arg)
    if config_path is not None:
        if not config_path.is_absolute():
            config_path = REPO_ROOT / config_path
        print(f"Overriding runner config with {config_path}:")
        with open(config_path, "r", encoding="utf-8") as handle:
            print(handle.read())
        exec(config_path.read_text(encoding="utf-8"), config_ns)
    return config_ns, filtered


def build_curriculum_wandb_run_id(base_name, train_out_dir, config_path):
    seed = f"{base_name}|{Path(train_out_dir)}|{Path(config_path)}"
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]
    return f"hcurr-{digest}"


def build_train_cmd(
    repo_dir,
    config_path,
    out_dir,
    init_from,
    max_iters,
    segment_guided_ratio,
    segment_source_json,
    segment_top_k_pairs,
    segment_max_len,
    segment_max_units_per_order,
    segment_lock_final_units=True,
    wandb_project=None,
    wandb_run_name=None,
    wandb_run_id=None,
    train_eval_interval=-1,
    train_eval_iters=-1,
    train_wandb_log=None,
    online_pair_stats_enabled=False,
    online_pair_stats_out_dir="",
    online_pair_stats_write_every=250,
    online_pair_stats_min_count=1,
    online_pair_stats_top_k_export=256,
    online_attn_probe_enabled=False,
    online_attn_probe_interval=0,
    online_attn_probe_batch_size=8,
    online_attn_probe_top_k=4,
    online_attn_probe_ema_decay=0.95,
    online_spectral_enabled=False,
    online_spectral_mode="cache_exact",
    online_spectral_out_dir="",
    online_spectral_interval=0,
    online_spectral_batch_size=8,
    online_spectral_max_updates=0,
    online_spectral_write_every=1,
    online_spectral_export_type="with_none",
    online_spectral_ema_decay=0.95,
    online_spectral_subspace_rank=8,
    online_spectral_subspace_steps_per_update=1,
    online_spectral_subspace_seed=12345,
    online_spectral_threshold_percentile=60.0,
    online_spectral_transform="relu",
    online_spectral_temperature=1.0,
):
    cmd = [
        sys.executable,
        "train.py",
        str(config_path),
        f"--out_dir={out_dir}",
        f"--init_from={init_from}",
        f"--max_iters={int(max_iters)}",
        f"--segment_guided_ratio={float(segment_guided_ratio)}",
        f"--segment_top_k_pairs={int(segment_top_k_pairs)}",
        f"--segment_max_len={int(segment_max_len)}",
        f"--segment_max_units_per_order={int(segment_max_units_per_order)}",
        f"--segment_lock_final_units={bool(segment_lock_final_units)}",
    ]
    if segment_source_json:
        cmd.append(f"--segment_source_json={segment_source_json}")
    if wandb_project:
        cmd.append(f"--wandb_project={wandb_project}")
    if wandb_run_name:
        cmd.append(f"--wandb_run_name={wandb_run_name}")
    if wandb_run_id:
        cmd.append(f"--wandb_run_id={wandb_run_id}")
    if int(train_eval_interval) > 0:
        cmd.append(f"--eval_interval={int(train_eval_interval)}")
    if int(train_eval_iters) > 0:
        cmd.append(f"--eval_iters={int(train_eval_iters)}")
    if train_wandb_log is not None:
        cmd.append(f"--wandb_log={bool(train_wandb_log)}")
    if bool(online_pair_stats_enabled):
        cmd.extend(
            [
                "--online_pair_stats_enabled=True",
                f"--online_pair_stats_out_dir={online_pair_stats_out_dir}",
                f"--online_pair_stats_write_every={int(online_pair_stats_write_every)}",
                f"--online_pair_stats_min_count={int(online_pair_stats_min_count)}",
                f"--online_pair_stats_top_k_export={int(online_pair_stats_top_k_export)}",
            ]
        )
    if bool(online_attn_probe_enabled):
        cmd.extend(
            [
                "--online_attn_probe_enabled=True",
                f"--online_attn_probe_interval={int(online_attn_probe_interval)}",
                f"--online_attn_probe_batch_size={int(online_attn_probe_batch_size)}",
                f"--online_attn_probe_top_k={int(online_attn_probe_top_k)}",
                f"--online_attn_probe_ema_decay={float(online_attn_probe_ema_decay)}",
            ]
        )
    if bool(online_spectral_enabled):
        cmd.extend(
            [
                "--online_spectral_enabled=True",
                f"--online_spectral_mode={str(online_spectral_mode)}",
                f"--online_spectral_out_dir={online_spectral_out_dir}",
                f"--online_spectral_interval={int(online_spectral_interval)}",
                f"--online_spectral_batch_size={int(online_spectral_batch_size)}",
                f"--online_spectral_max_updates={int(online_spectral_max_updates)}",
                f"--online_spectral_write_every={int(online_spectral_write_every)}",
                f"--online_spectral_export_type={str(online_spectral_export_type)}",
                f"--online_spectral_ema_decay={float(online_spectral_ema_decay)}",
                f"--online_spectral_subspace_rank={int(online_spectral_subspace_rank)}",
                f"--online_spectral_subspace_steps_per_update={int(online_spectral_subspace_steps_per_update)}",
                f"--online_spectral_subspace_seed={int(online_spectral_subspace_seed)}",
                f"--online_spectral_threshold_percentile={float(online_spectral_threshold_percentile)}",
                f"--online_spectral_transform={str(online_spectral_transform)}",
                f"--online_spectral_temperature={float(online_spectral_temperature)}",
            ]
        )
    return cmd


def build_benchmark_cmd(
    benchmark_out_dir,
    ckpt_path,
    split,
    batch_size,
    pair_mining_batches,
    pair_eval_batch_size,
    pair_chunk_size,
    forward_eval_batch_size,
    aggregate_top_k_pairs,
    aggregation_margin_threshold,
    aggregation_rank_key,
    pair_aggregation_mode,
    greedy_pair_cover,
    target_next_num_units,
    halve_each_level,
    segment_len,
    pair_score_k,
    pair_score_mode,
    tv_weight,
    drop_weight,
    token_micro_topk,
    token_micro_min_affinity,
    token_micro_max_degree,
    token_micro_max_segment_size,
    token_micro_min_segment_size,
    token_micro_min_density,
    token_micro_num_order_candidates,
    token_micro_eval_batches,
    token_micro_eval_batch_size,
    token_micro_random_suffixes,
    token_micro_random_orders,
    token_micro_min_pair_consistency,
    token_micro_reverse_margin_threshold,
    token_micro_random_margin_threshold,
    token_micro_keep_singletons,
    token_micro_max_accepted_segments,
    num_levels,
    early_stop_enabled,
    early_stop_eval_split,
    early_stop_min_pairs_after_margin,
    early_stop_gain_eval_batches,
    early_stop_gain_eval_batch_size,
    early_stop_gain_random_contexts,
    early_stop_gain_threshold,
    early_stop_gain_p10_threshold,
    early_stop_gain_positive_rate,
    early_stop_min_gain_pass_rate,
    early_stop_filter_segments_by_gain,
    early_stop_policy_eval_batches,
    early_stop_policy_eval_batch_size,
    early_stop_policy_epsilon,
    early_stop_boundary_eval_batches,
    early_stop_boundary_eval_batch_size,
    early_stop_boundary_random_contexts,
    early_stop_boundary_gain_threshold,
    early_stop_boundary_gain_p10_threshold,
    early_stop_boundary_positive_rate,
    early_stop_max_bad_boundary_ratio,
    pair_mining_mode,
    attn_top_k,
    attn_num_batches,
    attn_batch_size,
    attn_mode,
    attn_symmetrize,
    attn_export_type,
    initial_units_json="",
):
    cmd = [
        sys.executable,
        str(SCRIPTS_ROOT / "benchmark" / "hierarchical_structured_benchmark.py"),
        f"--ckpt_path={ckpt_path}",
        f"--out_dir={benchmark_out_dir}",
        f"--split={str(split)}",
        f"--batch_size={int(batch_size)}",
        f"--pair_mining_batches={int(pair_mining_batches)}",
        f"--pair_eval_batch_size={int(pair_eval_batch_size)}",
        f"--pair_chunk_size={int(pair_chunk_size)}",
        f"--forward_eval_batch_size={int(forward_eval_batch_size)}",
        f"--aggregate_top_k_pairs={int(aggregate_top_k_pairs)}",
        f"--aggregation_margin_threshold={float(aggregation_margin_threshold)}",
        f"--aggregation_rank_key={str(aggregation_rank_key)}",
        f"--pair_aggregation_mode={str(pair_aggregation_mode)}",
        f"--segment_len={int(segment_len)}",
        f"--pair_score_k={int(pair_score_k)}",
        f"--pair_score_mode={str(pair_score_mode)}",
        f"--num_levels={int(num_levels)}",
        f"--tv_weight={float(tv_weight)}",
        f"--drop_weight={float(drop_weight)}",
        f"--token_micro_topk={int(token_micro_topk)}",
        f"--token_micro_min_affinity={float(token_micro_min_affinity)}",
        f"--token_micro_max_degree={int(token_micro_max_degree)}",
        f"--token_micro_max_segment_size={int(token_micro_max_segment_size)}",
        f"--token_micro_min_segment_size={int(token_micro_min_segment_size)}",
        f"--token_micro_min_density={float(token_micro_min_density)}",
        f"--token_micro_num_order_candidates={int(token_micro_num_order_candidates)}",
        f"--token_micro_eval_batches={int(token_micro_eval_batches)}",
        f"--token_micro_eval_batch_size={int(token_micro_eval_batch_size)}",
        f"--token_micro_random_suffixes={int(token_micro_random_suffixes)}",
        f"--token_micro_random_orders={int(token_micro_random_orders)}",
        f"--token_micro_min_pair_consistency={float(token_micro_min_pair_consistency)}",
        f"--token_micro_reverse_margin_threshold={float(token_micro_reverse_margin_threshold)}",
        f"--token_micro_random_margin_threshold={float(token_micro_random_margin_threshold)}",
        f"--token_micro_max_accepted_segments={int(token_micro_max_accepted_segments)}",
        f"--pair_mining_mode={str(pair_mining_mode)}",
        f"--early_stop_eval_split={str(early_stop_eval_split)}",
        f"--early_stop_min_pairs_after_margin={int(early_stop_min_pairs_after_margin)}",
        f"--early_stop_gain_eval_batches={int(early_stop_gain_eval_batches)}",
        f"--early_stop_gain_eval_batch_size={int(early_stop_gain_eval_batch_size)}",
        f"--early_stop_gain_random_contexts={int(early_stop_gain_random_contexts)}",
        f"--early_stop_gain_threshold={float(early_stop_gain_threshold)}",
        f"--early_stop_gain_p10_threshold={float(early_stop_gain_p10_threshold)}",
        f"--early_stop_gain_positive_rate={float(early_stop_gain_positive_rate)}",
        f"--early_stop_min_gain_pass_rate={float(early_stop_min_gain_pass_rate)}",
        f"--early_stop_policy_eval_batches={int(early_stop_policy_eval_batches)}",
        f"--early_stop_policy_eval_batch_size={int(early_stop_policy_eval_batch_size)}",
        f"--early_stop_policy_epsilon={float(early_stop_policy_epsilon)}",
        f"--early_stop_boundary_eval_batches={int(early_stop_boundary_eval_batches)}",
        f"--early_stop_boundary_eval_batch_size={int(early_stop_boundary_eval_batch_size)}",
        f"--early_stop_boundary_random_contexts={int(early_stop_boundary_random_contexts)}",
        f"--early_stop_boundary_gain_threshold={float(early_stop_boundary_gain_threshold)}",
        f"--early_stop_boundary_gain_p10_threshold={float(early_stop_boundary_gain_p10_threshold)}",
        f"--early_stop_boundary_positive_rate={float(early_stop_boundary_positive_rate)}",
        f"--early_stop_max_bad_boundary_ratio={float(early_stop_max_bad_boundary_ratio)}",
    ]
    if bool(token_micro_keep_singletons):
        cmd.append("--token_micro_keep_singletons")
    if bool(early_stop_enabled):
        cmd.append("--early_stop_enabled")
    if bool(early_stop_filter_segments_by_gain):
        cmd.append("--early_stop_filter_segments_by_gain")
    if bool(greedy_pair_cover):
        cmd.append("--greedy_pair_cover")
    if int(target_next_num_units) > 0:
        cmd.append(f"--target_next_num_units={int(target_next_num_units)}")
    if bool(halve_each_level):
        cmd.append("--halve_each_level")
    if str(pair_mining_mode) == "attention_pruned":
        cmd.extend(
            [
                f"--attn_top_k={int(attn_top_k)}",
                f"--attn_num_batches={int(attn_num_batches)}",
                f"--attn_batch_size={int(attn_batch_size)}",
                f"--attn_mode={str(attn_mode)}",
                f"--attn_symmetrize={str(attn_symmetrize)}",
                f"--attn_export_type={str(attn_export_type)}",
            ]
        )
    if initial_units_json:
        cmd.append(f"--initial_units_json={initial_units_json}")
    return cmd


def build_attention_spectral_benchmark_cmd(
    benchmark_out_dir,
    ckpt_path,
    split,
    attn_num_batches,
    attn_batch_size,
    attn_mode,
    attn_export_type,
    max_primary_matrices,
    num_components,
    component_pairs,
    num_angles,
    k_values,
    group_methods,
    threshold_percentile,
    transform,
    temperature,
    direction_lambdas,
    directed_score_weight,
    band_quality_weight,
    score_adjacency_sym,
    max_candidates,
    export_top_candidates,
    output_policy,
    rerank_num_candidates,
    rerank_num_batches,
    rerank_batch_size,
    rerank_forward_batch_size,
    rerank_prefix_k,
    rerank_include_reverse,
    rerank_attention_weight,
    rerank_full_loss_weight,
    rerank_prefix_loss_weight,
    rerank_signed_drop_weight,
    rerank_reverse_margin_weight,
    rerank_positive_rate_weight,
    rerank_stability_weight,
    seed,
    device,
    dtype,
    spectral_solver="exact",
    attention_state_npz="",
):
    cmd = [
        sys.executable,
        str(SCRIPTS_ROOT / "benchmark" / "attention_spectral_cross_axis_benchmark.py"),
        f"--ckpt_path={ckpt_path}",
        f"--out_dir={benchmark_out_dir}",
        f"--split={str(split)}",
        f"--attn_num_batches={int(attn_num_batches)}",
        f"--attn_batch_size={int(attn_batch_size)}",
        f"--attn_mode={str(attn_mode)}",
        f"--attn_export_type={str(attn_export_type)}",
        f"--max_primary_matrices={int(max_primary_matrices)}",
        f"--num_components={int(num_components)}",
        f"--component_pairs={str(component_pairs)}",
        f"--num_angles={int(num_angles)}",
        f"--k_values={str(k_values)}",
        f"--group_methods={str(group_methods)}",
        f"--threshold_percentile={float(threshold_percentile)}",
        f"--transform={str(transform)}",
        f"--temperature={float(temperature)}",
        f"--direction_lambdas={str(direction_lambdas)}",
        f"--directed_score_weight={float(directed_score_weight)}",
        f"--band_quality_weight={float(band_quality_weight)}",
        f"--score_adjacency_sym={str(score_adjacency_sym)}",
        f"--max_candidates={int(max_candidates)}",
        f"--export_top_candidates={int(export_top_candidates)}",
        f"--output_policy={str(output_policy)}",
        f"--rerank_num_candidates={int(rerank_num_candidates)}",
        f"--rerank_num_batches={int(rerank_num_batches)}",
        f"--rerank_batch_size={int(rerank_batch_size)}",
        f"--rerank_forward_batch_size={int(rerank_forward_batch_size)}",
        f"--rerank_prefix_k={int(rerank_prefix_k)}",
        f"--rerank_include_reverse={1 if bool(rerank_include_reverse) else 0}",
        f"--rerank_attention_weight={float(rerank_attention_weight)}",
        f"--rerank_full_loss_weight={float(rerank_full_loss_weight)}",
        f"--rerank_prefix_loss_weight={float(rerank_prefix_loss_weight)}",
        f"--rerank_signed_drop_weight={float(rerank_signed_drop_weight)}",
        f"--rerank_reverse_margin_weight={float(rerank_reverse_margin_weight)}",
        f"--rerank_positive_rate_weight={float(rerank_positive_rate_weight)}",
        f"--rerank_stability_weight={float(rerank_stability_weight)}",
        f"--seed={int(seed)}",
        f"--device={str(device)}",
        f"--dtype={str(dtype)}",
    ]
    cmd.append(f"--spectral_solver={str(spectral_solver)}")
    if attention_state_npz:
        cmd.append(f"--attention_state_npz={attention_state_npz}")
    return cmd


def build_online_pair_stats_aggregation_cmd(
    benchmark_out_dir,
    stats_dir,
    extra_stats_dirs,
    num_blocks,
    top_k_pairs,
    min_count,
    margin_threshold,
    positive_rate_threshold,
    max_signed_drop_std,
    drop_weight,
    allow_missing_reverse,
    use_attention_filter,
    aggregation_strategy,
    max_chain_blocks,
    consensus_min_windows,
):
    stats_dir = Path(stats_dir)
    attn_topk_json = stats_dir / "online_attn_topk.json"
    extra_stats_dirs_arg = ";".join(str(Path(item)) for item in extra_stats_dirs)
    return [
        sys.executable,
        str(SCRIPTS_ROOT / "benchmark" / "online_pair_stats_aggregate.py"),
        f"--stats_dir={stats_dir}",
        f"--extra_stats_dirs={extra_stats_dirs_arg}",
        f"--out_dir={benchmark_out_dir}",
        f"--num_blocks={int(num_blocks)}",
        f"--top_k_pairs={int(top_k_pairs)}",
        f"--min_count={int(min_count)}",
        f"--margin_threshold={float(margin_threshold)}",
        f"--positive_rate_threshold={float(positive_rate_threshold)}",
        f"--max_signed_drop_std={float(max_signed_drop_std)}",
        f"--drop_weight={float(drop_weight)}",
        f"--allow_missing_reverse={1 if bool(allow_missing_reverse) else 0}",
        f"--use_attention_filter={1 if bool(use_attention_filter) else 0}",
        f"--attn_topk_json={attn_topk_json}",
        f"--aggregation_strategy={str(aggregation_strategy)}",
        f"--max_chain_blocks={int(max_chain_blocks)}",
        f"--consensus_min_windows={int(consensus_min_windows)}",
    ]


def build_online_relation_tournament_cmd(
    benchmark_out_dir,
    stats_dir,
    extra_stats_dirs,
    previous_results_json,
    top_k_edges,
    min_count,
    margin_threshold,
    positive_rate_threshold,
    max_signed_drop_std,
    allow_missing_reverse,
    max_segment_len,
    consensus_min_windows,
    selected_method,
    history_decay,
    merge_margin_threshold,
    order_adjacent_margin_threshold,
    strong_edge_top_k,
    attention_min_windows,
    attention_require_current,
    attention_symmetric,
):
    stats_dirs = [Path(stats_dir)] + [Path(item) for item in extra_stats_dirs]
    cmd = [
        sys.executable,
        str(SCRIPTS_ROOT / "benchmark" / "online_relation_memory_tournament.py"),
        "--stats_dirs",
        *[str(path) for path in stats_dirs],
        f"--out_dir={benchmark_out_dir}",
        f"--previous_results_json={previous_results_json}",
        "--num_blocks=0",
        f"--top_k_edges={int(top_k_edges)}",
        f"--strong_edge_top_k={int(strong_edge_top_k)}",
        f"--min_count={int(min_count)}",
        f"--positive_rate_threshold={float(positive_rate_threshold)}",
        f"--margin_threshold={float(margin_threshold)}",
        f"--max_signed_drop_std={float(max_signed_drop_std)}",
        f"--allow_missing_reverse={1 if bool(allow_missing_reverse) else 0}",
        f"--history_decay={float(history_decay)}",
        f"--max_segment_len={int(max_segment_len)}",
        f"--consensus_min_windows={int(consensus_min_windows)}",
        f"--merge_margin_threshold={float(merge_margin_threshold)}",
        f"--order_adjacent_margin_threshold={float(order_adjacent_margin_threshold)}",
        "--attention_topk_jsons",
        *[str(path / "online_attn_topk.json") for path in stats_dirs],
        f"--attention_min_windows={int(attention_min_windows)}",
        f"--attention_require_current={1 if bool(attention_require_current) else 0}",
        f"--attention_symmetric={1 if bool(attention_symmetric) else 0}",
        f"--selected_method={str(selected_method)}",
    ]
    return cmd


def parse_optional_stage_text_list(raw_value):
    if raw_value is None or not str(raw_value).strip():
        return []
    return [item.strip() for item in str(raw_value).split(";") if item.strip()]


def parse_args():
    config_ns, filtered_argv = load_runner_config_from_argv(sys.argv[1:])
    parser = argparse.ArgumentParser(
        description=(
            "Hierarchical stage-wise curriculum runner: train random backbone, "
            "benchmark one locked-unit fold level, resume training, then repeat."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(config_ns.get("config", "config/WikiText103/seq256/permute/block1/random.py")),
    )
    parser.add_argument(
        "--train_out_dir",
        type=Path,
        default=Path(
            config_ns.get(
                "train_out_dir",
                "out/curriculum/permute/seq256/block1/out-wikitext103-seq256-random-b1-hcurr",
            )
        ),
    )
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        default=Path(
            config_ns.get(
                "benchmark_root",
                "Report/curriculum/permute/seq256/block1/hierarchical_block1",
            )
        ),
    )
    parser.add_argument("--wandb_project", type=str, default=config_ns.get("wandb_project", None))
    parser.add_argument("--wandb_run_name", type=str, default=config_ns.get("wandb_run_name", None))
    parser.add_argument("--wandb_run_id", type=str, default=str(config_ns.get("wandb_run_id", "")))
    parser.add_argument("--warmup_iters", type=int, default=int(config_ns.get("warmup_iters", 4000)))
    parser.add_argument("--stage_iters", type=int, default=int(config_ns.get("stage_iters", 4000)))
    parser.add_argument("--num_curriculum_stages", type=int, default=int(config_ns.get("num_curriculum_stages", 4)))
    parser.add_argument(
        "--resume_existing",
        action="store_true",
        default=bool(config_ns.get("resume_existing", False)),
        help=(
            "If train_out_dir/ckpt.pt or stage results already exist, reuse them "
            "and continue from the first unfinished curriculum step."
        ),
    )
    parser.add_argument(
        "--run_to_end_after_single_unit",
        action="store_true",
        default=bool(config_ns.get("run_to_end_after_single_unit", True)),
        help=(
            "After a benchmark stage collapses to one final unit, skip later "
            "benchmark/resume interruptions and train once to the final max_iters."
        ),
    )
    parser.add_argument(
        "--segment_guided_ratios",
        type=str,
        default=str(config_ns.get("segment_guided_ratios", "0.3,0.5,0.7,0.9")),
    )
    parser.add_argument(
        "--segment_max_lens",
        type=str,
        default=str(config_ns.get("segment_max_lens", "2,2,2,2")),
        help="Kept for train.py compatibility; pair folds are controlled by pair_aggregation_mode in the benchmark.",
    )
    parser.add_argument("--segment_max_units_per_order", type=int, default=int(config_ns.get("segment_max_units_per_order", 6)))
    parser.add_argument(
        "--segment_lock_final_units",
        type=str,
        default=str(config_ns.get("segment_lock_final_units", "True")),
        help=(
            "When true, train.py treats result JSON final_units as locked and uses all "
            "non-overlapping segments in each guided sample. Set false for relation segment dropout."
        ),
    )
    parser.add_argument("--segment_top_k_pairs", type=int, default=int(config_ns.get("segment_top_k_pairs", 64)))
    parser.add_argument(
        "--freeze_policy_at_full_guidance",
        action="store_true",
        default=bool(config_ns.get("freeze_policy_at_full_guidance", False)),
        help=(
            "When the current stage's segment_guided_ratio is at least "
            "freeze_policy_ratio_threshold and a previous policy exists, skip "
            "new policy recovery and reuse the previous stage results."
        ),
    )
    parser.add_argument(
        "--freeze_policy_ratio_threshold",
        type=float,
        default=float(config_ns.get("freeze_policy_ratio_threshold", 1.0)),
    )
    parser.add_argument(
        "--curriculum_recovery_mode",
        type=str,
        default=str(config_ns.get("curriculum_recovery_mode", "pair_aggregation")),
        choices=[
            "none",
            "pair_aggregation",
            "attention_spectral_cross_axis",
            "online_pair_stats",
            "online_relation_tournament",
        ],
        help=(
            "How each stage discovers the next training policy. "
            "none writes a no-op results file for pause/resume ablations; "
            "pair_aggregation keeps the historical hierarchical pair merge path; "
            "attention_spectral_cross_axis recovers one learned order directly "
            "from checkpoint attention spectral axes; online_pair_stats aggregates "
            "first-two-unit statistics collected during the preceding train segment; "
            "online_relation_tournament runs a relation-memory tournament from those stats."
        ),
    )
    parser.add_argument(
        "--attention_spectral_backend",
        type=str,
        default=str(config_ns.get("attention_spectral_backend", "offline")),
        choices=["offline", "online_cache_exact", "online_subspace"],
        help=(
            "Switch for attention_spectral_cross_axis recovery. offline keeps the "
            "original benchmark-time attention mining path. online_cache_exact "
            "collects layer/head attention matrices during train and does exact "
            "stage-end EVD from that cache. online_subspace additionally tracks "
            "an online eigenspace and uses a small Rayleigh-Ritz solve at recovery."
        ),
    )
    parser.add_argument(
        "--noop_num_blocks",
        type=int,
        default=int(config_ns.get("noop_num_blocks", 0)),
        help="Number of singleton final_units to write when curriculum_recovery_mode='none'.",
    )
    parser.add_argument(
        "--benchmark_split",
        type=str,
        default=str(config_ns.get("benchmark_split", "val")),
        choices=["train", "val"],
    )
    parser.add_argument(
        "--online_pair_stats_enabled",
        action="store_true",
        default=bool(config_ns.get("online_pair_stats_enabled", False)),
        help="Ask train.py to write observed first-two-unit online pair statistics.",
    )
    parser.add_argument(
        "--online_pair_stats_write_every",
        type=int,
        default=int(config_ns.get("online_pair_stats_write_every", 250)),
    )
    parser.add_argument(
        "--online_pair_stats_min_count",
        type=int,
        default=int(config_ns.get("online_pair_stats_min_count", 8)),
    )
    parser.add_argument(
        "--online_pair_stats_top_k_export",
        type=int,
        default=int(config_ns.get("online_pair_stats_top_k_export", 256)),
    )
    parser.add_argument(
        "--online_pair_stats_aggregate_top_k_pairs",
        type=int,
        default=int(config_ns.get("online_pair_stats_aggregate_top_k_pairs", config_ns.get("aggregate_top_k_pairs", 128))),
    )
    parser.add_argument(
        "--online_pair_stats_margin_threshold",
        type=float,
        default=float(config_ns.get("online_pair_stats_margin_threshold", 0.0)),
    )
    parser.add_argument(
        "--online_pair_stats_positive_rate_threshold",
        type=float,
        default=float(config_ns.get("online_pair_stats_positive_rate_threshold", 0.55)),
    )
    parser.add_argument(
        "--online_pair_stats_max_signed_drop_std",
        type=float,
        default=float(config_ns.get("online_pair_stats_max_signed_drop_std", 1e30)),
    )
    parser.add_argument(
        "--online_pair_stats_allow_missing_reverse",
        action="store_true",
        default=bool(config_ns.get("online_pair_stats_allow_missing_reverse", True)),
    )
    parser.add_argument(
        "--online_pair_stats_use_attention_filter",
        action="store_true",
        default=bool(config_ns.get("online_pair_stats_use_attention_filter", False)),
    )
    parser.add_argument(
        "--online_pair_stats_aggregation_strategy",
        type=str,
        choices=["disjoint_pairs", "directed_chains", "window_consensus_chains"],
        default=str(config_ns.get("online_pair_stats_aggregation_strategy", "disjoint_pairs")),
    )
    parser.add_argument(
        "--online_pair_stats_max_chain_blocks",
        type=int,
        default=int(config_ns.get("online_pair_stats_max_chain_blocks", 8)),
    )
    parser.add_argument(
        "--online_pair_stats_history_window",
        type=int,
        default=int(config_ns.get("online_pair_stats_history_window", 1)),
    )
    parser.add_argument(
        "--online_pair_stats_consensus_min_windows",
        type=int,
        default=int(config_ns.get("online_pair_stats_consensus_min_windows", 2)),
    )
    parser.add_argument(
        "--online_relation_tournament_selected_method",
        type=str,
        default=str(
            config_ns.get(
                "online_relation_tournament_selected_method",
                "prev_seeded_segment_merge_bt_sorted",
            )
        ),
    )
    parser.add_argument(
        "--online_relation_tournament_history_decay",
        type=float,
        default=float(config_ns.get("online_relation_tournament_history_decay", 0.85)),
    )
    parser.add_argument(
        "--online_relation_tournament_merge_margin_threshold",
        type=float,
        default=float(config_ns.get("online_relation_tournament_merge_margin_threshold", 0.25)),
    )
    parser.add_argument(
        "--online_relation_tournament_order_adjacent_margin_threshold",
        type=float,
        default=float(config_ns.get("online_relation_tournament_order_adjacent_margin_threshold", 0.0)),
    )
    parser.add_argument(
        "--online_relation_tournament_strong_edge_top_k",
        type=int,
        default=int(config_ns.get("online_relation_tournament_strong_edge_top_k", 256)),
    )
    parser.add_argument(
        "--online_relation_tournament_attention_min_windows",
        type=int,
        default=int(config_ns.get("online_relation_tournament_attention_min_windows", 2)),
    )
    parser.add_argument(
        "--online_relation_tournament_attention_require_current",
        action="store_true",
        default=bool(config_ns.get("online_relation_tournament_attention_require_current", True)),
    )
    parser.add_argument(
        "--online_relation_tournament_attention_symmetric",
        action="store_true",
        default=bool(config_ns.get("online_relation_tournament_attention_symmetric", True)),
    )
    parser.add_argument(
        "--train_eval_interval",
        type=int,
        default=int(config_ns.get("train_eval_interval", -1)),
    )
    parser.add_argument(
        "--train_eval_iters",
        type=int,
        default=int(config_ns.get("train_eval_iters", -1)),
    )
    parser.add_argument(
        "--train_wandb_log",
        type=str,
        default=str(config_ns.get("train_wandb_log", "")),
    )
    parser.add_argument(
        "--online_attn_probe_enabled",
        action="store_true",
        default=bool(config_ns.get("online_attn_probe_enabled", False)),
    )
    parser.add_argument(
        "--online_attn_probe_interval",
        type=int,
        default=int(config_ns.get("online_attn_probe_interval", 0)),
    )
    parser.add_argument(
        "--online_attn_probe_batch_size",
        type=int,
        default=int(config_ns.get("online_attn_probe_batch_size", 8)),
    )
    parser.add_argument(
        "--online_attn_probe_top_k",
        type=int,
        default=int(config_ns.get("online_attn_probe_top_k", 4)),
    )
    parser.add_argument(
        "--online_attn_probe_ema_decay",
        type=float,
        default=float(config_ns.get("online_attn_probe_ema_decay", 0.95)),
    )
    parser.add_argument(
        "--online_spectral_interval",
        type=int,
        default=int(config_ns.get("online_spectral_interval", 250)),
    )
    parser.add_argument(
        "--online_spectral_batch_size",
        type=int,
        default=int(config_ns.get("online_spectral_batch_size", config_ns.get("attn_batch_size", 8))),
    )
    parser.add_argument(
        "--online_spectral_max_updates",
        type=int,
        default=int(config_ns.get("online_spectral_max_updates", 0)),
    )
    parser.add_argument(
        "--online_spectral_write_every",
        type=int,
        default=int(config_ns.get("online_spectral_write_every", 1)),
    )
    parser.add_argument(
        "--online_spectral_ema_decay",
        type=float,
        default=float(config_ns.get("online_spectral_ema_decay", 0.95)),
    )
    parser.add_argument(
        "--online_spectral_subspace_rank",
        type=int,
        default=int(config_ns.get("online_spectral_subspace_rank", 8)),
    )
    parser.add_argument(
        "--online_spectral_subspace_steps_per_update",
        type=int,
        default=int(config_ns.get("online_spectral_subspace_steps_per_update", 1)),
    )
    parser.add_argument(
        "--online_spectral_subspace_seed",
        type=int,
        default=int(config_ns.get("online_spectral_subspace_seed", 12345)),
    )
    parser.add_argument("--benchmark_batch_size", type=int, default=int(config_ns.get("benchmark_batch_size", 64)))
    parser.add_argument("--pair_mining_batches", type=int, default=int(config_ns.get("pair_mining_batches", 24)))
    parser.add_argument("--pair_eval_batch_size", type=int, default=int(config_ns.get("pair_eval_batch_size", 8)))
    parser.add_argument(
        "--pair_chunk_size",
        type=int,
        default=int(config_ns.get("pair_chunk_size", config_ns.get("pair_eval_batch_size", 8))),
    )
    parser.add_argument(
        "--forward_eval_batch_size",
        type=int,
        default=int(config_ns.get("forward_eval_batch_size", 0)),
    )
    parser.add_argument("--aggregate_top_k_pairs", type=int, default=int(config_ns.get("aggregate_top_k_pairs", 64)))
    parser.add_argument(
        "--aggregate_top_k_pairs_per_stage",
        type=str,
        default=str(config_ns.get("aggregate_top_k_pairs_per_stage", "")),
    )
    parser.add_argument(
        "--aggregation_margin_threshold",
        type=float,
        default=float(config_ns.get("aggregation_margin_threshold", -1e30)),
    )
    parser.add_argument(
        "--aggregation_margin_thresholds",
        type=str,
        default=str(config_ns.get("aggregation_margin_thresholds", "")),
    )
    parser.add_argument(
        "--aggregation_rank_key",
        type=str,
        default=str(config_ns.get("aggregation_rank_key", "score")),
        choices=["score", "margin", "margin_then_score"],
    )
    parser.add_argument(
        "--pair_aggregation_mode",
        type=str,
        default=str(config_ns.get("pair_aggregation_mode", "disjoint_pairs")),
        choices=["chain", "disjoint_pairs", "token_micro_segments"],
    )
    parser.add_argument(
        "--greedy_pair_cover",
        action="store_true",
        default=bool(config_ns.get("greedy_pair_cover", False)),
    )
    parser.add_argument(
        "--target_next_num_units",
        type=int,
        default=int(config_ns.get("target_next_num_units", 0)),
    )
    parser.add_argument(
        "--halve_each_level",
        action="store_true",
        default=bool(config_ns.get("halve_each_level", False)),
    )
    parser.add_argument("--pair_score_k", type=int, default=int(config_ns.get("pair_score_k", 2)))
    parser.add_argument(
        "--pair_score_mode",
        type=str,
        default=str(config_ns.get("pair_score_mode", "abs_tv")),
        choices=["abs_tv", "signed_drop"],
    )
    parser.add_argument("--tv_weight", type=float, default=float(config_ns.get("tv_weight", 0.3)))
    parser.add_argument("--drop_weight", type=float, default=float(config_ns.get("drop_weight", 0.0)))
    parser.add_argument(
        "--drop_weights",
        type=str,
        default=str(config_ns.get("drop_weights", "")),
    )
    parser.add_argument("--token_micro_topk", type=int, default=int(config_ns.get("token_micro_topk", 4)))
    parser.add_argument(
        "--token_micro_min_affinity",
        type=float,
        default=float(config_ns.get("token_micro_min_affinity", -1e30)),
    )
    parser.add_argument("--token_micro_max_degree", type=int, default=int(config_ns.get("token_micro_max_degree", 6)))
    parser.add_argument(
        "--token_micro_max_segment_size",
        type=int,
        default=int(config_ns.get("token_micro_max_segment_size", 4)),
    )
    parser.add_argument(
        "--token_micro_min_segment_size",
        type=int,
        default=int(config_ns.get("token_micro_min_segment_size", 2)),
    )
    parser.add_argument(
        "--token_micro_min_density",
        type=float,
        default=float(config_ns.get("token_micro_min_density", 0.4)),
    )
    parser.add_argument(
        "--token_micro_num_order_candidates",
        type=int,
        default=int(config_ns.get("token_micro_num_order_candidates", 0)),
    )
    parser.add_argument(
        "--token_micro_eval_batches",
        type=int,
        default=int(config_ns.get("token_micro_eval_batches", 8)),
    )
    parser.add_argument(
        "--token_micro_eval_batch_size",
        type=int,
        default=int(config_ns.get("token_micro_eval_batch_size", 8)),
    )
    parser.add_argument(
        "--token_micro_random_suffixes",
        type=int,
        default=int(config_ns.get("token_micro_random_suffixes", 4)),
    )
    parser.add_argument(
        "--token_micro_random_orders",
        type=int,
        default=int(config_ns.get("token_micro_random_orders", 4)),
    )
    parser.add_argument(
        "--token_micro_min_pair_consistency",
        type=float,
        default=float(config_ns.get("token_micro_min_pair_consistency", 0.01)),
    )
    parser.add_argument(
        "--token_micro_reverse_margin_threshold",
        type=float,
        default=float(config_ns.get("token_micro_reverse_margin_threshold", 0.02)),
    )
    parser.add_argument(
        "--token_micro_random_margin_threshold",
        type=float,
        default=float(config_ns.get("token_micro_random_margin_threshold", 0.01)),
    )
    parser.add_argument(
        "--token_micro_keep_singletons",
        action="store_true",
        default=bool(config_ns.get("token_micro_keep_singletons", False)),
    )
    parser.add_argument(
        "--token_micro_max_accepted_segments",
        type=int,
        default=int(config_ns.get("token_micro_max_accepted_segments", 0)),
    )
    parser.add_argument(
        "--benchmark_num_levels",
        type=int,
        default=int(config_ns.get("benchmark_num_levels", 1)),
        help="Use 1 for strict stage-by-stage locked-unit curriculum.",
    )
    parser.add_argument(
        "--early_stop_enabled",
        action="store_true",
        default=bool(config_ns.get("early_stop_enabled", False)),
    )
    parser.add_argument(
        "--early_stop_eval_split",
        type=str,
        default=str(config_ns.get("early_stop_eval_split", "val")),
        choices=["train", "val"],
    )
    parser.add_argument(
        "--early_stop_min_pairs_after_margin",
        type=int,
        default=int(config_ns.get("early_stop_min_pairs_after_margin", 1)),
    )
    parser.add_argument(
        "--early_stop_gain_eval_batches",
        type=int,
        default=int(config_ns.get("early_stop_gain_eval_batches", 0)),
    )
    parser.add_argument(
        "--early_stop_gain_eval_batch_size",
        type=int,
        default=int(config_ns.get("early_stop_gain_eval_batch_size", 8)),
    )
    parser.add_argument(
        "--early_stop_gain_random_contexts",
        type=int,
        default=int(config_ns.get("early_stop_gain_random_contexts", 4)),
    )
    parser.add_argument(
        "--early_stop_gain_threshold",
        type=float,
        default=float(config_ns.get("early_stop_gain_threshold", 0.0)),
    )
    parser.add_argument(
        "--early_stop_gain_p10_threshold",
        type=float,
        default=float(config_ns.get("early_stop_gain_p10_threshold", 0.0)),
    )
    parser.add_argument(
        "--early_stop_gain_positive_rate",
        type=float,
        default=float(config_ns.get("early_stop_gain_positive_rate", 0.7)),
    )
    parser.add_argument(
        "--early_stop_min_gain_pass_rate",
        type=float,
        default=float(config_ns.get("early_stop_min_gain_pass_rate", 0.0)),
    )
    parser.add_argument(
        "--early_stop_filter_segments_by_gain",
        action="store_true",
        default=bool(config_ns.get("early_stop_filter_segments_by_gain", False)),
    )
    parser.add_argument(
        "--early_stop_policy_eval_batches",
        type=int,
        default=int(config_ns.get("early_stop_policy_eval_batches", 0)),
    )
    parser.add_argument(
        "--early_stop_policy_eval_batch_size",
        type=int,
        default=int(config_ns.get("early_stop_policy_eval_batch_size", 8)),
    )
    parser.add_argument(
        "--early_stop_policy_epsilon",
        type=float,
        default=float(config_ns.get("early_stop_policy_epsilon", 0.01)),
    )
    parser.add_argument(
        "--early_stop_boundary_eval_batches",
        type=int,
        default=int(config_ns.get("early_stop_boundary_eval_batches", 0)),
    )
    parser.add_argument(
        "--early_stop_boundary_eval_batch_size",
        type=int,
        default=int(config_ns.get("early_stop_boundary_eval_batch_size", 8)),
    )
    parser.add_argument(
        "--early_stop_boundary_random_contexts",
        type=int,
        default=int(config_ns.get("early_stop_boundary_random_contexts", 4)),
    )
    parser.add_argument(
        "--early_stop_boundary_gain_threshold",
        type=float,
        default=float(config_ns.get("early_stop_boundary_gain_threshold", 0.0)),
    )
    parser.add_argument(
        "--early_stop_boundary_gain_p10_threshold",
        type=float,
        default=float(config_ns.get("early_stop_boundary_gain_p10_threshold", 0.0)),
    )
    parser.add_argument(
        "--early_stop_boundary_positive_rate",
        type=float,
        default=float(config_ns.get("early_stop_boundary_positive_rate", 0.6)),
    )
    parser.add_argument(
        "--early_stop_max_bad_boundary_ratio",
        type=float,
        default=float(config_ns.get("early_stop_max_bad_boundary_ratio", 1.0)),
    )
    parser.add_argument(
        "--pair_mining_mode",
        type=str,
        default=str(config_ns.get("pair_mining_mode", "attention_pruned")),
        choices=["full", "attention_pruned"],
    )
    parser.add_argument("--attn_top_k", type=int, default=int(config_ns.get("attn_top_k", 8)))
    parser.add_argument(
        "--attn_top_ks",
        type=str,
        default=str(config_ns.get("attn_top_ks", "")),
    )
    parser.add_argument("--attn_num_batches", type=int, default=int(config_ns.get("attn_num_batches", 24)))
    parser.add_argument("--attn_batch_size", type=int, default=int(config_ns.get("attn_batch_size", 32)))
    parser.add_argument(
        "--attn_mode",
        type=str,
        default=str(config_ns.get("attn_mode", "Random")),
        choices=["AR", "Random"],
    )
    parser.add_argument(
        "--attn_symmetrize",
        type=str,
        default=str(config_ns.get("attn_symmetrize", "mean")),
        choices=["mean", "max"],
    )
    parser.add_argument(
        "--attn_export_type",
        type=str,
        default=str(config_ns.get("attn_export_type", "with_none")),
        choices=["with_none", "without_none"],
    )
    parser.add_argument(
        "--spectral_max_primary_matrices",
        type=int,
        default=int(config_ns.get("spectral_max_primary_matrices", 16)),
    )
    parser.add_argument(
        "--spectral_max_primary_matrices_per_stage",
        type=str,
        default=str(config_ns.get("spectral_max_primary_matrices_per_stage", "")),
    )
    parser.add_argument("--spectral_num_components", type=int, default=int(config_ns.get("spectral_num_components", 4)))
    parser.add_argument("--spectral_component_pairs", type=str, default=str(config_ns.get("spectral_component_pairs", "1-2")))
    parser.add_argument("--spectral_num_angles", type=int, default=int(config_ns.get("spectral_num_angles", 16)))
    parser.add_argument(
        "--spectral_num_angles_per_stage",
        type=str,
        default=str(config_ns.get("spectral_num_angles_per_stage", "")),
    )
    parser.add_argument("--spectral_k_values", type=str, default=str(config_ns.get("spectral_k_values", "8,10")))
    parser.add_argument(
        "--spectral_k_values_per_stage",
        type=str,
        default=str(config_ns.get("spectral_k_values_per_stage", "")),
        help="Optional semicolon-separated per-stage k lists, e.g. '4,6;6,8;8,10'.",
    )
    parser.add_argument("--spectral_group_methods", type=str, default=str(config_ns.get("spectral_group_methods", "gap")))
    parser.add_argument(
        "--spectral_threshold_percentile",
        type=float,
        default=float(config_ns.get("spectral_threshold_percentile", 60.0)),
    )
    parser.add_argument("--spectral_transform", type=str, default=str(config_ns.get("spectral_transform", "relu")))
    parser.add_argument("--spectral_temperature", type=float, default=float(config_ns.get("spectral_temperature", 1.0)))
    parser.add_argument(
        "--spectral_direction_lambdas",
        type=str,
        default=str(config_ns.get("spectral_direction_lambdas", "0,0.1,0.25")),
    )
    parser.add_argument(
        "--spectral_directed_score_weight",
        type=float,
        default=float(config_ns.get("spectral_directed_score_weight", 0.25)),
    )
    parser.add_argument(
        "--spectral_band_quality_weight",
        type=float,
        default=float(config_ns.get("spectral_band_quality_weight", 0.05)),
    )
    parser.add_argument(
        "--spectral_score_adjacency_sym",
        type=str,
        default=str(config_ns.get("spectral_score_adjacency_sym", "max")),
        choices=["mean", "max"],
    )
    parser.add_argument("--spectral_max_candidates", type=int, default=int(config_ns.get("spectral_max_candidates", 4096)))
    parser.add_argument(
        "--spectral_export_top_candidates",
        type=int,
        default=int(config_ns.get("spectral_export_top_candidates", 128)),
    )
    parser.add_argument(
        "--spectral_output_policy",
        type=str,
        default=str(config_ns.get("spectral_output_policy", "single_order")),
        choices=["single_order", "bands"],
    )
    parser.add_argument(
        "--spectral_rerank_num_candidates",
        type=int,
        default=int(config_ns.get("spectral_rerank_num_candidates", 0)),
    )
    parser.add_argument(
        "--spectral_rerank_num_batches",
        type=int,
        default=int(config_ns.get("spectral_rerank_num_batches", 0)),
    )
    parser.add_argument(
        "--spectral_rerank_batch_size",
        type=int,
        default=int(config_ns.get("spectral_rerank_batch_size", 8)),
    )
    parser.add_argument(
        "--spectral_rerank_forward_batch_size",
        type=int,
        default=int(config_ns.get("spectral_rerank_forward_batch_size", 0)),
    )
    parser.add_argument(
        "--spectral_rerank_prefix_k",
        type=int,
        default=int(config_ns.get("spectral_rerank_prefix_k", 16)),
    )
    parser.add_argument(
        "--spectral_rerank_include_reverse",
        type=int,
        default=int(config_ns.get("spectral_rerank_include_reverse", 1)),
        choices=[0, 1],
    )
    parser.add_argument(
        "--spectral_rerank_attention_weight",
        type=float,
        default=float(config_ns.get("spectral_rerank_attention_weight", 1.0)),
    )
    parser.add_argument(
        "--spectral_rerank_full_loss_weight",
        type=float,
        default=float(config_ns.get("spectral_rerank_full_loss_weight", 0.15)),
    )
    parser.add_argument(
        "--spectral_rerank_prefix_loss_weight",
        type=float,
        default=float(config_ns.get("spectral_rerank_prefix_loss_weight", 0.25)),
    )
    parser.add_argument(
        "--spectral_rerank_signed_drop_weight",
        type=float,
        default=float(config_ns.get("spectral_rerank_signed_drop_weight", 0.10)),
    )
    parser.add_argument(
        "--spectral_rerank_reverse_margin_weight",
        type=float,
        default=float(config_ns.get("spectral_rerank_reverse_margin_weight", 0.30)),
    )
    parser.add_argument(
        "--spectral_rerank_positive_rate_weight",
        type=float,
        default=float(config_ns.get("spectral_rerank_positive_rate_weight", 0.10)),
    )
    parser.add_argument(
        "--spectral_rerank_stability_weight",
        type=float,
        default=float(config_ns.get("spectral_rerank_stability_weight", 0.10)),
    )
    parser.add_argument("--spectral_seed", type=int, default=int(config_ns.get("spectral_seed", 12345)))
    parser.add_argument("--spectral_device", type=str, default=str(config_ns.get("spectral_device", "cuda")))
    parser.add_argument("--spectral_dtype", type=str, default=str(config_ns.get("spectral_dtype", "bfloat16")))
    return parser.parse_args(filtered_argv)


def main():
    args = parse_args()
    repo_dir = REPO_ROOT
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    ratios = parse_csv_list(args.segment_guided_ratios, float)
    segment_lens = parse_csv_list(args.segment_max_lens, int)
    aggregation_margin_thresholds = parse_optional_csv_list(args.aggregation_margin_thresholds, float)
    aggregate_top_k_pairs_per_stage = parse_optional_csv_list(args.aggregate_top_k_pairs_per_stage, int)
    drop_weights = parse_optional_csv_list(args.drop_weights, float)
    attn_top_ks = parse_optional_csv_list(args.attn_top_ks, int)
    spectral_max_primary_matrices_per_stage = parse_optional_csv_list(
        args.spectral_max_primary_matrices_per_stage,
        int,
    )
    spectral_num_angles_per_stage = parse_optional_csv_list(args.spectral_num_angles_per_stage, int)
    spectral_k_values_per_stage = parse_optional_stage_text_list(args.spectral_k_values_per_stage)
    train_wandb_log_override = parse_optional_bool(args.train_wandb_log)
    segment_lock_final_units = parse_optional_bool(args.segment_lock_final_units)
    if segment_lock_final_units is None:
        segment_lock_final_units = True
    if len(ratios) != int(args.num_curriculum_stages):
        raise ValueError("segment_guided_ratios length must equal num_curriculum_stages")
    if len(segment_lens) != int(args.num_curriculum_stages):
        raise ValueError("segment_max_lens length must equal num_curriculum_stages")
    per_stage_lists = {
        "aggregation_margin_thresholds": aggregation_margin_thresholds,
        "aggregate_top_k_pairs_per_stage": aggregate_top_k_pairs_per_stage,
        "drop_weights": drop_weights,
        "attn_top_ks": attn_top_ks,
        "spectral_max_primary_matrices_per_stage": spectral_max_primary_matrices_per_stage,
        "spectral_num_angles_per_stage": spectral_num_angles_per_stage,
        "spectral_k_values_per_stage": spectral_k_values_per_stage,
    }
    for name, values in per_stage_lists.items():
        if values and len(values) != int(args.num_curriculum_stages):
            raise ValueError(f"{name} length must equal num_curriculum_stages when provided")

    train_out_dir = args.train_out_dir if args.train_out_dir.is_absolute() else REPO_ROOT / args.train_out_dir
    benchmark_root = args.benchmark_root if args.benchmark_root.is_absolute() else REPO_ROOT / args.benchmark_root
    benchmark_root.mkdir(parents=True, exist_ok=True)
    ckpt_path = train_out_dir / "ckpt.pt"
    online_stats_train_enabled = bool(args.online_pair_stats_enabled) or str(
        args.curriculum_recovery_mode
    ) in {"online_pair_stats", "online_relation_tournament"}
    online_spectral_train_enabled = (
        str(args.curriculum_recovery_mode) == "attention_spectral_cross_axis"
        and str(args.attention_spectral_backend) in {"online_cache_exact", "online_subspace"}
    )

    def online_stats_dir_for_stage(stage_idx):
        return benchmark_root / f"stage_{int(stage_idx):02d}" / "online_train_stats"

    def online_spectral_dir_for_stage(stage_idx):
        return benchmark_root / f"stage_{int(stage_idx):02d}" / "online_spectral_state"

    def online_spectral_state_npz_for_stage(stage_idx):
        return online_spectral_dir_for_stage(stage_idx) / "online_spectral_state.npz"

    def online_train_kwargs(stage_idx):
        return {
            "online_pair_stats_enabled": online_stats_train_enabled,
            "online_pair_stats_out_dir": str(online_stats_dir_for_stage(stage_idx)),
            "online_pair_stats_write_every": int(args.online_pair_stats_write_every),
            "online_pair_stats_min_count": int(args.online_pair_stats_min_count),
            "online_pair_stats_top_k_export": int(args.online_pair_stats_top_k_export),
            "online_attn_probe_enabled": bool(args.online_attn_probe_enabled),
            "online_attn_probe_interval": int(args.online_attn_probe_interval),
            "online_attn_probe_batch_size": int(args.online_attn_probe_batch_size),
            "online_attn_probe_top_k": int(args.online_attn_probe_top_k),
            "online_attn_probe_ema_decay": float(args.online_attn_probe_ema_decay),
            "online_spectral_enabled": online_spectral_train_enabled,
            "online_spectral_mode": (
                "subspace"
                if str(args.attention_spectral_backend) == "online_subspace"
                else "cache_exact"
            ),
            "online_spectral_out_dir": str(online_spectral_dir_for_stage(stage_idx)),
            "online_spectral_interval": int(args.online_spectral_interval),
            "online_spectral_batch_size": int(args.online_spectral_batch_size),
            "online_spectral_max_updates": int(args.online_spectral_max_updates),
            "online_spectral_write_every": int(args.online_spectral_write_every),
            "online_spectral_export_type": str(args.attn_export_type),
            "online_spectral_ema_decay": float(args.online_spectral_ema_decay),
            "online_spectral_subspace_rank": int(args.online_spectral_subspace_rank),
            "online_spectral_subspace_steps_per_update": int(args.online_spectral_subspace_steps_per_update),
            "online_spectral_subspace_seed": int(args.online_spectral_subspace_seed) + 7000 * int(stage_idx),
            "online_spectral_threshold_percentile": float(args.spectral_threshold_percentile),
            "online_spectral_transform": str(args.spectral_transform),
            "online_spectral_temperature": float(args.spectral_temperature),
        }

    wandb_run_id = str(args.wandb_run_id).strip()
    if args.wandb_project and not wandb_run_id:
        wandb_run_id = build_curriculum_wandb_run_id(
            base_name=args.wandb_run_name or train_out_dir.name,
            train_out_dir=train_out_dir,
            config_path=config_path,
        )

    existing_iter = checkpoint_iter(ckpt_path) if bool(args.resume_existing) else None
    if existing_iter is not None and existing_iter >= int(args.warmup_iters):
        print(
            "[runner] resume_existing: skipping warmup because "
            f"{ckpt_path} is already at iter {existing_iter}."
        )
    else:
        warmup_init_from = "resume" if existing_iter is not None and existing_iter > 0 else "scratch"
        run_command(
            build_train_cmd(
                repo_dir=repo_dir,
                config_path=config_path,
                out_dir=train_out_dir,
                init_from=warmup_init_from,
                max_iters=args.warmup_iters,
                segment_guided_ratio=0.0,
                segment_source_json="",
                segment_top_k_pairs=args.segment_top_k_pairs,
                segment_max_len=segment_lens[0],
                segment_max_units_per_order=args.segment_max_units_per_order,
                segment_lock_final_units=segment_lock_final_units,
                wandb_project=args.wandb_project,
                wandb_run_name=args.wandb_run_name,
                wandb_run_id=wandb_run_id,
                train_eval_interval=args.train_eval_interval,
                train_eval_iters=args.train_eval_iters,
                train_wandb_log=train_wandb_log_override,
                **online_train_kwargs(1),
            ),
            cwd=repo_dir,
        )

    cumulative_max_iters = int(args.warmup_iters)
    previous_stage_results = ""
    final_max_iters = int(args.warmup_iters) + int(args.stage_iters) * int(args.num_curriculum_stages)
    single_unit_run_to_end_triggered = False
    single_unit_run_to_end_stage = None
    single_unit_run_to_end_results = ""
    policy_freeze_events = []
    for stage_idx in range(1, int(args.num_curriculum_stages) + 1):
        stage_ratio = float(ratios[stage_idx - 1])
        stage_aggregation_margin_threshold = stage_value(
            aggregation_margin_thresholds,
            float(args.aggregation_margin_threshold),
            stage_idx,
        )
        stage_aggregate_top_k_pairs = stage_value(
            aggregate_top_k_pairs_per_stage,
            int(args.aggregate_top_k_pairs),
            stage_idx,
        )
        stage_drop_weight = stage_value(drop_weights, float(args.drop_weight), stage_idx)
        stage_attn_top_k = stage_value(attn_top_ks, int(args.attn_top_k), stage_idx)
        stage_spectral_max_primary = stage_value(
            spectral_max_primary_matrices_per_stage,
            int(args.spectral_max_primary_matrices),
            stage_idx,
        )
        stage_spectral_num_angles = stage_value(
            spectral_num_angles_per_stage,
            int(args.spectral_num_angles),
            stage_idx,
        )
        stage_spectral_k_values = stage_value(
            spectral_k_values_per_stage,
            str(args.spectral_k_values),
            stage_idx,
        )
        benchmark_dir = benchmark_root / f"stage_{stage_idx:02d}"
        current_stage_results = str(benchmark_dir / "results.json")
        freeze_this_stage = (
            bool(args.freeze_policy_at_full_guidance)
            and stage_ratio >= float(args.freeze_policy_ratio_threshold)
            and bool(previous_stage_results)
        )
        if freeze_this_stage:
            freeze_meta = write_frozen_policy_results(
                source_results=previous_stage_results,
                target_results=current_stage_results,
                stage_idx=stage_idx,
                ratio=stage_ratio,
                threshold=float(args.freeze_policy_ratio_threshold),
            )
            policy_freeze_events.append(freeze_meta)
            print(
                "[runner] freeze_policy_at_full_guidance: skipping stage "
                f"{stage_idx} recovery and reusing {previous_stage_results}"
            )
            benchmark_cmd = None
        elif str(args.curriculum_recovery_mode) == "none":
            if bool(args.resume_existing) and results_json_complete(current_stage_results):
                print(f"[runner] resume_existing: reusing no-op {current_stage_results}")
            else:
                write_no_recovery_results(
                    target_results=current_stage_results,
                    stage_idx=stage_idx,
                    ratio=stage_ratio,
                    num_blocks=int(args.noop_num_blocks),
                )
                print(f"[runner] no recovery: wrote no-op stage results to {current_stage_results}")
            benchmark_cmd = None
        elif str(args.curriculum_recovery_mode) == "online_pair_stats":
            history_start = max(1, stage_idx - max(1, int(args.online_pair_stats_history_window)) + 1)
            extra_online_stats_dirs = [
                online_stats_dir_for_stage(prev_stage_idx)
                for prev_stage_idx in range(history_start, stage_idx)
            ]
            benchmark_cmd = build_online_pair_stats_aggregation_cmd(
                benchmark_out_dir=benchmark_dir,
                stats_dir=online_stats_dir_for_stage(stage_idx),
                extra_stats_dirs=extra_online_stats_dirs,
                num_blocks=0,
                top_k_pairs=args.online_pair_stats_aggregate_top_k_pairs,
                min_count=args.online_pair_stats_min_count,
                margin_threshold=args.online_pair_stats_margin_threshold,
                positive_rate_threshold=args.online_pair_stats_positive_rate_threshold,
                max_signed_drop_std=args.online_pair_stats_max_signed_drop_std,
                drop_weight=stage_drop_weight,
                allow_missing_reverse=bool(args.online_pair_stats_allow_missing_reverse),
                use_attention_filter=bool(args.online_pair_stats_use_attention_filter),
                aggregation_strategy=str(args.online_pair_stats_aggregation_strategy),
                max_chain_blocks=int(args.online_pair_stats_max_chain_blocks),
                consensus_min_windows=int(args.online_pair_stats_consensus_min_windows),
            )
        elif str(args.curriculum_recovery_mode) == "online_relation_tournament":
            history_start = max(1, stage_idx - max(1, int(args.online_pair_stats_history_window)) + 1)
            extra_online_stats_dirs = [
                online_stats_dir_for_stage(prev_stage_idx)
                for prev_stage_idx in range(history_start, stage_idx)
            ]
            benchmark_cmd = build_online_relation_tournament_cmd(
                benchmark_out_dir=benchmark_dir,
                stats_dir=online_stats_dir_for_stage(stage_idx),
                extra_stats_dirs=extra_online_stats_dirs,
                previous_results_json=previous_stage_results,
                top_k_edges=args.online_pair_stats_aggregate_top_k_pairs,
                min_count=args.online_pair_stats_min_count,
                margin_threshold=args.online_pair_stats_margin_threshold,
                positive_rate_threshold=args.online_pair_stats_positive_rate_threshold,
                max_signed_drop_std=args.online_pair_stats_max_signed_drop_std,
                allow_missing_reverse=bool(args.online_pair_stats_allow_missing_reverse),
                max_segment_len=int(args.online_pair_stats_max_chain_blocks),
                consensus_min_windows=int(args.online_pair_stats_consensus_min_windows),
                selected_method=str(args.online_relation_tournament_selected_method),
                history_decay=float(args.online_relation_tournament_history_decay),
                merge_margin_threshold=float(args.online_relation_tournament_merge_margin_threshold),
                order_adjacent_margin_threshold=float(
                    args.online_relation_tournament_order_adjacent_margin_threshold
                ),
                strong_edge_top_k=int(args.online_relation_tournament_strong_edge_top_k),
                attention_min_windows=int(args.online_relation_tournament_attention_min_windows),
                attention_require_current=bool(
                    args.online_relation_tournament_attention_require_current
                ),
                attention_symmetric=bool(args.online_relation_tournament_attention_symmetric),
            )
        elif str(args.curriculum_recovery_mode) == "attention_spectral_cross_axis":
            spectral_solver = {
                "offline": "exact",
                "online_cache_exact": "state_exact",
                "online_subspace": "state_subspace",
            }[str(args.attention_spectral_backend)]
            attention_state_npz = ""
            if str(args.attention_spectral_backend) != "offline":
                attention_state_path = online_spectral_state_npz_for_stage(stage_idx)
                if not attention_state_path.exists():
                    raise FileNotFoundError(
                        "attention_spectral_backend="
                        f"{args.attention_spectral_backend!r} requires online spectral state "
                        f"from the preceding train segment, but it was not found: {attention_state_path}"
                    )
                attention_state_npz = str(attention_state_path)
            benchmark_cmd = build_attention_spectral_benchmark_cmd(
                benchmark_out_dir=benchmark_dir,
                ckpt_path=ckpt_path,
                split=args.benchmark_split,
                attn_num_batches=args.attn_num_batches,
                attn_batch_size=args.attn_batch_size,
                attn_mode=args.attn_mode,
                attn_export_type=args.attn_export_type,
                max_primary_matrices=stage_spectral_max_primary,
                num_components=args.spectral_num_components,
                component_pairs=args.spectral_component_pairs,
                num_angles=stage_spectral_num_angles,
                k_values=stage_spectral_k_values,
                group_methods=args.spectral_group_methods,
                threshold_percentile=args.spectral_threshold_percentile,
                transform=args.spectral_transform,
                temperature=args.spectral_temperature,
                direction_lambdas=args.spectral_direction_lambdas,
                directed_score_weight=args.spectral_directed_score_weight,
                band_quality_weight=args.spectral_band_quality_weight,
                score_adjacency_sym=args.spectral_score_adjacency_sym,
                max_candidates=args.spectral_max_candidates,
                export_top_candidates=args.spectral_export_top_candidates,
                output_policy=args.spectral_output_policy,
                rerank_num_candidates=args.spectral_rerank_num_candidates,
                rerank_num_batches=args.spectral_rerank_num_batches,
                rerank_batch_size=args.spectral_rerank_batch_size,
                rerank_forward_batch_size=args.spectral_rerank_forward_batch_size,
                rerank_prefix_k=args.spectral_rerank_prefix_k,
                rerank_include_reverse=bool(args.spectral_rerank_include_reverse),
                rerank_attention_weight=args.spectral_rerank_attention_weight,
                rerank_full_loss_weight=args.spectral_rerank_full_loss_weight,
                rerank_prefix_loss_weight=args.spectral_rerank_prefix_loss_weight,
                rerank_signed_drop_weight=args.spectral_rerank_signed_drop_weight,
                rerank_reverse_margin_weight=args.spectral_rerank_reverse_margin_weight,
                rerank_positive_rate_weight=args.spectral_rerank_positive_rate_weight,
                rerank_stability_weight=args.spectral_rerank_stability_weight,
                seed=int(args.spectral_seed) + 7000 * stage_idx,
                device=args.spectral_device,
                dtype=args.spectral_dtype,
                spectral_solver=spectral_solver,
                attention_state_npz=attention_state_npz,
            )
        else:
            benchmark_cmd = build_benchmark_cmd(
                    benchmark_out_dir=benchmark_dir,
                    ckpt_path=ckpt_path,
                    split=args.benchmark_split,
                    batch_size=args.benchmark_batch_size,
                    pair_mining_batches=args.pair_mining_batches,
                    pair_eval_batch_size=args.pair_eval_batch_size,
                    pair_chunk_size=args.pair_chunk_size,
                    forward_eval_batch_size=args.forward_eval_batch_size,
                    aggregate_top_k_pairs=stage_aggregate_top_k_pairs,
                    aggregation_margin_threshold=stage_aggregation_margin_threshold,
                    aggregation_rank_key=args.aggregation_rank_key,
                    pair_aggregation_mode=args.pair_aggregation_mode,
                    greedy_pair_cover=args.greedy_pair_cover,
                    target_next_num_units=args.target_next_num_units,
                    halve_each_level=args.halve_each_level,
                    segment_len=segment_lens[stage_idx - 1],
                    pair_score_k=args.pair_score_k,
                    pair_score_mode=args.pair_score_mode,
                    tv_weight=args.tv_weight,
                    drop_weight=stage_drop_weight,
                    token_micro_topk=args.token_micro_topk,
                    token_micro_min_affinity=args.token_micro_min_affinity,
                    token_micro_max_degree=args.token_micro_max_degree,
                    token_micro_max_segment_size=args.token_micro_max_segment_size,
                    token_micro_min_segment_size=args.token_micro_min_segment_size,
                    token_micro_min_density=args.token_micro_min_density,
                    token_micro_num_order_candidates=args.token_micro_num_order_candidates,
                    token_micro_eval_batches=args.token_micro_eval_batches,
                    token_micro_eval_batch_size=args.token_micro_eval_batch_size,
                    token_micro_random_suffixes=args.token_micro_random_suffixes,
                    token_micro_random_orders=args.token_micro_random_orders,
                    token_micro_min_pair_consistency=args.token_micro_min_pair_consistency,
                    token_micro_reverse_margin_threshold=args.token_micro_reverse_margin_threshold,
                    token_micro_random_margin_threshold=args.token_micro_random_margin_threshold,
                    token_micro_keep_singletons=args.token_micro_keep_singletons,
                    token_micro_max_accepted_segments=args.token_micro_max_accepted_segments,
                    num_levels=args.benchmark_num_levels,
                    early_stop_enabled=args.early_stop_enabled,
                    early_stop_eval_split=args.early_stop_eval_split,
                    early_stop_min_pairs_after_margin=args.early_stop_min_pairs_after_margin,
                    early_stop_gain_eval_batches=args.early_stop_gain_eval_batches,
                    early_stop_gain_eval_batch_size=args.early_stop_gain_eval_batch_size,
                    early_stop_gain_random_contexts=args.early_stop_gain_random_contexts,
                    early_stop_gain_threshold=args.early_stop_gain_threshold,
                    early_stop_gain_p10_threshold=args.early_stop_gain_p10_threshold,
                    early_stop_gain_positive_rate=args.early_stop_gain_positive_rate,
                    early_stop_min_gain_pass_rate=args.early_stop_min_gain_pass_rate,
                    early_stop_filter_segments_by_gain=args.early_stop_filter_segments_by_gain,
                    early_stop_policy_eval_batches=args.early_stop_policy_eval_batches,
                    early_stop_policy_eval_batch_size=args.early_stop_policy_eval_batch_size,
                    early_stop_policy_epsilon=args.early_stop_policy_epsilon,
                    early_stop_boundary_eval_batches=args.early_stop_boundary_eval_batches,
                    early_stop_boundary_eval_batch_size=args.early_stop_boundary_eval_batch_size,
                    early_stop_boundary_random_contexts=args.early_stop_boundary_random_contexts,
                    early_stop_boundary_gain_threshold=args.early_stop_boundary_gain_threshold,
                    early_stop_boundary_gain_p10_threshold=args.early_stop_boundary_gain_p10_threshold,
                    early_stop_boundary_positive_rate=args.early_stop_boundary_positive_rate,
                    early_stop_max_bad_boundary_ratio=args.early_stop_max_bad_boundary_ratio,
                    pair_mining_mode=args.pair_mining_mode,
                    attn_top_k=stage_attn_top_k,
                    attn_num_batches=args.attn_num_batches,
                    attn_batch_size=args.attn_batch_size,
                    attn_mode=args.attn_mode,
                    attn_symmetrize=args.attn_symmetrize,
                    attn_export_type=args.attn_export_type,
                    initial_units_json=previous_stage_results,
                )
        if benchmark_cmd is None:
            pass
        elif bool(args.resume_existing) and results_json_complete(current_stage_results):
            print(f"[runner] resume_existing: reusing {current_stage_results}")
        else:
            run_command(benchmark_cmd, cwd=repo_dir)

        num_final_units = count_final_units(current_stage_results)
        if (
            bool(args.run_to_end_after_single_unit)
            and str(args.curriculum_recovery_mode) == "pair_aggregation"
            and num_final_units == 1
            and stage_idx < int(args.num_curriculum_stages)
        ):
            print(
                "[runner] benchmark collapsed to one final unit at "
                f"stage {stage_idx}; training directly to final max_iters={final_max_iters}."
            )
            run_command(
                build_train_cmd(
                    repo_dir=repo_dir,
                    config_path=config_path,
                    out_dir=train_out_dir,
                    init_from="resume",
                    max_iters=final_max_iters,
                    segment_guided_ratio=ratios[-1],
                    segment_source_json=current_stage_results,
                    segment_top_k_pairs=args.segment_top_k_pairs,
                    segment_max_len=segment_lens[-1],
                    segment_max_units_per_order=args.segment_max_units_per_order,
                    segment_lock_final_units=segment_lock_final_units,
                    wandb_project=args.wandb_project,
                    wandb_run_name=args.wandb_run_name,
                    wandb_run_id=wandb_run_id,
                    train_eval_interval=args.train_eval_interval,
                    train_eval_iters=args.train_eval_iters,
                    train_wandb_log=train_wandb_log_override,
                    **online_train_kwargs(stage_idx + 1),
                ),
                cwd=repo_dir,
            )
            cumulative_max_iters = final_max_iters
            previous_stage_results = current_stage_results
            single_unit_run_to_end_triggered = True
            single_unit_run_to_end_stage = int(stage_idx)
            single_unit_run_to_end_results = current_stage_results
            break

        cumulative_max_iters += int(args.stage_iters)
        existing_iter = checkpoint_iter(ckpt_path) if bool(args.resume_existing) else None
        if existing_iter is not None and existing_iter >= int(cumulative_max_iters):
            print(
                "[runner] resume_existing: skipping stage "
                f"{stage_idx} training because checkpoint is already at iter {existing_iter}."
            )
        else:
            run_command(
                build_train_cmd(
                    repo_dir=repo_dir,
                    config_path=config_path,
                    out_dir=train_out_dir,
                    init_from="resume",
                    max_iters=cumulative_max_iters,
                    segment_guided_ratio=stage_ratio,
                    segment_source_json=current_stage_results,
                    segment_top_k_pairs=args.segment_top_k_pairs,
                    segment_max_len=segment_lens[stage_idx - 1],
                    segment_max_units_per_order=args.segment_max_units_per_order,
                    segment_lock_final_units=segment_lock_final_units,
                    wandb_project=args.wandb_project,
                    wandb_run_name=args.wandb_run_name,
                    wandb_run_id=wandb_run_id,
                    train_eval_interval=args.train_eval_interval,
                    train_eval_iters=args.train_eval_iters,
                    train_wandb_log=train_wandb_log_override,
                    **online_train_kwargs(stage_idx + 1),
                ),
                cwd=repo_dir,
            )
        previous_stage_results = current_stage_results

    runner_meta = {
        "config": str(config_path),
        "train_out_dir": str(train_out_dir),
        "benchmark_root": str(benchmark_root),
        "wandb_project": args.wandb_project,
        "wandb_run_name": args.wandb_run_name,
        "wandb_run_id": wandb_run_id,
        "warmup_iters": int(args.warmup_iters),
        "stage_iters": int(args.stage_iters),
        "num_curriculum_stages": int(args.num_curriculum_stages),
        "final_max_iters": int(final_max_iters),
        "resume_existing": bool(args.resume_existing),
        "curriculum_recovery_mode": str(args.curriculum_recovery_mode),
        "noop_num_blocks": int(args.noop_num_blocks),
        "online_pair_stats_enabled": bool(online_stats_train_enabled),
        "online_pair_stats_write_every": int(args.online_pair_stats_write_every),
        "online_pair_stats_min_count": int(args.online_pair_stats_min_count),
        "online_pair_stats_top_k_export": int(args.online_pair_stats_top_k_export),
        "online_pair_stats_aggregate_top_k_pairs": int(args.online_pair_stats_aggregate_top_k_pairs),
        "online_pair_stats_margin_threshold": float(args.online_pair_stats_margin_threshold),
        "online_pair_stats_positive_rate_threshold": float(args.online_pair_stats_positive_rate_threshold),
        "online_pair_stats_max_signed_drop_std": float(args.online_pair_stats_max_signed_drop_std),
        "online_pair_stats_allow_missing_reverse": bool(args.online_pair_stats_allow_missing_reverse),
        "online_pair_stats_use_attention_filter": bool(args.online_pair_stats_use_attention_filter),
        "online_pair_stats_aggregation_strategy": str(args.online_pair_stats_aggregation_strategy),
        "online_pair_stats_max_chain_blocks": int(args.online_pair_stats_max_chain_blocks),
        "online_pair_stats_history_window": int(args.online_pair_stats_history_window),
        "online_pair_stats_consensus_min_windows": int(args.online_pair_stats_consensus_min_windows),
        "online_relation_tournament_selected_method": str(args.online_relation_tournament_selected_method),
        "online_relation_tournament_history_decay": float(args.online_relation_tournament_history_decay),
        "online_relation_tournament_merge_margin_threshold": float(
            args.online_relation_tournament_merge_margin_threshold
        ),
        "online_relation_tournament_order_adjacent_margin_threshold": float(
            args.online_relation_tournament_order_adjacent_margin_threshold
        ),
        "online_relation_tournament_strong_edge_top_k": int(
            args.online_relation_tournament_strong_edge_top_k
        ),
        "online_relation_tournament_attention_min_windows": int(
            args.online_relation_tournament_attention_min_windows
        ),
        "online_relation_tournament_attention_require_current": bool(
            args.online_relation_tournament_attention_require_current
        ),
        "online_relation_tournament_attention_symmetric": bool(
            args.online_relation_tournament_attention_symmetric
        ),
        "train_eval_interval": int(args.train_eval_interval),
        "train_eval_iters": int(args.train_eval_iters),
        "train_wandb_log": train_wandb_log_override,
        "online_attn_probe_enabled": bool(args.online_attn_probe_enabled),
        "online_attn_probe_interval": int(args.online_attn_probe_interval),
        "online_attn_probe_batch_size": int(args.online_attn_probe_batch_size),
        "online_attn_probe_top_k": int(args.online_attn_probe_top_k),
        "online_attn_probe_ema_decay": float(args.online_attn_probe_ema_decay),
        "attention_spectral_backend": str(args.attention_spectral_backend),
        "online_spectral_enabled": bool(online_spectral_train_enabled),
        "online_spectral_interval": int(args.online_spectral_interval),
        "online_spectral_batch_size": int(args.online_spectral_batch_size),
        "online_spectral_max_updates": int(args.online_spectral_max_updates),
        "online_spectral_write_every": int(args.online_spectral_write_every),
        "online_spectral_ema_decay": float(args.online_spectral_ema_decay),
        "online_spectral_subspace_rank": int(args.online_spectral_subspace_rank),
        "online_spectral_subspace_steps_per_update": int(args.online_spectral_subspace_steps_per_update),
        "online_spectral_subspace_seed": int(args.online_spectral_subspace_seed),
        "run_to_end_after_single_unit": bool(args.run_to_end_after_single_unit),
        "single_unit_run_to_end_triggered": bool(single_unit_run_to_end_triggered),
        "single_unit_run_to_end_stage": single_unit_run_to_end_stage,
        "single_unit_run_to_end_results": single_unit_run_to_end_results,
        "segment_guided_ratios": ratios,
        "segment_max_lens": segment_lens,
        "segment_max_units_per_order": int(args.segment_max_units_per_order),
        "segment_lock_final_units": bool(segment_lock_final_units),
        "segment_top_k_pairs": int(args.segment_top_k_pairs),
        "freeze_policy_at_full_guidance": bool(args.freeze_policy_at_full_guidance),
        "freeze_policy_ratio_threshold": float(args.freeze_policy_ratio_threshold),
        "policy_freeze_events": policy_freeze_events,
        "benchmark_split": str(args.benchmark_split),
        "benchmark_batch_size": int(args.benchmark_batch_size),
        "pair_mining_batches": int(args.pair_mining_batches),
        "pair_eval_batch_size": int(args.pair_eval_batch_size),
        "pair_chunk_size": int(args.pair_chunk_size),
        "forward_eval_batch_size": int(args.forward_eval_batch_size),
        "aggregate_top_k_pairs": int(args.aggregate_top_k_pairs),
        "aggregate_top_k_pairs_per_stage": aggregate_top_k_pairs_per_stage,
        "aggregation_margin_threshold": float(args.aggregation_margin_threshold),
        "aggregation_margin_thresholds": aggregation_margin_thresholds,
        "aggregation_rank_key": str(args.aggregation_rank_key),
        "pair_aggregation_mode": str(args.pair_aggregation_mode),
        "greedy_pair_cover": bool(args.greedy_pair_cover),
        "target_next_num_units": int(args.target_next_num_units),
        "halve_each_level": bool(args.halve_each_level),
        "pair_score_k": int(args.pair_score_k),
        "pair_score_mode": str(args.pair_score_mode),
        "tv_weight": float(args.tv_weight),
        "drop_weight": float(args.drop_weight),
        "drop_weights": drop_weights,
        "token_micro_topk": int(args.token_micro_topk),
        "token_micro_min_affinity": float(args.token_micro_min_affinity),
        "token_micro_max_degree": int(args.token_micro_max_degree),
        "token_micro_max_segment_size": int(args.token_micro_max_segment_size),
        "token_micro_min_segment_size": int(args.token_micro_min_segment_size),
        "token_micro_min_density": float(args.token_micro_min_density),
        "token_micro_num_order_candidates": int(args.token_micro_num_order_candidates),
        "token_micro_eval_batches": int(args.token_micro_eval_batches),
        "token_micro_eval_batch_size": int(args.token_micro_eval_batch_size),
        "token_micro_random_suffixes": int(args.token_micro_random_suffixes),
        "token_micro_random_orders": int(args.token_micro_random_orders),
        "token_micro_min_pair_consistency": float(args.token_micro_min_pair_consistency),
        "token_micro_reverse_margin_threshold": float(args.token_micro_reverse_margin_threshold),
        "token_micro_random_margin_threshold": float(args.token_micro_random_margin_threshold),
        "token_micro_keep_singletons": bool(args.token_micro_keep_singletons),
        "token_micro_max_accepted_segments": int(args.token_micro_max_accepted_segments),
        "benchmark_num_levels": int(args.benchmark_num_levels),
        "early_stop_enabled": bool(args.early_stop_enabled),
        "early_stop_eval_split": str(args.early_stop_eval_split),
        "early_stop_min_pairs_after_margin": int(args.early_stop_min_pairs_after_margin),
        "early_stop_gain_eval_batches": int(args.early_stop_gain_eval_batches),
        "early_stop_gain_eval_batch_size": int(args.early_stop_gain_eval_batch_size),
        "early_stop_gain_random_contexts": int(args.early_stop_gain_random_contexts),
        "early_stop_gain_threshold": float(args.early_stop_gain_threshold),
        "early_stop_gain_p10_threshold": float(args.early_stop_gain_p10_threshold),
        "early_stop_gain_positive_rate": float(args.early_stop_gain_positive_rate),
        "early_stop_min_gain_pass_rate": float(args.early_stop_min_gain_pass_rate),
        "early_stop_filter_segments_by_gain": bool(args.early_stop_filter_segments_by_gain),
        "early_stop_policy_eval_batches": int(args.early_stop_policy_eval_batches),
        "early_stop_policy_eval_batch_size": int(args.early_stop_policy_eval_batch_size),
        "early_stop_policy_epsilon": float(args.early_stop_policy_epsilon),
        "early_stop_boundary_eval_batches": int(args.early_stop_boundary_eval_batches),
        "early_stop_boundary_eval_batch_size": int(args.early_stop_boundary_eval_batch_size),
        "early_stop_boundary_random_contexts": int(args.early_stop_boundary_random_contexts),
        "early_stop_boundary_gain_threshold": float(args.early_stop_boundary_gain_threshold),
        "early_stop_boundary_gain_p10_threshold": float(args.early_stop_boundary_gain_p10_threshold),
        "early_stop_boundary_positive_rate": float(args.early_stop_boundary_positive_rate),
        "early_stop_max_bad_boundary_ratio": float(args.early_stop_max_bad_boundary_ratio),
        "pair_mining_mode": str(args.pair_mining_mode),
        "attn_top_k": int(args.attn_top_k),
        "attn_top_ks": attn_top_ks,
        "attn_num_batches": int(args.attn_num_batches),
        "attn_batch_size": int(args.attn_batch_size),
        "attn_mode": str(args.attn_mode),
        "attn_symmetrize": str(args.attn_symmetrize),
        "attn_export_type": str(args.attn_export_type),
        "spectral_max_primary_matrices": int(args.spectral_max_primary_matrices),
        "spectral_max_primary_matrices_per_stage": spectral_max_primary_matrices_per_stage,
        "spectral_num_components": int(args.spectral_num_components),
        "spectral_component_pairs": str(args.spectral_component_pairs),
        "spectral_num_angles": int(args.spectral_num_angles),
        "spectral_num_angles_per_stage": spectral_num_angles_per_stage,
        "spectral_k_values": str(args.spectral_k_values),
        "spectral_k_values_per_stage": spectral_k_values_per_stage,
        "spectral_group_methods": str(args.spectral_group_methods),
        "spectral_threshold_percentile": float(args.spectral_threshold_percentile),
        "spectral_transform": str(args.spectral_transform),
        "spectral_temperature": float(args.spectral_temperature),
        "spectral_direction_lambdas": str(args.spectral_direction_lambdas),
        "spectral_directed_score_weight": float(args.spectral_directed_score_weight),
        "spectral_band_quality_weight": float(args.spectral_band_quality_weight),
        "spectral_score_adjacency_sym": str(args.spectral_score_adjacency_sym),
        "spectral_max_candidates": int(args.spectral_max_candidates),
        "spectral_export_top_candidates": int(args.spectral_export_top_candidates),
        "spectral_output_policy": str(args.spectral_output_policy),
        "spectral_rerank_num_candidates": int(args.spectral_rerank_num_candidates),
        "spectral_rerank_num_batches": int(args.spectral_rerank_num_batches),
        "spectral_rerank_batch_size": int(args.spectral_rerank_batch_size),
        "spectral_rerank_forward_batch_size": int(args.spectral_rerank_forward_batch_size),
        "spectral_rerank_prefix_k": int(args.spectral_rerank_prefix_k),
        "spectral_rerank_include_reverse": bool(args.spectral_rerank_include_reverse),
        "spectral_rerank_attention_weight": float(args.spectral_rerank_attention_weight),
        "spectral_rerank_full_loss_weight": float(args.spectral_rerank_full_loss_weight),
        "spectral_rerank_prefix_loss_weight": float(args.spectral_rerank_prefix_loss_weight),
        "spectral_rerank_signed_drop_weight": float(args.spectral_rerank_signed_drop_weight),
        "spectral_rerank_reverse_margin_weight": float(args.spectral_rerank_reverse_margin_weight),
        "spectral_rerank_positive_rate_weight": float(args.spectral_rerank_positive_rate_weight),
        "spectral_rerank_stability_weight": float(args.spectral_rerank_stability_weight),
        "spectral_seed": int(args.spectral_seed),
        "spectral_device": str(args.spectral_device),
        "spectral_dtype": str(args.spectral_dtype),
        "last_stage_results": previous_stage_results,
    }
    (benchmark_root / "runner_meta.json").write_text(
        json.dumps(runner_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved runner meta to {benchmark_root / 'runner_meta.json'}")


if __name__ == "__main__":
    main()
