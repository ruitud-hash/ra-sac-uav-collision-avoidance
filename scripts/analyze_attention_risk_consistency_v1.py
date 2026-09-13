"""Frozen-checkpoint attention-risk consistency diagnostic (protocol v1)."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import heapq
from math import sqrt
from pathlib import Path
import shutil
import statistics
import sys
from typing import Iterable

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.observation import attention_observation
from envs import UAV2DEnv
from scripts.evaluate_sac_attention_checkpoint import build_agent
from utils.experiment_protocol import resolved_environment_config, stage1_arm_metadata


METHOD_SLUGS = {"RA-SAC": "ra_sac_v3_2_c", "SAC-Attention": "sac_attention_v3"}
SEED_METRICS = (
    "ra_delta_awr",
    "ra_delta_top_risk_mass",
    "ra_delta_top_match",
    "ra_content_awr",
    "ra_full_awr",
    "ra_content_top_risk_mass",
    "ra_full_top_risk_mass",
    "ra_content_top_match",
    "ra_full_top_match",
    "sac_awr",
    "sac_top_risk_mass",
    "sac_top_match",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="configs/publication_confirmation/attention_risk_consistency_protocol_v1.yaml",
    )
    parser.add_argument("--freeze-manifest")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def verify_freeze_manifest(path: Path) -> dict:
    manifest = load_yaml(path)
    failures = []
    for item in manifest["frozen_artifacts"]:
        artifact = resolve_project_path(item["path"])
        actual = sha256(artifact) if artifact.is_file() else "MISSING"
        if actual != item["sha256"]:
            failures.append(f"{item['path']}: expected {item['sha256']}, got {actual}")
    if failures:
        raise RuntimeError("Freeze-manifest verification failed:\n" + "\n".join(failures))
    return manifest


def unique_glob(pattern: str) -> Path:
    matches = sorted(PROJECT_ROOT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one match for {pattern!r}, found {len(matches)}")
    return matches[0]


def train_config(protocol: dict, method: str, training_seed: int) -> dict:
    pattern = protocol["checkpoints"][method]["train_config"]
    config = load_yaml(resolve_project_path(pattern.format(training_seed=training_seed)))
    stage1_arm_metadata(config)
    return config


def checkpoint_path(protocol: dict, method: str, training_seed: int) -> Path:
    return unique_glob(protocol["checkpoints"][method]["glob"].format(training_seed=training_seed))


def reference_episode_csv(environment: str, method: str, training_seed: int) -> Path:
    root = PROJECT_ROOT / "outputs/publication_confirmation_v1/formal_blind_batch1"
    env_slug = "medium_v3_unified_id" if environment == "Medium-ID" else "medium_v3_combined_ood"
    pattern = f"{env_slug}_{METHOD_SLUGS[method]}_seed_{training_seed}_policy_only_*/tables/episodes.csv"
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one reference episodes.csv for {environment}/{method}/{training_seed}, found {len(matches)}")
    return matches[0]


def load_reference_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = {int(row["seed"]): row for row in csv.DictReader(handle)}
    if len(rows) != 200:
        raise RuntimeError(f"Expected 200 reference episodes in {path}, found {len(rows)}")
    return rows


def selection_digest(
    namespace: str,
    environment: str,
    training_seed: int,
    episode_seed: int,
    generating_policy: str,
    timestep: int,
) -> str:
    payload = "|".join(
        [namespace, environment, str(training_seed), str(episode_seed), generating_policy, str(timestep)]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def keep_smallest(heap: list, digest: str, timestep: int, observation: np.ndarray, limit: int) -> None:
    rank = int(digest, 16)
    item = (-rank, -timestep, digest, timestep, observation.copy())
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def attention_weights(
    actor, observations: np.ndarray, include_prior: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoder = actor.encoder
    tensor = torch.as_tensor(observations, dtype=torch.float32, device=next(actor.parameters()).device)
    global_features, tokens, mask = encoder.unpack(tensor)
    query = encoder.query(encoder.global_net(global_features)).unsqueeze(1)
    keys = encoder.key(encoder.token_net(tokens))
    logits = torch.sum(query * keys, dim=-1) / sqrt(encoder.embed_dim)
    if include_prior:
        logits = encoder.attention_logits(query, keys, tokens)
    weights = torch.softmax(logits.masked_fill(mask <= 0.0, -1e9), dim=-1)
    no_valid = mask.sum(dim=-1, keepdim=True) <= 0.0
    weights = torch.where(no_valid, torch.zeros_like(weights), weights)
    return weights, tokens[:, :, 10], mask, logits


def attention_metrics(weights: torch.Tensor, risk: torch.Tensor, mask: torch.Tensor) -> dict[str, np.ndarray]:
    valid = mask > 0.0
    masked_risk = torch.where(valid, risk, torch.full_like(risk, -torch.inf))
    max_risk = masked_risk.max(dim=-1, keepdim=True).values
    top_risk = valid & (risk == max_risk)
    masked_weights = torch.where(valid, weights, torch.full_like(weights, -torch.inf))
    max_attention = masked_weights.max(dim=-1, keepdim=True).values
    top_attention = valid & (weights == max_attention)
    return {
        "awr": (weights * torch.where(valid, risk, torch.zeros_like(risk))).sum(dim=-1).cpu().numpy(),
        "top_risk_mass": (weights * top_risk).sum(dim=-1).cpu().numpy(),
        "top_match": (top_risk & top_attention).any(dim=-1).to(torch.float32).cpu().numpy(),
    }


def validate_attention(
    ra_actor,
    content_weights: torch.Tensor,
    full_weights: torch.Tensor,
    risks: torch.Tensor,
    mask: torch.Tensor,
    content_logits: torch.Tensor,
    full_logits: torch.Tensor,
) -> None:
    valid = mask > 0.0
    for label, weights in (("content", content_weights), ("full", full_weights)):
        if not torch.isfinite(weights[valid]).all() or (weights[valid] < 0.0).any():
            raise RuntimeError(f"Invalid {label} attention weights")
        if not torch.allclose(weights.sum(dim=-1), torch.ones(weights.shape[0], device=weights.device), atol=1e-6):
            raise RuntimeError(f"{label} attention weights do not sum to one")
    encoder = ra_actor.encoder
    if encoder.risk_bias_mode != "log1p_risk" or abs(float(encoder.risk_bias_scale) - 0.25) > 1e-12:
        raise RuntimeError("RA-SAC risk-bias configuration is not frozen log1p_risk at scale 0.25")
    observed = full_logits[valid] - content_logits[valid]
    expected = 0.25 * torch.log1p(risks[valid])
    if not torch.allclose(observed, expected, atol=2e-6, rtol=1e-5):
        raise RuntimeError("Reconstructed full/content logits do not match the analytic risk prior")


def evaluate_selected_states(ra_agent, sac_agent, observations: np.ndarray) -> dict[str, np.ndarray]:
    with torch.inference_mode():
        ra_content_w, risks, mask, ra_content_logits = attention_weights(
            ra_agent.actor, observations, include_prior=False
        )
        ra_full_w, full_risks, full_mask, ra_full_logits = attention_weights(
            ra_agent.actor, observations, include_prior=True
        )
        sac_w, sac_risks, sac_mask, _ = attention_weights(
            sac_agent.actor, observations, include_prior=False
        )
        if not torch.equal(mask, full_mask) or not torch.equal(mask, sac_mask):
            raise RuntimeError("Actor masks differ on the identical observation batch")
        if not torch.equal(risks, full_risks) or not torch.equal(risks, sac_risks):
            raise RuntimeError("Actor risk channels differ on the identical observation batch")
        validate_attention(
            ra_agent.actor,
            ra_content_w,
            ra_full_w,
            risks,
            mask,
            ra_content_logits,
            ra_full_logits,
        )
        if sac_agent.actor.encoder.risk_bias_mode != "none":
            raise RuntimeError("SAC-Attention contextual actor unexpectedly has a risk prior")
        content = attention_metrics(ra_content_w, risks, mask)
        full = attention_metrics(ra_full_w, risks, mask)
        sac = attention_metrics(sac_w, risks, mask)
    return {
        "ra_content_awr": content["awr"],
        "ra_full_awr": full["awr"],
        "ra_delta_awr": full["awr"] - content["awr"],
        "ra_content_top_risk_mass": content["top_risk_mass"],
        "ra_full_top_risk_mass": full["top_risk_mass"],
        "ra_delta_top_risk_mass": full["top_risk_mass"] - content["top_risk_mass"],
        "ra_content_top_match": content["top_match"],
        "ra_full_top_match": full["top_match"],
        "ra_delta_top_match": full["top_match"] - content["top_match"],
        "sac_awr": sac["awr"],
        "sac_top_risk_mass": sac["top_risk_mass"],
        "sac_top_match": sac["top_match"],
    }


def mean_rows(rows: Iterable[dict], keys: Iterable[str]) -> dict[str, float]:
    materialized = list(rows)
    return {key: statistics.fmean(float(row[key]) for row in materialized) for key in keys}


def aggregate(rows: list[dict], group_keys: tuple[str, ...], metric_keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    output = []
    for group, items in sorted(groups.items()):
        record = dict(zip(group_keys, group))
        record.update(mean_rows(items, metric_keys))
        record["n_lower_level_units"] = len(items)
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Tee:
    def __init__(self, path: Path):
        self.console = sys.stdout
        self.handle = path.open("w", encoding="utf-8")

    def write(self, text: str) -> int:
        self.console.write(text)
        self.handle.write(text)
        self.handle.flush()
        return len(text)

    def flush(self) -> None:
        self.console.flush()
        self.handle.flush()


def run_cell(
    protocol: dict,
    environment: str,
    training_seed: int,
    generating_policy: str,
    device: torch.device,
) -> tuple[list[dict], dict]:
    ra_config = train_config(protocol, "RA-SAC", training_seed)
    sac_config = train_config(protocol, "SAC-Attention", training_seed)
    generator_config = ra_config if generating_policy == "RA-SAC" else sac_config
    env_raw = load_yaml(resolve_project_path(protocol["environments"][environment]))
    env_config = resolved_environment_config(env_raw, generator_config)
    ra_agent = build_agent(env_config, ra_config, checkpoint_path(protocol, "RA-SAC", training_seed), device)
    sac_agent = build_agent(env_config, sac_config, checkpoint_path(protocol, "SAC-Attention", training_seed), device)
    ra_agent.actor.eval()
    sac_agent.actor.eval()
    generator = ra_agent if generating_policy == "RA-SAC" else sac_agent
    reference = load_reference_rows(reference_episode_csv(environment, generating_policy, training_seed))
    namespace = protocol["state_sampling"]["hash_namespace"]
    limit = int(protocol["state_sampling"]["maximum_states_per_episode"])
    start = int(protocol["blind_episode_seeds"]["start"])
    end = int(protocol["blind_episode_seeds"]["end"])
    state_rows = []
    eligible_episodes = 0
    eligible_decisions = 0

    for episode_index, episode_seed in enumerate(range(start, end + 1)):
        env = UAV2DEnv(deepcopy(env_config))
        obs_dict = env.reset(seed=episode_seed)
        heap = []
        timestep = 0
        done = False
        info = env.info()
        while not done:
            packed = attention_observation(obs_dict, env.world_size)
            mask = np.asarray(obs_dict["mask"], dtype=bool)
            risks = np.asarray(obs_dict["tokens"], dtype=np.float32)[mask, 10]
            if risks.size >= 2 and np.unique(risks).size >= 2:
                eligible_decisions += 1
                digest = selection_digest(namespace, environment, training_seed, episode_seed, generating_policy, timestep)
                keep_smallest(heap, digest, timestep, packed, limit)
            action = generator.act(packed, deterministic=True)
            obs_dict, _, done, info = env.step(action)
            timestep += 1

        expected = reference[episode_seed]
        if info["outcome"] != expected["outcome"] or int(info["steps"]) != int(expected["steps"]):
            raise RuntimeError(
                f"Replay mismatch {environment}/{training_seed}/{generating_policy}/{episode_seed}: "
                f"got {info['outcome']} at {info['steps']} steps; expected {expected['outcome']} at {expected['steps']}"
            )
        selected = sorted(
            [(item[2], item[3], item[4]) for item in heap],
            key=lambda item: item[0],
        )
        if selected:
            eligible_episodes += 1
            observations = np.stack([item[2] for item in selected])
            metrics = evaluate_selected_states(ra_agent, sac_agent, observations)
            for index, (digest, selected_timestep, _) in enumerate(selected):
                row = {
                    "environment": environment,
                    "training_seed": training_seed,
                    "episode_seed": episode_seed,
                    "generating_policy": generating_policy,
                    "timestep": selected_timestep,
                    "selection_sha256": digest,
                }
                row.update({key: float(values[index]) for key, values in metrics.items()})
                state_rows.append(row)
        if (episode_index + 1) % 50 == 0:
            print(f"  {episode_index + 1}/200 episodes replayed", flush=True)

    coverage = {
        "environment": environment,
        "training_seed": training_seed,
        "generating_policy": generating_policy,
        "episodes_replayed": end - start + 1,
        "episodes_integrity_matched": end - start + 1,
        "eligible_episodes": eligible_episodes,
        "eligible_decision_states_before_cap": eligible_decisions,
        "selected_states": len(state_rows),
        "maximum_states_per_episode": limit,
        "integrity_status": "PASS",
    }
    return state_rows, coverage


def run_cell_worker(payload: tuple[dict, str, int, str, str]) -> tuple[list[dict], dict]:
    protocol, environment, training_seed, generating_policy, device_name = payload
    return run_cell(protocol, environment, training_seed, generating_policy, torch.device(device_name))


def worker_init() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def self_test() -> None:
    observations = [np.array([value], dtype=np.float32) for value in range(20)]
    heap = []
    for timestep, observation in enumerate(observations):
        digest = selection_digest("test", "env", 1, 2, "policy", timestep)
        keep_smallest(heap, digest, timestep, observation, 10)
    selected = sorted(item[2] for item in heap)
    expected = sorted(selection_digest("test", "env", 1, 2, "policy", t) for t in range(20))[:10]
    assert selected == expected
    risk = torch.tensor([[0.1, 0.9]])
    mask = torch.ones_like(risk)
    metrics = attention_metrics(torch.tensor([[0.4, 0.6]]), risk, mask)
    assert np.allclose(metrics["awr"], [0.58])
    assert np.allclose(metrics["top_risk_mass"], [0.6])
    assert np.allclose(metrics["top_match"], [1.0])
    print("self-test PASS")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.freeze_manifest or not args.output_dir:
        raise ValueError("Formal execution requires --freeze-manifest and --output-dir")
    protocol_path = resolve_project_path(args.protocol).resolve()
    freeze_path = resolve_project_path(args.freeze_manifest).resolve()
    output_dir = resolve_project_path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Append-only output directory already exists: {output_dir}")
    protocol = load_yaml(protocol_path)
    if protocol["status"] != "frozen_before_execution":
        raise RuntimeError("Protocol is not marked frozen_before_execution")
    verify_freeze_manifest(freeze_path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Frozen device {args.device} is unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(protocol_path, output_dir / protocol_path.name)
    shutil.copy2(freeze_path, output_dir / freeze_path.name)
    tee = Tee(output_dir / "stdout.log")
    sys.stdout = tee
    started = datetime.now(timezone.utc)
    print(f"analysis_id={protocol['analysis_id']}")
    print(f"started_utc={started.isoformat()}")
    print(f"device={device}; torch={torch.__version__}; cuda={torch.version.cuda}")

    all_state_rows = []
    coverage_rows = []
    try:
        cells = [
            (protocol, environment, int(training_seed), generating_policy, str(device))
            for environment in protocol["environments"]
            for training_seed in protocol["training_seeds"]
            for generating_policy in protocol["trajectory_pool"]["generating_policies"]
        ]
        if args.workers == 1:
            for payload in cells:
                _, environment, training_seed, generating_policy, _ = payload
                print(f"cell={environment}/{training_seed}/{generating_policy}", flush=True)
                rows, coverage = run_cell_worker(payload)
                all_state_rows.extend(rows)
                coverage_rows.append(coverage)
        else:
            if args.device != "cpu":
                raise ValueError("Parallel replay is permitted only for the frozen CPU runtime")
            print(f"parallel_workers={args.workers}", flush=True)
            with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as executor:
                futures = {executor.submit(run_cell_worker, payload): payload[1:4] for payload in cells}
                for future in as_completed(futures):
                    environment, training_seed, generating_policy = futures[future]
                    rows, coverage = future.result()
                    all_state_rows.extend(rows)
                    coverage_rows.append(coverage)
                    print(
                        f"cell_complete={environment}/{training_seed}/{generating_policy}; "
                        f"selected_states={len(rows)}",
                        flush=True,
                    )

        episode_rows = aggregate(
            all_state_rows,
            ("environment", "training_seed", "generating_policy", "episode_seed"),
            SEED_METRICS,
        )
        generator_rows = aggregate(
            episode_rows,
            ("environment", "training_seed", "generating_policy"),
            SEED_METRICS,
        )
        seed_rows = aggregate(generator_rows, ("environment", "training_seed"), SEED_METRICS)
        summary_rows = []
        for environment in protocol["environments"]:
            environment_rows = [row for row in seed_rows if row["environment"] == environment]
            if len(environment_rows) != 5:
                raise RuntimeError(f"Expected five seed aggregates for {environment}, found {len(environment_rows)}")
            for metric in SEED_METRICS:
                values = [float(row[metric]) for row in environment_rows]
                summary_rows.append(
                    {
                        "environment": environment,
                        "metric": metric,
                        "n_training_seeds": len(values),
                        "mean": statistics.fmean(values),
                        "median": statistics.median(values),
                        "sample_sd": statistics.stdev(values),
                        "minimum": min(values),
                        "maximum": max(values),
                    }
                )

        write_csv(output_dir / "state_metrics.csv", all_state_rows)
        write_csv(output_dir / "episode_aggregates.csv", episode_rows)
        write_csv(output_dir / "generating_policy_aggregates.csv", generator_rows)
        write_csv(output_dir / "seed_aggregates.csv", seed_rows)
        write_csv(output_dir / "five_seed_summary.csv", summary_rows)
        write_csv(output_dir / "coverage_and_integrity.csv", coverage_rows)
        finished = datetime.now(timezone.utc)
        manifest = {
            "analysis_id": protocol["analysis_id"],
            "classification": protocol["classification"],
            "status": "PASS",
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "device": str(device),
            "python": sys.version,
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "selected_states": len(all_state_rows),
            "episode_aggregates": len(episode_rows),
            "training_seed_aggregates": len(seed_rows),
            "protocol_sha256": sha256(protocol_path),
            "freeze_manifest_sha256": sha256(freeze_path),
            "outputs": {
                path.name: sha256(path)
                for path in sorted(output_dir.glob("*.csv"))
            },
        }
        with (output_dir / "run_manifest.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(manifest, handle, sort_keys=False)
        print(f"status=PASS; selected_states={len(all_state_rows)}; elapsed_s={(finished - started).total_seconds():.1f}")
    except Exception as error:
        with (output_dir / "run_manifest.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                {
                    "analysis_id": protocol["analysis_id"],
                    "status": "FAIL",
                    "started_utc": started.isoformat(),
                    "failed_utc": datetime.now(timezone.utc).isoformat(),
                    "error": str(error),
                },
                handle,
                sort_keys=False,
            )
        raise
    finally:
        sys.stdout = tee.console
        tee.handle.close()


if __name__ == "__main__":
    main()
