# RA-SAC UAV 2D Collision Avoidance

Code accompanying the manuscript on risk-aware Soft Actor-Critic (RA-SAC)
for two-dimensional UAV navigation under dynamic obstacles, disturbances,
execution error, perception uncertainty, and control delay.

This repository contains the cleaned publication implementation only. Local
training outputs, model checkpoints, manuscript files, downloaded literature,
development experiments, and historical audit archives are intentionally not
included.

## Methods

The publication comparison contains five methods:

- RA-SAC-v3.2-c
- SAC-Attention-v3
- SAC-MLP
- TD3-MLP
- PPO-MLP

All methods use the same Medium-ID environment and observation information.
The four evaluation-only shifts are Disturbance-OOD, Behavior-OOD,
Composition-OOD, and Combined-OOD.

## Installation

Python 3.10.20 was used for the publication experiments. Package versions are
pinned in `requirements.txt`; install a platform-appropriate PyTorch build if
GPU acceleration is required.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Repository layout

```text
agents/      RL implementations and observation encoding
configs/     frozen publication environments and training protocols
envs/        2-D UAV environment
scripts/     training, evaluation, publication analysis, and latency tools
tests/       portable tests that do not require private checkpoints
utils/       geometry, risk, safety, logging, and provenance utilities
```

## Quick verification

```bash
python -m unittest discover -s tests -v
```

## Training

RA-SAC-v3.2-c:

```bash
python scripts/train_sac_attention.py --config configs/v3_2_tuning/train_ra_sac_v3_2c_medium_seed_18207.yaml
```

Architecture-matched SAC-Attention baseline:

```bash
python scripts/train_sac_attention.py --config configs/train_sac_attention_v3_medium_seed_18207.yaml
```

MLP baselines:

```bash
python scripts/train_sac_mlp.py --config configs/train_sac_mlp_v3_medium_seed_18207.yaml
python scripts/train_td3_mlp.py --config configs/train_td3_mlp_v3_medium_seed_18207.yaml
python scripts/train_ppo_mlp.py --config configs/train_ppo_mlp_v3_medium_seed_18207.yaml
```

The publication training seeds are `18207`, `28207`, `38207`, `48207`, and
`58207`. Historical standalone RA-SAC configuration files were retained only
for seeds `18207` and `38207`; this limitation is recorded in the frozen
publication protocol and no replacement files have been fabricated.

## Evaluation

Example policy-only evaluation:

```bash
python scripts/evaluate_sac_attention_checkpoint.py \
  --checkpoint PATH/TO/CHECKPOINT.pt \
  --env-config configs/env_medium_v3_combined_ood.yaml \
  --train-config configs/v3_2_tuning/train_ra_sac_v3_2c_medium_seed_18207.yaml \
  --episodes 200
```

The frozen publication contract is documented in
`configs/publication_confirmation/protocol.yaml`. Generated outputs and model
weights are excluded from version control; publish them separately only when
their sharing and anonymity requirements are settled.

## License

No license has been selected yet. Add an institution-approved open-source
license before making the repository public.
