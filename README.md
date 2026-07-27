# MotrixArena S1 VBot Reinforcement Learning

[中文](README.zh-CN.md)

VBot quadruped navigation project for MotrixArena S1, built with MotrixSim and SKRL PPO. The project focuses on long-course navigation, obstacle handling, segmented curriculum training, checkpoint transfer, and policy replay.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![RL](https://img.shields.io/badge/RL-SKRL-FF6F00)
![Simulation](https://img.shields.io/badge/Simulation-MotrixSim-2E8B57)
![Backend](https://img.shields.io/badge/Backend-JAX%20%7C%20PyTorch-FF6F00)

## Demo / Replay

This repository includes trained policy checkpoints and replay scripts. After installing dependencies and pulling Git LFS assets, replay a saved policy with:

```bash
uv run scripts/play.py --env vbot_navigation_section001 --policy checkpoints/best_agent001.pickle
uv run scripts/play.py --env vbot_navigation_full --policy checkpoints/best_agent.pickle
```

For visual inspection without loading a trained policy:

```bash
uv run scripts/view.py --env vbot_navigation_section001 --num-envs 1
```

## Highlights

- Competition result: MotrixArena S1 / VBot Competition third prize, ranked 8 / 30.
- Built a complete PPO workflow from environment registration and reward design to training, checkpointing, replay, and visualization.
- Split the long competition route into segment environments, then transferred checkpoints across stages for curriculum learning.
- Supported high-throughput training with 2048 parallel environments by default.
- Kept both JAX and PyTorch training backends available through `--train-backend`.
- Preserved trained checkpoints under `checkpoints/` for direct replay and continued training.

## Task Overview

The full competition course has three scoring stages, with a maximum score of 25 points.

| Stage | Route | Objective | Score |
| --- | --- | --- | --- |
| 1 | Section 011 -> 012 | Pass the rolling-ball area by either avoiding the balls or touching them without falling | 10-15 |
| 2 | Section 012 -> 013 | Traverse random terrain and reach the Chinese-knot target | 5 |
| 3 | Finish area | Execute the final celebration behavior | 5 |

Nominal navigation waypoints:

```text
(0, 7.5) -> (0, 24.3) -> (0, 32.3)
```

## Approach

| Challenge | Method | Outcome |
| --- | --- | --- |
| The full course is long and difficult to learn end-to-end | Decompose the route into `section001`, `section011`, `section012`, `section013`, and `full` environments | Reduced exploration difficulty and enabled staged policy development |
| Rolling balls, uneven terrain, and finish-zone behavior make policies fragile | Add forward-progress, arrival, finish, fall-penalty, and stage-specific rewards | Improved stability across competition segments |
| Sparse stage rewards can trap the policy in local behavior | Combine dense navigation rewards with checkpoint and stage rewards | Improved sample efficiency before full-course training |
| Long RL training needs repeatable entry points | Provide unified `train.py`, `play.py`, and `view.py` scripts | Made experiments reproducible and replayable |

## Quick Start

### 1. Clone and Pull Assets

```bash
git clone https://github.com/Logic-TARS/motrixarena-competition-S1.git
cd motrixarena-competition-S1
git lfs pull
```

### 2. Install Dependencies

The repository uses `uv` with a workspace containing `motrix_envs` and `motrix_rl`.

```bash
# JAX backend
uv sync --all-packages --extra skrl-jax

# Or PyTorch backend
uv sync --all-packages --extra skrl-torch
```

### 3. Train

```bash
# Train a segment
uv run scripts/train.py --env vbot_navigation_section001 --num-envs 2048

# Continue from a trained policy
uv run scripts/train.py --env vbot_navigation_section011 --policy checkpoints/best_agent001.pickle

# Train the full course
uv run scripts/train.py --env vbot_navigation_full --num-envs 2048
```

### 4. Replay

```bash
# Replay a specific checkpoint
uv run scripts/play.py --env vbot_navigation_section011 --policy checkpoints/best_agent011.pickle

# Use PyTorch or JAX explicitly for training
uv run scripts/train.py --env vbot_navigation_full --train-backend jax
uv run scripts/train.py --env vbot_navigation_full --train-backend torch
```

Useful script arguments:

| Argument | Purpose | Default |
| --- | --- | --- |
| `--num-envs` | Number of parallel environments | `2048` |
| `--train-backend` | RL backend, `jax` or `torch` | `jax` |
| `--policy` | Pretrained checkpoint for transfer learning or replay | `None` |
| `--checkpoint-interval` | Checkpoint save interval in timesteps | Config default |
| `--seed` / `--rand-seed` | Fixed or randomized seed | `None` / `False` |
| `--env-cfg` | Environment config overrides, for example `curriculum_from_001=True` | None |

## Environments

Core VBot competition environments used by the training and replay commands:

| Environment | Purpose |
| --- | --- |
| `vbot_navigation_flat` | Basic flat-ground navigation |
| `vbot_navigation_section001` | Initial section training |
| `vbot_navigation_section011` | Competition section 011 |
| `vbot_navigation_section012` | Competition section 012 |
| `vbot_navigation_section013` | Competition section 013 |
| `vbot_navigation_section011_012` | Combined section 011 -> 012 transfer task |
| `vbot_navigation_long_course` | Long-route navigation |
| `vbot_navigation_full` | Full competition route |

These names are implemented and registered under `motrix_envs/src/motrix_envs/navigation/vbot/`.

## Results and Checkpoints

| Item | Value |
| --- | --- |
| Competition | MotrixArena S1 / VBot Competition |
| Result | Third prize |
| Rank | 8 / 30 |
| Algorithm | PPO |
| Training framework | SKRL |
| Simulation platform | MotrixSim |
| Default parallel environments | 2048 |

Saved policies:

| Checkpoint | Intended use |
| --- | --- |
| `checkpoints/best_agent001.pickle` | Initial section policy |
| `checkpoints/best_agent011.pickle` | Section 011 policy |
| `checkpoints/best_agent012.pickle` | Section 012 policy |
| `checkpoints/best_agent013.pickle` | Section 013 policy |
| `checkpoints/best_agent.pickle` | Full-course / final policy |

## Repository Structure

```text
.
├── motrix_envs/              # Simulation environments
│   └── src/motrix_envs/
│       ├── navigation/vbot/  # VBot competition navigation environments
│       ├── locomotion/vbot/  # VBot locomotion environments
│       └── basic/            # Basic example environments
├── motrix_rl/                # RL training package
│   └── src/motrix_rl/skrl/
│       ├── jax/train/ppo.py
│       └── torch/train/ppo.py
├── scripts/                  # Training, replay, and visualization entry points
├── checkpoints/              # Saved policy checkpoints
├── docs/                     # Competition and project documentation
├── logs/                     # Training logs
├── archives/                 # Historical experiment archives
├── pyproject.toml            # uv workspace configuration
└── uv.lock
```

## Notes

This repository is organized as a portfolio and reproducibility record for the MotrixArena S1 competition. Upstream simulation assets, robot models, and related framework components remain the property of their respective maintainers.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for license and attribution details.
