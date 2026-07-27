# MotrixArena S1 VBot 强化学习项目

[English](README.md)

本项目是 MotrixArena S1 / VBot Competition 的四足机器人导航强化学习项目，基于 MotrixSim 仿真平台与 SKRL PPO 训练框架，重点解决长赛道导航、障碍物通过、分段课程学习、checkpoint 迁移和策略回放问题。

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![RL](https://img.shields.io/badge/RL-SKRL-FF6F00)
![Simulation](https://img.shields.io/badge/Simulation-MotrixSim-2E8B57)
![Backend](https://img.shields.io/badge/Backend-JAX%20%7C%20PyTorch-FF6F00)

## Demo / 回放

仓库包含已训练好的策略 checkpoint 和回放脚本。安装依赖并拉取 Git LFS 资产后，可以直接回放策略：

```bash
uv run scripts/play.py --env vbot_navigation_section001 --policy checkpoints/best_agent001.pickle
uv run scripts/play.py --env vbot_navigation_full --policy checkpoints/best_agent.pickle
```

如果只想查看仿真环境，不加载训练策略：

```bash
uv run scripts/view.py --env vbot_navigation_section001 --num-envs 1
```

## 项目亮点

- 比赛成绩：MotrixArena S1 / VBot Competition 三等奖，排名 8 / 30。
- 完成从环境注册、奖励设计、PPO 训练、checkpoint 保存到策略回放和可视化的完整链路。
- 将长赛道拆分为多个分段环境，通过阶段训练和 checkpoint 迁移降低训练难度。
- 默认支持 2048 个并行环境，提升强化学习采样效率。
- 通过 `--train-backend` 保留 JAX 与 PyTorch 两种训练后端。
- 在 `checkpoints/` 中保留训练好的策略，可用于直接回放或继续训练。

## 比赛任务

完整赛道包含三个计分阶段，满分 25 分。

| 阶段 | 路段 | 目标 | 分值 |
| --- | --- | --- | --- |
| 1 | Section 011 -> 012 | 通过滚动球区域，可选择避开滚球，或触碰滚球并保持不摔倒 | 10-15 |
| 2 | Section 012 -> 013 | 穿越随机地形，到达“中国结”终点 | 5 |
| 3 | 终点区域 | 完成终点庆祝动作 | 5 |

标称导航航点：

```text
(0, 7.5) -> (0, 24.3) -> (0, 32.3)
```

## 技术方案

| 问题 | 方法 | 效果 |
| --- | --- | --- |
| 完整赛道较长，端到端训练收敛慢 | 拆分为 `section001`、`section011`、`section012`、`section013`、`full` 等环境 | 降低探索难度，形成分阶段训练流程 |
| 滚动球、复杂地形和终点动作容易导致策略失稳 | 设计前进、到达、终点、摔倒惩罚和阶段性奖励 | 提升各赛段策略稳定性 |
| 阶段性奖励较稀疏，策略容易停留在局部行为 | 结合稠密导航奖励、checkpoint 奖励和阶段奖励 | 提高采样效率，减少从零训练完整赛道的难度 |
| 训练与回放需要可复现 | 统一封装 `train.py`、`play.py` 和 `view.py` | 便于复现实验、回放策略和展示结果 |

## 快速开始

### 1. 克隆仓库并拉取资产

```bash
git clone https://github.com/Logic-TARS/motrixarena-competition-S1.git
cd motrixarena-competition-S1
git lfs pull
```

### 2. 安装依赖

仓库使用 `uv` 管理依赖，工作区包含 `motrix_envs` 和 `motrix_rl`。

```bash
# JAX 后端
uv sync --all-packages --extra skrl-jax

# 或 PyTorch 后端
uv sync --all-packages --extra skrl-torch
```

### 3. 训练

```bash
# 训练单个赛段
uv run scripts/train.py --env vbot_navigation_section001 --num-envs 2048

# 基于已训练策略继续训练
uv run scripts/train.py --env vbot_navigation_section011 --policy checkpoints/best_agent001.pickle

# 训练完整赛道
uv run scripts/train.py --env vbot_navigation_full --num-envs 2048
```

### 4. 回放

```bash
# 回放指定 checkpoint
uv run scripts/play.py --env vbot_navigation_section011 --policy checkpoints/best_agent011.pickle

# 显式选择 JAX 或 PyTorch 后端训练
uv run scripts/train.py --env vbot_navigation_full --train-backend jax
uv run scripts/train.py --env vbot_navigation_full --train-backend torch
```

常用脚本参数：

| 参数 | 用途 | 默认值 |
| --- | --- | --- |
| `--num-envs` | 并行环境数量 | `2048` |
| `--train-backend` | 强化学习后端，`jax` 或 `torch` | `jax` |
| `--policy` | 用于迁移训练或回放的预训练 checkpoint | `None` |
| `--checkpoint-interval` | checkpoint 保存间隔 | 配置默认值 |
| `--seed` / `--rand-seed` | 固定或随机种子 | `None` / `False` |
| `--env-cfg` | 环境配置覆盖，例如 `curriculum_from_001=True` | None |

## 环境列表

训练和回放命令使用的核心 VBot 比赛环境：

| 环境名 | 用途 |
| --- | --- |
| `vbot_navigation_flat` | 基础平地导航 |
| `vbot_navigation_section001` | 初始赛段训练 |
| `vbot_navigation_section011` | 比赛 Section 011 |
| `vbot_navigation_section012` | 比赛 Section 012 |
| `vbot_navigation_section013` | 比赛 Section 013 |
| `vbot_navigation_section011_012` | Section 011 -> 012 合并迁移任务 |
| `vbot_navigation_long_course` | 长路线导航 |
| `vbot_navigation_full` | 完整比赛路线 |

这些环境位于并注册于 `motrix_envs/src/motrix_envs/navigation/vbot/`。

## 结果与 Checkpoints

| 项目 | 内容 |
| --- | --- |
| 比赛 | MotrixArena S1 / VBot Competition |
| 结果 | 三等奖 |
| 排名 | 8 / 30 |
| 算法 | PPO |
| 训练框架 | SKRL |
| 仿真平台 | MotrixSim |
| 默认并行环境数 | 2048 |

已保存策略：

| Checkpoint | 用途 |
| --- | --- |
| `checkpoints/best_agent001.pickle` | 初始赛段策略 |
| `checkpoints/best_agent011.pickle` | Section 011 策略 |
| `checkpoints/best_agent012.pickle` | Section 012 策略 |
| `checkpoints/best_agent013.pickle` | Section 013 策略 |
| `checkpoints/best_agent.pickle` | 完整赛道 / 最终策略 |

## 仓库结构

```text
.
├── motrix_envs/              # 仿真环境包
│   └── src/motrix_envs/
│       ├── navigation/vbot/  # VBot 比赛导航环境
│       ├── locomotion/vbot/  # VBot 行走环境
│       └── basic/            # 基础示例环境
├── motrix_rl/                # 强化学习训练包
│   └── src/motrix_rl/skrl/
│       ├── jax/train/ppo.py
│       └── torch/train/ppo.py
├── scripts/                  # 训练、回放、可视化入口
├── checkpoints/              # 已保存策略 checkpoint
├── docs/                     # 比赛与项目文档
├── logs/                     # 训练日志
├── archives/                 # 历史实验归档
├── pyproject.toml            # uv workspace 配置
└── uv.lock
```

## 说明

本仓库作为 MotrixArena S1 比赛的作品集展示与复现实验记录。上游仿真资产、机器人模型和相关框架组件归其原维护者所有。

许可证和归属信息见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。
