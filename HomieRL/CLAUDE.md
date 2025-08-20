# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HomieRL is a reinforcement learning framework for humanoid robots, specifically designed for training locomotion and manipulation skills. The project implements a sophisticated RL training pipeline using Isaac Gym for simulation, focusing on the Unitree G1 humanoid robot. The framework enables robots to walk, squat, and perform upper-body movements under continuously changing poses.

## Key Components Architecture

### Core Modules Structure

**legged_gym/**: Main training environment and robot simulation
- `envs/base/legged_robot.py`: Core robot environment class with physics simulation, reward computation, and observation processing
- `envs/g1/g1_29dof_config.py`: G1 robot-specific configuration including joint mappings, PD gains, reward scales
- `utils/task_registry.py`: Central registry for environments and training configurations (line 153 contains hardcoded model path)
- `scripts/`: Entry point scripts for training, playing, and data collection

**rsl_rl/**: RL algorithm implementation
- `algorithms/him_ppo.py`: HIM-enhanced PPO algorithm with symmetry loss
- `modules/him_actor_critic.py`: Actor-critic networks with history integration
- `runners/him_on_policy_runner.py`: Training loop management and logging

## Development Commands

### Training
```bash
python legged_gym/legged_gym/scripts/train.py --task g1 --num_envs 4096 --headless --run_name my_policy --rl_device cuda:0 --sim_device cuda:0
```

### Playing/Testing Trained Models
```bash
python legged_gym/legged_gym/scripts/play.py --num_envs 1 --task g1 --resume --rl_device cpu --sim_device cpu
python legged_gym/legged_gym/scripts/play_data_collect.py --num_envs 1 --task g1 --resume --rl_device cuda:0 --sim_device cuda:0
```

### Model Export
```bash
python legged_gym/legged_gym/scripts/export_onnx.py
```

### Installation Commands
```bash
# Create environment and install Isaac Gym
conda create -n homierl python=3.8
conda activate homierl
cd path_to_isaac_gym/python && pip install -e .

# Install HomieRL dependencies
pip install -r requirements.txt
cd rsl_rl && pip install -e .
cd ../legged_gym && pip install -e .
```

## Key Framework Features

1. **Upper-body Pose Curriculum**: Gradual learning progression for complex upper-body movements
2. **Height Reward Tracking**: Precise height control for squatting behaviors via r_height and r_knee rewards
3. **Symmetry Utilization**: Data augmentation and L_sym loss for improved training efficiency
4. **Parallel Data Collection**: Support for large-scale parallel environment training (up to 4096 envs)

## Configuration System

- Environments inherit from `LeggedRobotCfg` base configuration
- Training configs inherit from `LeggedRobotCfgPPO`
- Robot-specific configurations define joint mappings, PD gains, reward scales, and physics parameters
- Task registration in `envs/__init__.py` links environments with their configurations

## Data and Logging

- Training logs saved to `legged_gym/logs/` with wandb or tensorboard support
- Collected episode data stored as HDF5 files in `logs/act_dataset/`
- Model checkpoints saved as `.pt` files with configurable intervals
- Resume path hardcoded in `task_registry.py:153` - update this for different model loading

## Dependencies and Environment

- Requires Isaac Gym Preview 4.0, NVIDIA GPU (RTX 2070+), Python 3.8
- Key libraries: torch, isaacgym, rsl_rl, matplotlib, wandb, onnxruntime
- LidarSensor integration for perception (manual path configuration required)
- Supports both CPU and GPU execution with device selection via command line args
- 总是用中文和我对话