# 🚀 COMP4082 Autonomous Robotic System Coursework

This repository contains the codebase for reproducing the experimental results of the paper "Parseval-Regularized PPO for Continual Control: Geometry vs Catastrophic Forgetting".

# Overview
This repository implements a multi-task reinforcement learning training loop using Proximal Policy Optimization (PPO) with optional Parseval regularization.
It supports:
- Multi-task switching with dynamic physics parameters
- Periodic evaluation on all tasks
- Continual learning metrics (forgetting score, performance matrix, stability tracking)
- TensorBoard logging
- Optional Weights & Biases integration
- Model checkpointing and resuming
- Both standard PPO and Parseval-PPO architectures

The main entry point of the system is main.py, which loads environments, manages tasks, trains a PPO agent, evaluates periodically, and logs all results.

## 📦 Features
✔ 1. PPO Agent (Standard or Parseval)

The training loop supports two network types:
- "base" → default PPO MLP actor–critic
- "parseval" → uses orthogonality-regularized layers for stability

✔ 2. Multi-Task Environment Switching

Every task_switch_episode_interval episodes:
- The script switches to the next task
- Applies new physics via TaskManager
- Logs transitions

✔ 3. Full Evaluation Loop

On each task switch:
- Evaluate the agent on all tasks
- Compute: 
    - mean return
    - success rate
    - forgetting score
    - average performance
    - convergence detection
    
Results are logged to TensorBoard.

✔ 4. Continual Learning Metrics
The script automatically tracks:
- Best performance per task
- Forgetting matrix
- Stability hits (successes needed for convergence)

✔ 5. Training Metrics & Logging
- Logged to TensorBoard:
- episodic return / length
- policy loss, value loss, entropy
- KL divergence
- learning rate
- explained variance
- Parseval statistics (if enabled)
- per-task evaluation metrics
- task switching timeline

## 🏗️ Repository Structure
```
.
├── main.py                          # Main PPO + Multi-task training script
├── agent.py                         # PPOAgent implementation (base + Parseval)
├── tasks.py                         # TaskManager to dynamically modify task physics
├── eval.py                          # Evaluation utilities
├── utils.py                         # Environment creation, logging helpers
├── config/
│   └── task_config.yaml             # Physics configuration for each task
├── automate_scripts/
│   └── run_acrobot.sh               # automation script for acrobot-v1
│   └── run_cartpole.sh              # automation script for cartpole-v1
├── analysis_scripts/
│   └── analyse_model.py             # tensorfile post-processing
│   └── plot_compare_ci_multi.py     # evaluate model performance
│   └── plot_tb_scalars.py           # plot performance graphs
└── runs/                            # Automatically saved logs + models
```

# 📦 Installation
```
git clone https://github.com/swyannnn/Autonomous_Robotic_System_Coursework.git
cd Autonomous_Robotic_System_Coursework

# Create the conda environment
conda create -n ARS python=3.10
conda activate ARS

# Install dependencies
pip install -r requirements.txt
```

# 🧪 Reproducing Paper Results
Run the following scripts to reproduce the CartPole and Acrobot experiments:
```
bash automate_scripts/run_cartpole.sh
bash automate_scripts/run_acrobot.sh
```

# 📁 Output Directory

All outputs — including:
- TensorBoard event files
- saved model checkpoints
- logs 
are stored automatically in:
```
runs/
```

# 📊 Visualize Training with TensorBoard
```
tensorboard --logdir runs
```

Open your browser and navigate to:

👉 http://localhost:6006

# 📊 To do analysis on experiment results
```
analysis_scripts/plot_tb_scalars.py
```
This script is to extract all the specified logged values from tensorfile, and save as numpy and json files.

```
analysis_scripts/analyse_model.py
```
This script is to evaluate model performance of the specified model file.


```
analysis_scripts/plot_compare_ci_multi.py
```
This script is to plot performance graphs, that you saw in the paper.