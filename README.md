# 🚀 COMP4082 Autonomous Robotic System Coursework

This repository contains the codebase for reproducing the experimental results of our Autonomous Robotic System coursework, including CartPole and Acrobot continual reinforcement learning experiments.

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