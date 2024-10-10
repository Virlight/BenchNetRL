# Proximal Policy Optimization with Mamba Integration

## Goal
The aim of this project is to implement and analyze [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) using different architecture approximations (MLPs and [Mamba](https://arxiv.org/abs/2312.00752)) to improve the performance of RL in long-horizon environments with focus on metrics like sample efficiency, average return, and computational efficiency.

## Repository Inspiration
The implementation of this project is based on two primary repositories:

- [pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail): This repository offers a comprehensive implementation of PPO and other RL algorithms.
  
- [ppo-implementation-details](https://github.com/vwxyzjn/ppo-implementation-details): Key implementation details from here help refine the PPO implementation.

## Prerequisites
- **Python version**: 3.8+
- **PyTorch**: Version 1.12+ for compatibility with Mamba
- **CUDA**: For GPU acceleration (CUDA 11.6+ required for Mamba)
  
   Additional prerequisites for AMD cards and ROCm will be required if using an AMD setup (see [Mamba README](https://github.com/state-spaces/mamba)).

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/SafeRL-Lab/Mamba-RL.git
   cd Mamba-RL
   ```

2. **Create virtual environment**: 
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. **Install the project dependencies**: 
    ```bash
    pip install -r requirements.txt
    ```

## To-Do List

### 1. Core PPO Implementation
- [ ] Implement PPO with MLPs.
    - [X] Check on Discrete Spaces.
    - [ ] Check on Continuous Spaces.
    - [ ] Check on Key-to-Door environment.
    - [ ] Check on Atari.

### 2. Mamba Integration
- [ ] Integrate Mamba SSM layers into the PPO implementation.
    - [ ] Check on Discrete Spaces.
    - [ ] Check on Continuous Spaces.
    - [ ] Check on Key-to-Door environment.
    - [ ] Check on Atari.

### 3. Benchmarking and Results
- [ ] Set up experiment tracking with Tensorboard or Weights & Biases.
- [ ] Log results and create visual comparisons (learning curves, bar charts for performance metrics).
