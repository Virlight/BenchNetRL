# ALE/Breakout-v5 or ALE/Pong-v5 (same hyperparameters)
# ========================================
# Transformer-XL
python ppo_trxl.py \
    --gym-id ALE/Breakout-v5 \
    --seed 5 \
    --total-timesteps 10000000 \
    --num-envs 16 \
    --num-minibatches 8 \
    --trxl-dim 512 \
    --trxl-memory-length 64 \
    --trxl-num-layers 1 \
    --trxl-num-heads 2 \
    --final-lr 0 \
    --init-ent-coef 0.01 \
    --final-ent-coef 0.01 \
    --wandb-project-name atari-bench \
    --exp-name ppo_trxl \
    --track

# Mamba v1
python ppo_mamba.py \
    --gym-id ALE/Breakout-v5 \
    --mamba-version v1 \
    --seed 8 \
    --total-timesteps 10000000 \
    --num-envs 16 \
    --num-minibatches 8 \
    --expand 1 \
    --hidden-dim 450 \
    --wandb-project-name atari-bench \
    --exp-name ppo_mamba \
    --track

# Mamba v2
python ppo_mamba.py \
    --gym-id ALE/Breakout-v5 \
    --mamba-version v2 \
    --seed 5 \
    --total-timesteps 10000000 \
    --num-envs 16 \
    --num-minibatches 8 \
    --expand 1 \
    --d-state 64 \
    --d-conv 4 \
    --hidden-dim 512 \
    --wandb-project-name atari-bench \
    --exp-name ppo_mamba2 \
    --track

# LSTM
python ppo_lstm.py \
    --gym-id ALE/Breakout-v5 \
    --seed 8 \
    --total-timesteps 10000000 \
    --num-envs 16 \
    --num-minibatches 8 \
    --rnn-type lstm \
    --rnn-hidden-dim 256 \
    --wandb-project-name atari-bench \
    --exp-name ppo_lstm \
    --track

# GRU
python ppo_lstm.py \
    --gym-id ALE/Breakout-v5 \
    --seed 4 \
    --total-timesteps 10000000 \
    --num-envs 16 \
    --num-minibatches 8 \
    --rnn-type gru \
    --rnn-hidden-dim 256 \
    --wandb-project-name atari-bench \
    --exp-name ppo_gru \
    --track

# PPO with 4-frame stack
python ppo.py \
    --gym-id ALE/Breakout-v5 \
    --seed 7 \
    --total-timesteps 10000000 \
    --num-envs 16 \
    --num-minibatches 8 \
    --hidden-dim 512 \
    --frame-stack 4 \
    --wandb-project-name atari-bench \
    --exp-name ppo_4 \
    --track

# PPO with 1-frame stack
python ppo.py \
    --gym-id ALE/Breakout-v5 \
    --seed 2 \
    --total-timesteps 10000000 \
    --num-envs 16 \
    --num-minibatches 8 \
    --hidden-dim 512 \
    --frame-stack 1 \
    --wandb-project-name atari-bench \
    --exp-name ppo_1 \
    --track
