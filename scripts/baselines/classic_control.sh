# CartPole-v1
OMP_NUM_THREADS=1 python ppo.py --gym-id CartPole-v1 --track --seed 1 --wandb-project-name ppo-mamba --cuda True --total-timesteps 500000
OMP_NUM_THREADS=1 python ppo.py --gym-id CartPole-v1 --track --seed 2 --wandb-project-name ppo-mamba --cuda True --total-timesteps 500000
OMP_NUM_THREADS=1 python ppo.py --gym-id CartPole-v1 --track --seed 3 --wandb-project-name ppo-mamba --cuda True --total-timesteps 500000
# Acrobot-v1
OMP_NUM_THREADS=1 python ppo.py --gym-id Acrobot-v1 --track --seed 1 --wandb-project-name ppo-mamba --cuda True --total-timesteps 500000
OMP_NUM_THREADS=1 python ppo.py --gym-id Acrobot-v1 --track --seed 2 --wandb-project-name ppo-mamba --cuda True --total-timesteps 500000
OMP_NUM_THREADS=1 python ppo.py --gym-id Acrobot-v1 --track --seed 3 --wandb-project-name ppo-mamba --cuda True --total-timesteps 500000
# MountainCar-v0
OMP_NUM_THREADS=1 python ppo.py --gym-id MountainCar-v0 --track --seed 1 --wandb-project-name ppo-mamba --cuda True --total-timesteps 500000
OMP_NUM_THREADS=1 python ppo.py --gym-id MountainCar-v0 --track --seed 2 --wandb-project-name ppo-mamba --cuda True --total-timesteps 500000
OMP_NUM_THREADS=1 python ppo.py --gym-id MountainCar-v0 --track --seed 3 --wandb-project-name ppo-mamba --cuda True --total-timesteps 500000