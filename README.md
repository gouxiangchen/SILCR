# SILCR
source code for paper ["Self-Imitation Learning in Sparse Reward Settings"](https://arxiv.org/pdf/2010.06962.pdf)

# Requirements

- tensorflow (for tensorboard logging)
- pytorch (>=1.0, 1.0.1 used in my experiments)
- gym
- mujoco

# Usage

`python main.py` to run the training process of SILCR in Humanoid-v2 environment with default hyperparameters

or `python main.py --env_id 1` to switch to another environments.

Only 0-4 env_id supported, including Swimmer-v2, Hopper-v2, HalfCheetah-v2, Walker2d-v2, Humanoid-v2.

The code is easy to read and if you want to test other environments, you may just add the environment name into `env_dict` in `main.py`.