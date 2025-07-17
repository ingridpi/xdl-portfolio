# Stable Baselines DRL Models
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

MODELS = {
    "a2c": A2C,
    "ppo": PPO,
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC,
}

# Training parameters for stock trading task
MODEL_KWARGS_STOCK = {
    "a2c": {
        "n_steps": 5,
        "ent_coef": 0.01,
        "learning_rate": 0.0007,
    },
    "ppo": {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "batch_size": 64,
    },
    "ddpg": {
        "batch_size": 128,
        "buffer_size": 50000,
        "learning_rate": 0.001,
    },
    "td3": {
        "batch_size": 100,
        "buffer_size": 1000000,
        "learning_rate": 0.001,
    },
    "sac": {
        "batch_size": 64,
        "buffer_size": 1000000,
        "learning_rate": 0.0001,
        "learning_starts": 1000,
        "ent_coef": "auto_0.1",
    },
}

# Training parameters for portfolio optimisation task
MODEL_KWARGS_PORTFOLIO = {
    "a2c": {
        "n_steps": 10,
        "ent_coef": 0.005,
        "learning_rate": 0.0004,
    },
    "ppo": {
        "n_steps": 2048,
        "ent_coef": 0.005,
        "learning_rate": 0.001,
        "batch_size": 128,
    },
    "ddpg": {
        "batch_size": 128,
        "buffer_size": 50000,
        "learning_rate": 0.001,
    },
    "td3": {
        "batch_size": 100,
        "buffer_size": 1000000,
        "learning_rate": 0.001,
    },
    "sac": {
        "batch_size": 128,
        "buffer_size": 1000000,
        "learning_rate": 0.0003,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    },
}


# Training visualisation config
train_visualisation_config = {
    "a2c": {
        "x": "time/iterations",
        "y": [
            "train/reward",
            "train/policy_loss",
            "train/entropy_loss",
            "train/value_loss",
        ],
        "title": [
            "Iterations vs Reward",
            "Iterations vs Policy Loss",
            "Iterations vs Entropy loss",
            "Iterations vs Value loss",
        ],
    },
    "ppo": {
        "x": "time/iterations",
        "y": [
            "train/reward",
            "train/policy_gradient_loss",
            "train/entropy_loss",
            "train/value_loss",
        ],
        "title": [
            "Iterations vs Reward",
            "Iterations vs Policy Gradient Loss",
            "Iterations vs Entropy loss",
            "Iterations vs Value loss",
        ],
    },
    "ddpg": {
        "x": "time/episodes",
        "y": [
            "train/reward",
            "train/actor_loss",
            "train/critic_loss",
        ],
        "title": [
            "Episodes vs Reward",
            "Episodes vs Actor Loss",
            "Episodes vs Critic Loss",
        ],
    },
    "td3": {
        "x": "time/episodes",
        "y": [
            "train/reward",
            "train/actor_loss",
            "train/critic_loss",
        ],
        "title": [
            "Episodes vs Reward",
            "Episodes vs Actor Loss",
            "Episodes vs Critic Loss",
        ],
    },
    "sac": {
        "x": "time/episodes",
        "y": [
            "train/reward",
            "train/actor_loss",
            "train/critic_loss",
            "train/ent_coef_loss",
        ],
        "title": [
            "Episodes vs Reward",
            "Episodes vs Actor Loss",
            "Episodes vs Critic Loss",
            "Episodes vs Entropy Coefficient Loss",
        ],
    },
}
