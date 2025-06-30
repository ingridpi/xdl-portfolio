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
