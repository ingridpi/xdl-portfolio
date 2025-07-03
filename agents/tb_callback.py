from stable_baselines3.common.callbacks import BaseCallback


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        try:
            self.logger.record(
                key="train/reward", value=self.locals["rewards"][0]
            )
        except KeyError:
            self.logger.record(
                key="train/reward", value=self.locals["reward"][0]
            )
        except Exception as e:
            print(f"Error recording reward: {e}")
        return True
